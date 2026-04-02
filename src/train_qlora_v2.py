"""
ComicsPAP — QLoRA Fine-tuning Script (v2 - Paper Collator)
Fine-tunes Qwen2.5-VL-7B on Sequence Filling task using QLoRA (4-bit).
Uses the ORIGINAL collator from the ComicsPAP paper (no prompt masking).

Improvements over the paper:
1. QLoRA (4-bit) instead of LoRA (bf16) — less GPU memory
2. Higher LoRA rank (32 vs 8) — more capacity
3. More target modules (7 vs 2) — more expressivity
4. Single-task specialization (sequence_filling only)
"""

import os
import sys
import torch
import json
import argparse
from datetime import datetime
import numpy as np

sys.path.insert(0, os.path.expanduser("~/comicspap-project/official_repo"))

from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from datasets import load_dataset, load_from_disk
from data_utils import SingleImagePickAPanel
from qwen_vl_utils import process_vision_info


def parse_args():
    parser = argparse.ArgumentParser(description="ComicsPAP QLoRA Training v2")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--skill", type=str, default="sequence_filling")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_steps", type=int, default=500)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--output_dir", type=str, default="results/qlora_v2")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Path to resume from")
    parser.add_argument("--dataset_cache", type=str, default="/scratch-shared/scur2635/comicspap/dataset_cache")
    parser.add_argument("--font_path", type=str, default=os.path.expanduser("~/comicspap-project/assets/DejaVuSans.ttf"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def build_dataset_single_image(skill, split, cache_dir, font_path):
    """Load dataset and convert to single-image format."""
    single_image_path = os.path.join(cache_dir, f"{skill}_{split}_single_image")

    if os.path.exists(single_image_path):
        print(f"  Loading cached {split} dataset from {single_image_path}")
        return load_from_disk(single_image_path)

    print(f"  Processing {split} dataset to single-image format...")
    dataset = load_dataset("VLR-CVC/ComicsPAP", skill, split=split)

    processor = SingleImagePickAPanel(font_path=font_path)
    dataset = dataset.map(
        processor.map_to_single_image,
        batched=True,
        batch_size=32,
        remove_columns=['context', 'options']
    )

    os.makedirs(cache_dir, exist_ok=True)
    dataset.save_to_disk(single_image_path)
    print(f"  Saved to {single_image_path}")

    return dataset


class QwenTrainCollator:
    """
    Collator for Qwen2.5-VL training on single-image format.
    Implements completion-only loss: masks the entire prompt and computes
    loss ONLY on assistant response tokens.
    """

    ASSISTANT_START_SEQUENCE = [151644, 77091, 198]  # <|im_start|>assistant\n
    IMAGE_TOKENS = {151652, 151653, 151654, 151655}

    def __init__(self, processor, process_vision_info):
        self.processor = processor
        self.process_vision_info = process_vision_info
        self.prompt = (
            "Pick A Panel Task: In the image you have two rows of comic panels. "
            "The top row is the context and the bottom row is the options. "
            "The context row has a missing panel marked with a question mark. "
            "Choose the option that best fills the gap in the sequence. "
            "Respond with only the option number."
        )

    def _find_assistant_start(self, input_ids):
        """Find the index right after <|im_start|>assistant\\n in a 1D token list."""
        seq = self.ASSISTANT_START_SEQUENCE
        seq_len = len(seq)
        for i in range(len(input_ids) - seq_len + 1):
            if input_ids[i:i + seq_len] == seq:
                return i + seq_len
        return len(input_ids)  # fallback: mask everything

    def __call__(self, batch):
        messages = []
        for sample in batch:
            caption = sample.get('previous_panel_caption', '')
            msg = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {"type": "text", "text": f"Previous panel caption: {caption}" if caption else ""},
                        {"type": "image", "image": sample["single_image"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": str(sample['solution_index'])},
                    ],
                }
            ]
            messages.append(msg)

        # Tokenize full conversation
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = self.process_vision_info(messages)
        inputs = self.processor(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            padding_side='left',
            truncation=True,
            return_tensors="pt",
        )

        # === COMPLETION-ONLY MASKING ===
        labels = inputs["input_ids"].clone()

        for i in range(labels.shape[0]):
            token_list = inputs["input_ids"][i].tolist()

            # Find last occurrence of <|im_start|>assistant\n
            # (last because there's only one assistant turn, but safe against system/user)
            response_start = 0
            seq = self.ASSISTANT_START_SEQUENCE
            seq_len = len(seq)
            for j in range(len(token_list) - seq_len + 1):
                if token_list[j:j + seq_len] == seq:
                    response_start = j + seq_len

            # Mask everything before the response
            labels[i, :response_start] = -100

        # Mask padding tokens
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        # Mask image tokens
        for token_id in self.IMAGE_TOKENS:
            labels[labels == token_id] = -100

        inputs["labels"] = labels
        return inputs


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    experiment_name = f"qlora_v2_r{args.lora_rank}_{args.skill}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("ComicsPAP QLoRA Training — v2 (Paper Collator)")
    print("=" * 60)
    print(f"  Model:          {args.model}")
    print(f"  Skill:          {args.skill}")
    print(f"  LoRA rank:      {args.lora_rank}")
    print(f"  LoRA alpha:     {args.lora_alpha}")
    print(f"  Max steps:      {args.max_steps}")
    print(f"  Batch size:     {args.batch_size}")
    print(f"  Grad accum:     {args.gradient_accumulation_steps}")
    print(f"  Effective BS:   {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Learning rate:  {args.learning_rate}")
    print(f"  Loss:           full sequence (paper approach)")
    print(f"  Output:         {output_dir}")
    print("=" * 60)

    # Load model in 4-bit (QLoRA)
    print("\nLoading model in 4-bit (QLoRA)...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.model,
        quantization_config=bnb_config,
        device_map="auto",
    )

    processor = AutoProcessor.from_pretrained(args.model)

    gpu_mem = torch.cuda.memory_allocated() / 1e9
    print(f"  Model loaded — GPU memory: {gpu_mem:.1f} GB")

    # Prepare for QLoRA
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                         "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    print(f"\nLoading dataset ({args.skill})...")
    train_dataset = build_dataset_single_image(
        args.skill, "train", args.dataset_cache, args.font_path
    )
    print(f"  Train samples: {len(train_dataset)}")
    
    eval_dataset = build_dataset_single_image(
        args.skill, "val", args.dataset_cache, args.font_path
    )
    print(f"  Val samples: {len(eval_dataset)}")

    # Collator (paper's original approach)
    collator = QwenTrainCollator(processor, process_vision_info)
    
    # Training config
    training_args = SFTConfig(
        output_dir=output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        logging_steps=10,
        eval_strategy="steps",   
        eval_steps=args.eval_steps,    
        eval_accumulation_steps=1, 
        per_device_eval_batch_size=1,
        save_strategy="steps",
        save_steps=args.eval_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.10,
        push_to_hub=False,
        report_to="wandb",
        run_name=experiment_name,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        dataloader_num_workers=8,
        dataloader_persistent_workers=True,
        seed=args.seed,
    )

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = np.argmax(logits, axis=-1)
        
        mask = labels != -100
        labels = labels[mask]
        predictions = predictions[mask]
        
        acc = (predictions == labels).mean() if len(labels) > 0 else 0
        return {"accuracy": acc}

    # Train
    print("\nStarting training...")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Save
    print(f"\nSaving model to {output_dir}")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    # Save config
    config = vars(args)
    config["experiment_name"] = experiment_name
    config["loss_type"] = "full_sequence (paper approach)"
    config["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config["total_params"] = sum(p.numel() for p in model.parameters())
    with open(os.path.join(output_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("\nTraining complete!")
    print(f"  Model saved to: {output_dir}")
    print(f"  Check wandb for training curves")


if __name__ == "__main__":
    main()