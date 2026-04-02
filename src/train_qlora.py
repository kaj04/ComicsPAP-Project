"""
ComicsPAP — QLoRA Fine-tuning Script
Fine-tunes Qwen2.5-VL-7B on Sequence Filling task using QLoRA (4-bit).
Based on the official ComicsPAP sft.py with QLoRA modifications.

Improvements over the paper:
1. QLoRA (4-bit) instead of LoRA (bf16)
2. Higher LoRA rank (32 vs 8)
3. More target modules (7 vs 2)
4. Single-task specialization (sequence_filling only)
5. Completion-only loss (prompt tokens masked)
"""

import os
import sys
import torch
import json
import argparse
from datetime import datetime

sys.path.insert(0, os.path.expanduser("~/comicspap-project/official_repo"))

from trl import SFTConfig, SFTTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration
from datasets import load_dataset, load_from_disk
from data_utils import SingleImagePickAPanel
from qwen_vl_utils import process_vision_info


def parse_args():
    parser = argparse.ArgumentParser(description="ComicsPAP QLoRA Training")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--skill", type=str, default="sequence_filling")
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--max_steps", type=int, default=5000)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--output_dir", type=str, default="results/qlora_r32_sf")
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
    Implements completion-only loss: the loss is computed ONLY on the
    assistant response tokens ("answer: X"), masking the prompt.
    This follows best practices from QLoRA (Dettmers et al., 2023)
    and standard SFT post-training pipelines (PyTorch Blog, 2025).
    """

    def __init__(self, processor, process_vision_info):
        self.processor = processor
        self.process_vision_info = process_vision_info
        self.prompt = (
            "Pick A Panel Task: In the image you have two rows of comic panels. "
            "The top row is the context and the bottom row is the options. "
            "The context row has a missing panel marked with a question mark. "
            "Choose the option that best fits the missing panel. "
            "You must return your final answer as a number with "
            "'answer: <your answer here>'\n\n"
        )

    def __call__(self, batch):
        messages = []
        for sample in batch:
            caption = sample.get('previous_panel_caption', '')
            msg = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {"type": "text", "text": f"context: {caption}"},
                        {"type": "image", "image": sample["single_image"]},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": f"answer: {sample['solution_index']}"},
                    ],
                }
            ]
            messages.append(msg)

        # Tokenize full conversation (prompt + response)
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

        # === COMPLETION-ONLY LOSS ===
        # Tokenize prompt-only (without assistant response) to find where response starts
        prompt_only_messages = []
        for sample in batch:
            caption = sample.get('previous_panel_caption', '')
            msg = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {"type": "text", "text": f"context: {caption}"},
                        {"type": "image", "image": sample["single_image"]},
                    ],
                },
            ]
            prompt_only_messages.append(msg)

        prompt_only_text = self.processor.apply_chat_template(
            prompt_only_messages, tokenize=False, add_generation_prompt=True
        )
        prompt_only_tokens = self.processor.tokenizer(
            prompt_only_text, padding=True, padding_side='left', return_tensors="pt"
        )
        prompt_lengths = prompt_only_tokens["input_ids"].shape[1]

        # Create labels: -100 for everything except the response
        labels = inputs["input_ids"].clone()
        # Mask padding tokens
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        # Mask image tokens
        image_tokens = [151652, 151653, 151654, 151655]
        for token_id in image_tokens:
            labels[labels == token_id] = -100
        # Mask all prompt tokens (keep only completion)
        labels[:, :prompt_lengths] = -100

        inputs["labels"] = labels

        return inputs


def main():
    args = parse_args()

    # Set seeds
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # Experiment name
    experiment_name = f"qlora_r{args.lora_rank}_{args.skill}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = os.path.join(args.output_dir, experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print(f"ComicsPAP QLoRA Training")
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
    print(f"  Output:         {output_dir}")
    print("=" * 60)

    # ---- Load model with 4-bit quantization (QLoRA) ----
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
        # attn_implementation="flash_attention_2",  # removed: not installed
    )

    processor = AutoProcessor.from_pretrained(args.model)

    gpu_mem = torch.cuda.memory_allocated() / 1e9
    print(f"  Model loaded — GPU memory: {gpu_mem:.1f} GB")

    # ---- Prepare for QLoRA training ----
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

    # ---- Load dataset ----
    print(f"\nLoading dataset ({args.skill})...")
    train_dataset = build_dataset_single_image(
        args.skill, "train", args.dataset_cache, args.font_path
    )
    print(f"  Train samples: {len(train_dataset)}")

    # ---- Setup collator ----
    collator = QwenTrainCollator(processor, process_vision_info)

    # ---- Training config ----
    training_args = SFTConfig(
        output_dir=output_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant",
        logging_steps=10,
        save_strategy="steps",
        save_steps=args.eval_steps,
        bf16=True,
        tf32=True,
        max_grad_norm=0.3,
        warmup_ratio=0.05,
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

    # ---- Train ----
    print("\nStarting training...")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=collator,
    )

    trainer.train()

    # ---- Save ----
    print(f"\nSaving model to {output_dir}")
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)

    # Save training config
    config = vars(args)
    config["experiment_name"] = experiment_name
    config["trainable_params"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    config["total_params"] = sum(p.numel() for p in model.parameters())
    with open(os.path.join(output_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print("\nTraining complete!")
    print(f"  Model saved to: {output_dir}")
    print(f"  Check wandb for training curves")


if __name__ == "__main__":
    main()