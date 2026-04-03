import os
import sys
import torch
import json
import argparse
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from datasets import load_dataset

# Setup dei path per Snellius
venv_path = "/gpfs/home3/scur2635/comicspap-project/venv/lib/python3.12/site-packages"
project_root = "/gpfs/home3/scur2635/comicspap-project/official_repo"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data_utils import SingleImagePickAPanel

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-VL LoRA on ComicsPAP")
    parser.add_argument("--adapter_path", type=str, required=True, help="Path to LoRA checkpoint")
    parser.add_argument("--output_dir", type=str, default="results/evaluations", help="Directory to save results")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to evaluate (val or test)")
    return parser.parse_args()

def main():
    # 1. Inizializza gli argomenti (Risolve il NameError)
    args = parse_args()
    
    # 2. Configurazione Directory e Nomi
    os.makedirs(args.output_dir, exist_ok=True)
    # Estraiamo il nome del checkpoint per differenziare i file di output
    exp_name = os.path.basename(os.path.normpath(args.adapter_path))
    base_model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

    # --- CARICAMENTO MODELLO ---
    print(f"Loading base model: {base_model_id}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        ignore_mismatched_sizes=True 
    )

    print(f"Injecting LoRA weights from: {args.adapter_path}")
    model = PeftModel.from_pretrained(
        model, 
        args.adapter_path,
        device_map={"": 0}
    )
    model.eval()
    
    processor = AutoProcessor.from_pretrained(base_model_id)

    # --- SETUP DATASET & PROCESSOR ---
    os.environ['HF_DATASETS_CACHE'] = "/scratch-shared/scur2635/comicspap/hf_cache"
    
    print(f"Loading dataset split: {args.split}")
    ds_to_eval = load_dataset("VLR-CVC/ComicsPAP", "sequence_filling", split=args.split)
    
    font_path = "/home/scur2635/comicspap-project/assets/DejaVuSans.ttf"
    panel_processor = SingleImagePickAPanel(font_path=font_path)

    system_prompt = (
        "Pick A Panel Task: In the image you have two rows of comic panels. "
        "The top row is the context and the bottom row is the options. "
        "The context row has a missing panel marked with a question mark. "
        "Choose the option that best fills the gap in the sequence. "
        "Respond with only the option number."
    )

    # --- INFERENZA ---
    results = []
    predictions = []
    ground_truths = []

    print(f"Starting evaluation on {len(ds_to_eval)} samples...")
    for i in tqdm(range(len(ds_to_eval))):
        sample = ds_to_eval[i]
        
        # Pre-processing Immagine
        batch_examples = {k: [v] for k, v in sample.items()}
        processed_batch = panel_processor.map_to_single_image(batch_examples)
        img = processed_batch['single_image'][0]
        
        caption = sample.get('previous_panel_caption', '')
        messages = [{
            "role": "user",
            "content": [
                {"type": "text", "text": system_prompt},
                {"type": "text", "text": f"Previous panel caption: {caption}" if caption else ""},
                {"type": "image", "image": img},
            ],
        }]

        # Pre-processing Testo/Vision
        text_inp = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inp, video_inp = process_vision_info(messages)
        inputs = processor(
            text=[text_inp], 
            images=image_inp, 
            videos=video_inp, 
            padding=True, 
            return_tensors="pt"
        ).to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        
        # Decoding
        gen_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        pred = processor.batch_decode(gen_trimmed, skip_special_tokens=True)[0].strip()
        
        gt = str(sample['solution_index'])
        is_correct = (pred == gt)

        predictions.append(pred)
        ground_truths.append(gt)
        results.append({
            "sample_id": sample['sample_id'],
            "prediction": pred,
            "ground_truth": gt,
            "is_correct": is_correct,
            "caption": caption
        })

    # --- REPORTING ---
    accuracy = sum(1 for p, g in zip(predictions, ground_truths) if p == g) / len(predictions)
    dist = dict(collections.Counter(predictions))
    
    # Salvataggio CSV (Dettagliato)
    df = pd.DataFrame(results)
    csv_filename = f"{exp_name}_{args.split}_results.csv"
    df.to_csv(os.path.join(args.output_dir, csv_filename), index=False)

    # Salvataggio JSON (Riassunto per Web App)
    summary = {
        "experiment": exp_name,
        "split": args.split,
        "accuracy": accuracy,
        "distribution": dist,
        "total_samples": len(ds_to_eval),
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    json_filename = f"{exp_name}_{args.split}_summary.json"
    with open(os.path.join(args.output_dir, json_filename), "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\n✅ Evaluation Finished!")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Results saved in {args.output_dir}")

if __name__ == "__main__":
    main()