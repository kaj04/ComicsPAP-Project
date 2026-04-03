import os
import sys
import torch
import json
import collections
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from datasets import load_dataset

# 1. Path Setup (replicando la logica del notebook)
venv_path = "/gpfs/home3/scur2635/comicspap-project/venv/lib/python3.12/site-packages"
project_root = "/gpfs/home3/scur2635/comicspap-project/official_repo"
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Importiamo il tuo processore personalizzato dal tuo progetto
from data_utils import SingleImagePickAPanel

def main():
    # --- CONFIGURAZIONE PATH (Da cambiare via SLURM se vuoi) ---
    adapter_path = os.getenv("ADAPTER_PATH", "/scratch-shared/scur2635/comicspap/checkpoints/qlora_v2_r16_sequence_filling_20260402_230902/checkpoint-600")
    base_model_id = "Qwen/Qwen2.5-VL-7B-Instruct"
    output_dir = "results/evaluations"
    os.makedirs(output_dir, exist_ok=True)
    exp_name = os.path.basename(os.path.normpath(adapter_path))

    # --- 2. CARICAMENTO MODELLO (Logica Notebook) ---
    print(f"Loading base model: {base_model_id}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        base_model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        ignore_mismatched_sizes=True # RISOLVE IL RUNTIME ERROR
    )

    print(f"Injecting LoRA weights from: {adapter_path}")
    model = PeftModel.from_pretrained(
        model, 
        adapter_path,
        device_map={"": 0}
    )
    model.eval()
    
    # Il processor va caricato dall'adapter se hai salvato config specifiche, o dalla base
    processor = AutoProcessor.from_pretrained(base_model_id)

    # --- 3. SETUP DATASET & PROCESSOR ---
    os.environ['HF_DATASETS_CACHE'] = "/scratch-shared/scur2635/comicspap/hf_cache"
    ds_test = load_dataset("VLR-CVC/ComicsPAP", "sequence_filling", split=args.split)
    
    font_path = "/home/scur2635/comicspap-project/assets/DejaVuSans.ttf"
    panel_processor = SingleImagePickAPanel(font_path=font_path)

    system_prompt = (
        "Pick A Panel Task: In the image you have two rows of comic panels. "
        "The top row is the context and the bottom row is the options. "
        "The context row has a missing panel marked with a question mark. "
        "Choose the option that best fills the gap in the sequence. "
        "Respond with only the option number."
    )

    # --- 4. INFERENZA ---
    results = []
    predictions = []
    ground_truths = []

    print(f"Starting evaluation on {len(ds_test)} samples...")
    for i in tqdm(range(len(ds_test))):
        sample = ds_test[i]
        
        # Composizione immagine (Tua logica PanelProcessor)
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

        # Pre-processing
        text_inp = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inp, video_inp = process_vision_info(messages)
        inputs = processor(
            text=[text_inp], 
            images=image_inp, 
            videos=video_inp, 
            padding=True, 
            return_tensors="pt"
        ).to(model.device)

        # Generazione
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        
        # Decoding
        gen_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        pred = processor.batch_decode(gen_trimmed, skip_special_tokens=True)[0].strip()
        
        gt = str(sample['solution_index'])
        is_correct = (pred == gt)

        # Salvataggio dati
        predictions.append(pred)
        ground_truths.append(gt)
        results.append({
            "sample_id": sample['sample_id'],
            "prediction": pred,
            "ground_truth": gt,
            "is_correct": is_correct,
            "caption": caption
        })

    # --- 5. REPORTING ---
    accuracy = sum(1 for p, g in zip(predictions, ground_truths) if p == g) / len(predictions)
    dist = dict(collections.Counter(predictions))
    
    # Save CSV
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, f"{exp_name}_results.csv"), index=False)

    # Save JSON Summary
    summary = {
        "accuracy": accuracy,
        "distribution": dist,
        "total": len(ds_test),
        "results": results
    }
    with open(os.path.join(output_dir, f"{exp_name}_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    print(f"\n✅ Evaluation Finished!")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Results saved in {output_dir}")

if __name__ == "__main__":
    main()