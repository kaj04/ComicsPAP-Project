#!/bin/bash
#SBATCH --job-name=test-gpu
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --output=slurm/logs/test_gpu_%j.out
#SBATCH --error=slurm/logs/test_gpu_%j.err

echo "GPU + QLoRA test — $(date)"

module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
module load CUDA/12.6.0
source ~/comicspap-project/venv/bin/activate

export HF_HOME=/scratch-shared/$USER/comicspap/hf_cache

python << 'EOF'
import torch
import time

print("=" * 50)
print("GPU Info")
print("=" * 50)
print(f"  PyTorch:   {torch.__version__}")
print(f"  CUDA:      {torch.version.cuda}")
print(f"  GPU:       {torch.cuda.get_device_name(0)}")
mem = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"  VRAM:      {mem:.1f} GB")

print()
print("=" * 50)
print("Loading model in 4-bit (QLoRA)...")
print("=" * 50)
start = time.time()

from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    quantization_config=bnb_config,
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

elapsed = time.time() - start
gpu_mem = torch.cuda.memory_allocated() / 1e9
print(f"  Loaded in {elapsed:.1f}s")
print(f"  GPU memory used: {gpu_mem:.1f} GB")
print(f"  Model class: {type(model).__name__}")

print()
print("=" * 50)
print("Testing LoRA adapter...")
print("=" * 50)

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=32,
    lora_alpha=64,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

gpu_mem_after = torch.cuda.memory_allocated() / 1e9
print(f"  GPU memory after LoRA: {gpu_mem_after:.1f} GB")

print()
print("=" * 50)
print("Testing dataset access...")
print("=" * 50)

from datasets import load_dataset
ds = load_dataset("VLR-CVC/ComicsPAP", "sequence_filling", split="val")
print(f"  Val set: {len(ds)} samples")
sample = ds[0]
print(f"  Context:  {len(sample['context'])} panels")
print(f"  Options:  {len(sample['options'])} choices")
print(f"  Answer:   {sample['solution_index']}")

print()
print("=" * 50)
print("ALL TESTS PASSED — ready for training!")
print("=" * 50)
EOF

echo "Done — $(date)"
