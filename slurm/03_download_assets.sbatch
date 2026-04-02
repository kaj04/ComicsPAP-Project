#!/bin/bash
#SBATCH --job-name=download-assets
#SBATCH --partition=staging
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm/logs/download_%j.out
#SBATCH --error=slurm/logs/download_%j.err

echo "Downloading model and dataset — $(date)"

module purge
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
source ~/comicspap-project/venv/bin/activate

export HF_HOME=/scratch-shared/$USER/comicspap/hf_cache
export TRANSFORMERS_CACHE=/scratch-shared/$USER/comicspap/hf_cache

python << 'EOF'
import time
import os

cache_dir = os.environ["HF_HOME"]

# 1. Download model
print("=" * 50)
print("Downloading Qwen2.5-VL-7B-Instruct (~16 GB)...")
print("=" * 50)
start = time.time()

from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    cache_dir=cache_dir
)
print(f"  Processor downloaded ({time.time()-start:.0f}s)")

model = AutoModelForImageTextToText.from_pretrained(
    "Qwen/Qwen2.5-VL-7B-Instruct",
    cache_dir=cache_dir,
    torch_dtype="auto"
)
print(f"  Model downloaded ({time.time()-start:.0f}s)")
print(f"  Model class: {type(model).__name__}")

del model, processor
import gc
gc.collect()

# 2. Download dataset
print()
print("=" * 50)
print("Downloading ComicsPAP sequence_filling...")
print("=" * 50)
start = time.time()

from datasets import load_dataset

for split in ["train", "val", "test"]:
    ds = load_dataset(
        "VLR-CVC/ComicsPAP",
        "sequence_filling",
        split=split,
        cache_dir=cache_dir
    )
    print(f"  {split}: {len(ds)} samples ({time.time()-start:.0f}s)")

# 3. Quick inspection
print()
print("=" * 50)
print("Dataset inspection")
print("=" * 50)
sample = ds[0]
print(f"  Sample ID:       {sample['sample_id']}")
print(f"  Context images:  {len(sample['context'])}")
print(f"  Option images:   {len(sample['options'])}")
print(f"  Solution index:  {sample['solution_index']}")
print(f"  Task type:       {sample['task_type']}")

print()
print("All downloads complete!")
EOF

echo ""
du -sh /scratch-shared/$USER/comicspap/hf_cache/
echo "Done — $(date)"
