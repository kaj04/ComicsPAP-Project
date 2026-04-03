#!/bin/bash
#SBATCH --job-name=eval_comicspap
#SBATCH --output=slurm/logs/eval_%j.out
#SBATCH --error=slurm/logs/eval_%j.err
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=02:00:00

# Load modules and activate environment
module load 2024
module load Python/3.12.3-GCCcore-13.3.0
source venv/bin/activate

# CONFIGURATION: Change this path to evaluate different LoRA weights
ADAPTER_PATH="/scratch-shared/scur2635/comicspap/checkpoints/qlora_v2_r16_sequence_filling_20260402_230902/checkpoint-600"
OUTPUT_DIR="results/validation_best_model"

echo "Starting evaluation for: $ADAPTER_PATH"

export HF_HOME="/scratch-shared/scur2635/comicspap/hf_cache"
# export HF_DATASETS_OFFLINE=0
export TRANSFORMERS_OFFLINE=1

python src/evaluate.py \
    --adapter_path "$ADAPTER_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --split "val"

echo "Evaluation finished."