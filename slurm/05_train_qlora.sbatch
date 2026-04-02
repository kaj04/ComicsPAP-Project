#!/bin/bash
#SBATCH --job-name=qlora-train
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --output=slurm/logs/train_qlora_%j.out
#SBATCH --error=slurm/logs/train_qlora_%j.err

echo "Starting QLoRA training — $(date)"

module purge
module load 2024
module load CUDA/12.6.0
module load Python/3.12.3-GCCcore-13.3.0
source ~/comicspap-project/venv/bin/activate

export HF_HOME=/scratch-shared/$USER/comicspap/hf_cache
export TRANSFORMERS_CACHE=/scratch-shared/$USER/comicspap/hf_cache
export WANDB_PROJECT="ComicsPAP"
export WANDB_API_KEY=$(grep password ~/.netrc | awk '{print $2}')

cd ~/comicspap-project

python src/train_qlora.py \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --skill sequence_filling \
    --lora_rank 32 \
    --lora_alpha 64 \
    --max_steps 500 \
    --eval_steps 100 \
    --batch_size 2 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-4 \
    --output_dir /scratch-shared/$USER/comicspap/checkpoints

echo "Done — $(date)"
