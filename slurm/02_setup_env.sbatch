#!/bin/bash
#SBATCH --job-name=setup-env-v2
#SBATCH --partition=staging
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=slurm/logs/setup_env_v2_%j.out
#SBATCH --error=slurm/logs/setup_env_v2_%j.err

echo "Setting up venv environment — $(date)"

module purge
module load 2024
module load CUDA/12.6.0
module load Python/3.12.3-GCCcore-13.3.0

cd ~/comicspap-project

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

echo " Installing PyTorch with CUDA 12.6..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

echo " Installing HuggingFace stack..."
pip install transformers peft trl datasets accelerate

echo " Installing QLoRA + utilities..."
pip install bitsandbytes qwen-vl-utils pillow gradio wandb

echo " Verifying..."
python -c "
import torch
print(f'Python:        {__import__(\"sys\").version}')
print(f'PyTorch:       {torch.__version__}')
print(f'CUDA in torch: {torch.version.cuda}')

import transformers
print(f'Transformers:  {transformers.__version__}')

import peft
print(f'PEFT:          {peft.__version__}')

import trl
print(f'TRL:           {trl.__version__}')

import bitsandbytes
print(f'BitsAndBytes:  {bitsandbytes.__version__}')

import datasets
print(f'Datasets:      {datasets.__version__}')

import accelerate
print(f'Accelerate:    {accelerate.__version__}')

print()
print('All packages installed successfully!')
"

echo "Done — $(date)"
