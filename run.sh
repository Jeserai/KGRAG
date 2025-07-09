#!/bin/bash

#SBATCH --job-name=kgrag
#SBATCH --gres=gpu:A6000:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --partition=debug
#SBATCH --output=logs/kgrag_%j.out
#SBATCH --error=logs/kgrag_%j.err
#SBATCH --pty bash

set -euo pipefail

# Load conda
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate kgrag

# hugging face caches
export HF_HOME=/data/user_data/yuanguan/.hf_cache
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
#export HF_HUB_OFFLINE=1  # Optional

# Run pipeline (built-in test docs; pass --input to use your own)
python tests/main.py --config configs/config.yaml 