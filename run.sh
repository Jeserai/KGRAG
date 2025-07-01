#!/bin/bash

#SBATCH --job-name=kgrag_pipeline
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/kgrag_%j.out
#SBATCH --error=logs/kgrag_%j.err

# =============================================================================
# KGRAG SLURM Submission Script
#
# Description:
#   This script runs the KGRAG pipeline on a SLURM-managed cluster node.
#   It activates the pre-configured Conda environment and executes the main
#   Python script.
#
# Pre-requisites:
#   1. A Conda environment named 'KGRAG' must exist and have all
#      dependencies from requirements.txt installed.
#   2. All required Hugging Face models must be pre-downloaded in the cache.
#   3. The 'logs' and 'results' directories should exist.
# =============================================================================

set -e # Exit immediately if a command exits with a non-zero status.
set -u # Treat unset variables as an error.

# --- Job Information ---
echo "============================================"
echo "KGRAG Pipeline Job Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Submission Time: $(date)"
echo "============================================"

# --- Configuration ---
PROJECT_DIR=$(pwd) # Assumes you are submitting from the project root
CONDA_ENV_NAME="KGRAG"
MAIN_SCRIPT="tests/main.py"
CONFIG_FILE="configs/config.yaml"

# Create log and result directories if they don't exist
mkdir -p logs
mkdir -p results

# --- Environment Activation ---
echo "Activating Conda environment: $CONDA_ENV_NAME"

# Source conda.sh to make the 'conda' command available
if [ -f "$(conda info --base)/etc/profile.d/conda.sh" ]; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
else
    echo "Error: conda.sh not found. Cannot initialize Conda." >&2
    exit 1
fi

conda activate "$CONDA_ENV_NAME"
echo "Environment activated."
echo "Python executable: $(which python)"

# --- Set Environment Variables ---
export HF_HUB_OFFLINE=1
# The cache directories should be set in your ~/.bashrc or job submission command
# for consistency, but we can set them here as a fallback.
export HF_HOME=${HF_HOME:-"/data/user_data/$USER/.hf_cache"}
export HF_HUB_CACHE=${HF_HUB_CACHE:-"/data/hf_cache/hub"}

echo "Running in OFFLINE mode."
echo "Using Hugging Face cache: $HF_HUB_CACHE"
echo "Project Directory: $PROJECT_DIR"

# --- Run Main Pipeline ---
echo "Executing KGRAG pipeline script: $MAIN_SCRIPT"

python "$MAIN_SCRIPT" --run-pipeline --clear --config "$CONFIG_FILE"

EXIT_CODE=$?

# --- Job Completion ---
echo "============================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Job completed successfully."
else
    echo "Job failed with exit code $EXIT_CODE."
fi
echo "End Time: $(date)"
echo "============================================"

exit $EXIT_CODE