#!/bin/bash

# ==============================================================================
# KGRAG Experiment Runner
#
# Description:
#   This script is designed to run a KGRAG experiment on a cluster node.
#   It activates the appropriate Conda environment, sets environment variables,
#   and executes the main Python script.
#
# Usage:
#   ./run_experiment.sh [path/to/your/data]
#
#   - The first argument should be the path to the directory containing
#     the documents you want to process.
#   - If no argument is provided, it will default to 'data/sample_docs'.
#
# ==============================================================================

set -e  # Exit immediately if a command exits with a non-zero status.
set -u  # Treat unset variables as an error when substituting.

# --- Configuration ---
CONDA_ENV_NAME="kgrag"
MAIN_SCRIPT_PATH="tests/main.py"
DEFAULT_DATA_DIR="data/sample_docs"
 
export HF_HOME=/data/user_data/$USER/.hf_cache
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HUB_OFFLINE=1
# --- Helper Functions ---
print_info() {
    echo -e "\033[0;34m[INFO]\033[0m $1"
}

print_error() {
    echo -e "\033[0;31m[ERROR]\033[0m $1" >&2
}

# --- Argument Parsing ---
# Use the provided data directory or the default one
DATA_DIR="${1:-$DEFAULT_DATA_DIR}"

# --- Script Execution ---
print_info "Starting KGRAG experiment..."
print_info "Using data directory: $DATA_DIR"

# 1. Check for Conda
if ! command -v conda &> /dev/null; then
    print_error "Conda could not be found. Please ensure Conda is installed and initialized."
    exit 1
fi

# 2. Activate Conda Environment
print_info "Activating Conda environment: $CONDA_ENV_NAME"
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda activate "$CONDA_ENV_NAME"; then
    print_error "Failed to activate Conda environment '$CONDA_ENV_NAME'. Please ensure it has been created."
    exit 1
fi
print_info "Python executable: $(which python)"
print_info "Python version: $(python --version)"

# 3. Set Python Path
# Ensures that modules in the 'src' directory can be imported
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"
print_info "PYTHONPATH set to: $PYTHONPATH"

# 4. Run the main experiment script
print_info "Executing main script: $MAIN_SCRIPT_PATH"
if [ ! -f "$MAIN_SCRIPT_PATH" ]; then
    print_error "Main script not found at '$MAIN_SCRIPT_PATH'"
    exit 1
fi

# Execute the script, passing the data directory as an argument
python "$MAIN_SCRIPT_PATH" --data_dir "$DATA_DIR"

print_info "Experiment finished successfully." 