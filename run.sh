#!/bin/bash

#SBATCH --job-name=graphrag_experiment
#SBATCH --partition=gpu                    # Use your cluster's GPU partition name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8                  # Adjust based on your cluster
#SBATCH --gres=gpu:1                       # Request 1 GPU (adjust if needed)
#SBATCH --mem=32G                          # Memory requirement for Qwen 7B
#SBATCH --time=02:00:00                    # 2 hour time limit (adjust as needed)
#SBATCH --output=graphrag_%j.out           # Output file with job ID
#SBATCH --error=graphrag_%j.err            # Error file with job ID
#SBATCH --mail-type=BEGIN,END,FAIL         # Email notifications
#SBATCH --mail-user=your.email@university.edu  # Replace with your email

# =============================================================================
# GraphRAG Cluster Experiment Script
# 
# This script sets up the environment and runs GraphRAG experiments on a
# cluster with SLURM job scheduler. It handles model downloading, environment
# setup, and experiment execution.
# =============================================================================

# Exit on any error for safer execution
set -e

# Print job information for debugging
echo "============================================"
echo "GraphRAG Experiment Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Started at: $(date)"
echo "============================================"

# Configuration variables - MODIFY THESE FOR YOUR CLUSTER
PROJECT_DIR="$HOME/graphrag_project"          # Path to your project directory
VENV_NAME="graphrag_env"                      # Python virtual environment name
CONDA_ENV_NAME="graphrag"                    # If using conda instead of venv
USE_CONDA=false                              # Set to true if using conda
INPUT_DATA_PATH=""                           # Path to your input documents (optional)
CONFIG_PATH="configs/config.yaml"            # Path to your config file

# =============================================================================
# SECTION 1: Environment Setup
# =============================================================================

echo "Setting up environment..."

# Common module names, but check with 'module avail' on your cluster
module load python/3.9                       # Load Python (adjust version)
module load cuda/11.8                        # Load CUDA (adjust version)
module load gcc/9.3.0                        # Load GCC if needed

echo "Loaded modules:"
module list

# Navigate to project directory
cd "$PROJECT_DIR"
echo "Working directory: $(pwd)"

# Set up Hugging Face cache
export HF_HOME=/data/user_data/yuanguan/.hf_cache
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HUB_OFFLINE=1

echo "HuggingFace cache directory: $HF_HOME"

# =============================================================================
# SECTION 2: Python Environment Activation
# =============================================================================

echo "Activating Python environment..."

if [ "$USE_CONDA" = true ]; then
    # Using Conda environment
    echo "Using Conda environment: $CONDA_ENV_NAME"
    source activate "$CONDA_ENV_NAME"
else
    # Using Python virtual environment
    echo "Using virtual environment: $VENV_NAME"
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "$VENV_NAME" ]; then
        echo "Creating new virtual environment..."
        python -m venv "$VENV_NAME"
    fi
    
    # Activate virtual environment
    source "$VENV_NAME/bin/activate"
fi

# Verify Python and pip
echo "Python version: $(python --version)"
echo "Pip version: $(pip --version)"

# =============================================================================
# SECTION 3: Install Dependencies
# =============================================================================

echo "Installing/updating dependencies..."

# Upgrade pip first
pip install --upgrade pip

# Install requirements with cluster-friendly settings
pip install -r requirements.txt --user --no-cache-dir

# Install additional dependencies that might be needed on clusters
pip install --user --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Dependencies installed successfully"

# =============================================================================
# SECTION 4: Model Download (if needed)
# =============================================================================

echo "Checking if models need to be downloaded..."

# Check if models are already cached
QWEN_MODEL_PATH="$HF_HUB_CACHE/models--Qwen--Qwen2.5-7B-Instruct"
BGE_MODEL_PATH="$HF_HUB_CACHE/models--BAAI--bge-large-en-v1.5"

if [ ! -d "$QWEN_MODEL_PATH" ] || [ ! -d "$BGE_MODEL_PATH" ]; then
    echo "Downloading models..."
    python src/models/download_models.py
    
    if [ $? -eq 0 ]; then
        echo "Models downloaded successfully"
    else
        echo "ERROR: Model download failed"
        exit 1
    fi
else
    echo "Models already cached, skipping download"
fi

# =============================================================================
# SECTION 5: GPU and System Information
# =============================================================================

echo "System information:"
echo "GPU Information:"
nvidia-smi || echo "nvidia-smi not available"

echo "CPU Information:"
echo "Cores: $(nproc)"
echo "Memory: $(free -h | grep '^Mem:' | awk '{print $2}')"

echo "CUDA Version:"
nvcc --version 2>/dev/null || echo "nvcc not available"

# =============================================================================
# SECTION 6: Test Model Loading
# =============================================================================

echo "Testing model loading..."

# Test if models can be loaded properly
python tests/main.py --test-models --config "$CONFIG_PATH"

if [ $? -ne 0 ]; then
    echo "ERROR: Model loading test failed"
    exit 1
fi

echo "Model loading test passed"

# =============================================================================
# SECTION 7: Run Main Experiment
# =============================================================================

echo "Starting main GraphRAG experiment..."

# Prepare experiment command
EXPERIMENT_CMD="python tests/main.py --config $CONFIG_PATH"

# Add input data if specified
if [ -n "$INPUT_DATA_PATH" ] && [ -d "$INPUT_DATA_PATH" ]; then
    EXPERIMENT_CMD="$EXPERIMENT_CMD --input $INPUT_DATA_PATH"
    echo "Using input data from: $INPUT_DATA_PATH"
else
    # Use built-in test data
    EXPERIMENT_CMD="$EXPERIMENT_CMD --run-pipeline"
    echo "Using built-in test data"
fi

# Add other experiment options
EXPERIMENT_CMD="$EXPERIMENT_CMD --clear --stats --validate"

echo "Executing: $EXPERIMENT_CMD"

# Run the experiment with error handling
$EXPERIMENT_CMD

EXPERIMENT_EXIT_CODE=$?

# =============================================================================
# SECTION 8: Post-Processing and Cleanup
# =============================================================================

if [ $EXPERIMENT_EXIT_CODE -eq 0 ]; then
    echo "Experiment completed successfully"
    
    # Export results if graph storage is available
    echo "Exporting results..."
    python tests/main.py --export "results_${SLURM_JOB_ID}.json" --config "$CONFIG_PATH"
    
    # Generate final statistics
    echo "Final graph statistics:"
    python tests/main.py --stats --config "$CONFIG_PATH"
    
else
    echo "ERROR: Experiment failed with exit code $EXPERIMENT_EXIT_CODE"
fi

# =============================================================================
# SECTION 9: Result Summary and Cleanup
# =============================================================================

echo "============================================"
echo "Job Summary"
echo "============================================"
echo "Job ID: $SLURM_JOB_ID"
echo "End time: $(date)"
echo "Exit code: $EXPERIMENT_EXIT_CODE"

# Show file sizes of outputs
echo "Output files:"
ls -lh graphrag_*.out graphrag_*.err 2>/dev/null || echo "No output files found"
ls -lh *.json 2>/dev/null || echo "No JSON result files found"
ls -lh *.log 2>/dev/null || echo "No log files found"

# Optional: Copy results to a results directory
RESULTS_DIR="$HOME/graphrag_results/job_$SLURM_JOB_ID"
mkdir -p "$RESULTS_DIR"

# Copy important files to results directory
cp -f *.json "$RESULTS_DIR/" 2>/dev/null || echo "No JSON files to copy"
cp -f *.log "$RESULTS_DIR/" 2>/dev/null || echo "No log files to copy"
cp -f "graphrag_${SLURM_JOB_ID}.out" "$RESULTS_DIR/" 2>/dev/null || echo "No stdout file to copy"
cp -f "graphrag_${SLURM_JOB_ID}.err" "$RESULTS_DIR/" 2>/dev/null || echo "No stderr file to copy"

echo "Results copied to: $RESULTS_DIR"

# Clean up temporary files if needed
# rm -f temp_*.txt 2>/dev/null || true

echo "GraphRAG experiment completed"
echo "============================================"

# Exit with the same code as the main experiment
exit $EXPERIMENT_EXIT_CODE