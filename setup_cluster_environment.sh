#!/bin/bash

# Setup script for KGRAG on school server cluster
# Assumes Python 3.8, CUDA 11.4, and models in /data/models/huggingface

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE} $1${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Check if we're on the cluster
check_cluster_environment() {
    print_header "Checking Cluster Environment"
    
    # Check if we're in a cluster environment
    if [[ -n "$SLURM_JOB_ID" || -n "$PBS_JOBID" ]]; then
        print_info "Detected cluster environment"
        if [[ -n "$SLURM_JOB_ID" ]]; then
            print_info "SLURM job ID: $SLURM_JOB_ID"
        fi
        if [[ -n "$PBS_JOBID" ]]; then
            print_info "PBS job ID: $PBS_JOBID"
        fi
    else
        print_warning "Not in a cluster environment - this script is designed for cluster use"
    fi
    
    # Check Python version
    python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
    print_info "Python version: $python_version"
    
    if [[ ! "$python_version" =~ ^3\.8 ]]; then
        print_warning "Python 3.8.x is recommended, found: $python_version"
    fi
    
    # Check CUDA
    if command -v nvidia-smi &> /dev/null; then
        cuda_version=$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader,nounits | head -1)
        print_info "CUDA version: $cuda_version"
        
        if [[ "$cuda_version" != "11.4" ]]; then
            print_warning "CUDA 11.4 is recommended, found: $cuda_version"
        fi
    else
        print_error "nvidia-smi not found - CUDA may not be available"
        exit 1
    fi
    
    # Check available GPU memory
    if command -v nvidia-smi &> /dev/null; then
        gpu_memory=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
        print_info "GPU memory: ${gpu_memory}MB"
        
        if [[ $gpu_memory -lt 8000 ]]; then
            print_warning "At least 8GB GPU memory recommended for 7B models"
        fi
    fi
}

# Setup conda environment
setup_conda_environment() {
    print_header "Setting up Conda Environment"
    
    # Check if conda is available
    if ! command -v conda &> /dev/null; then
        print_error "Conda not found. Please install conda first."
        exit 1
    fi
    
    env_name="kgrag"
    
    # Check if environment already exists
    if conda env list | grep -q "^$env_name "; then
        print_warning "Conda environment '$env_name' already exists"
        read -p "Do you want to recreate it? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_info "Removing existing environment..."
            conda env remove -n $env_name
        else
            print_info "Using existing environment"
            return
        fi
    fi
    
    print_info "Creating conda environment: $env_name"
    conda create -n $env_name python=3.8 -y
    
    print_success "Conda environment created successfully"
}

# Install PyTorch with CUDA 11.4
install_pytorch() {
    print_header "Installing PyTorch with CUDA 11.4"
    
    # Activate conda environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate kgrag
    
    # Check if PyTorch is already installed
    if python3 -c "import torch; print('PyTorch version:', torch.__version__)" 2>/dev/null; then
        print_info "PyTorch is already installed"
        pytorch_version=$(python3 -c "import torch; print(torch.__version__)" 2>/dev/null)
        print_info "Current PyTorch version: $pytorch_version"
        
        # Check if CUDA is available
        cuda_available=$(python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
        if [[ "$cuda_available" == "True" ]]; then
            cuda_version=$(python3 -c "import torch; print(torch.version.cuda)" 2>/dev/null)
            print_success "CUDA is available with version: $cuda_version"
        else
            print_warning "CUDA is not available in PyTorch"
        fi
        
        # Ask user if they want to reinstall
        read -p "Do you want to reinstall PyTorch? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Skipping PyTorch installation"
            return
        fi
    fi
    
    print_info "Installing PyTorch 1.12.1 with CUDA 11.4..."
    pip install torch==1.12.1+cu114 -f https://download.pytorch.org/whl/torch_stable.html
    
    # Verify PyTorch installation
    python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
    
    print_success "PyTorch installed successfully"
}

# Install other requirements
install_requirements() {
    print_header "Installing Other Requirements"
    
    # Activate conda environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate kgrag
    
    print_info "Installing requirements from requirements.txt..."
    pip install -r requirements.txt
    
    print_success "Requirements installed successfully"
}

# Check model availability
check_models() {
    print_header "Checking Model Availability"
    
    models_dir="/data/models/huggingface"
    
    if [[ ! -d "$models_dir" ]]; then
        print_error "Models directory not found: $models_dir"
        exit 1
    fi
    
    print_info "Checking for required models..."
    
    required_models=(
        "meta-llama/Llama-3.2-3B-Instruct"
        "meta-llama/Meta-Llama-3-70B"
        "qwen/Qwen1.5-7B"
        "qwen/Qwen1.5-32B"
    )
    
    for model in "${required_models[@]}"; do
        model_path="$models_dir/$model"
        if [[ -d "$model_path" ]]; then
            print_success "✓ Found: $model"
        else
            print_warning "✗ Missing: $model"
        fi
    done
}

# Test model loading
test_model_loading() {
    print_header "Testing Model Loading"
    
    # Activate conda environment
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate kgrag
    
    print_info "Testing model configuration loading..."
    python3 -c "
from src.models.model_config import config_manager
print('Configuration loaded successfully')
print('Available LLM models:', list(config_manager.list_available_models('llm').keys()))
print('Available embedding models:', list(config_manager.list_available_models('embedding').keys()))
"
    
    print_success "Model configuration test passed"
}

# Setup environment variables
setup_environment_variables() {
    print_header "Setting up Environment Variables"
    
    # Create .env file
    cat > .env << EOF
# KGRAG Environment Variables for Cluster
MODEL_CACHE_DIR=/data/models/huggingface
HF_HOME=/data/models/huggingface
TRANSFORMERS_CACHE=/data/models/huggingface

# Neo4j settings (update these for your cluster)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Performance settings
CUDA_VISIBLE_DEVICES=0
PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
EOF
    
    print_success "Environment variables configured"
    print_info "Please update .env file with your specific Neo4j credentials"
}

# Main setup function
main() {
    print_header "KGRAG Cluster Environment Setup"
    
    check_cluster_environment
    setup_conda_environment
    install_pytorch
    install_requirements
    check_models
    setup_environment_variables
    test_model_loading
    
    print_header "Setup Complete!"
    print_success "KGRAG environment is ready for cluster use"
    print_info "To activate the environment: conda activate kgrag"
    print_info "To run tests: conda activate kgrag && pytest tests -v"
    print_info "To submit jobs: conda activate kgrag && ./submit_job.sh"
}

# Run main function
main "$@" 