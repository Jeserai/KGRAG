#!/bin/bash

# Knowledge Graph RAG Environment Setup Script
# This script ensures all required components are ready on a compute node

set -e  # Exit on any error

echo "🚀 Setting up Knowledge Graph RAG Environment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
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

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check Python version
check_python_version() {
    print_status "Checking Python version..."
    if command_exists python3; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
        
        if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 9 ]; then
            print_success "Python $PYTHON_VERSION found (>= 3.9 required)"
        else
            print_error "Python $PYTHON_VERSION found, but 3.9+ is required"
            exit 1
        fi
    else
        print_error "Python 3 not found. Please install Python 3.9+"
        exit 1
    fi
}

# Function to check CUDA availability
check_cuda() {
    print_status "Checking CUDA availability..."
    if command_exists nvidia-smi; then
        CUDA_VERSION=$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader,nounits | head -1)
        if [ ! -z "$CUDA_VERSION" ]; then
            print_success "CUDA $CUDA_VERSION found"
            nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits | while read line; do
                print_status "GPU: $line"
            done
        else
            print_warning "CUDA not available, will use CPU"
        fi
    else
        print_warning "nvidia-smi not found, will use CPU"
    fi
}

# Function to check system resources
check_system_resources() {
    print_status "Checking system resources..."
    
    # Check RAM
    if command_exists free; then
        TOTAL_RAM=$(free -g | awk '/^Mem:/{print $2}')
        print_status "Total RAM: ${TOTAL_RAM}GB"
        
        if [ "$TOTAL_RAM" -lt 8 ]; then
            print_warning "Less than 8GB RAM detected. Performance may be limited."
        fi
    else
        print_warning "Cannot check RAM (free command not available)"
    fi
    
    # Check disk space
    DISK_SPACE=$(df -h . | awk 'NR==2 {print $4}')
    print_status "Available disk space: $DISK_SPACE"
    
    # Check CPU cores
    if command_exists nproc; then
        CPU_CORES=$(nproc)
        print_status "CPU cores: $CPU_CORES"
    else
        print_warning "Cannot check CPU cores (nproc command not available)"
    fi
}

# Function to upgrade pip
upgrade_pip() {
    print_status "Upgrading pip..."
    python3 -m pip install --upgrade pip
    print_success "pip upgraded"
}

# Function to install Python dependencies
install_python_dependencies() {
    print_status "Installing Python dependencies..."
    
    if [ -f "requirements.txt" ]; then
        print_status "Installing from requirements.txt..."
        python3 -m pip install -r requirements.txt
        print_success "Python dependencies installed"
    else
        print_error "requirements.txt not found"
        exit 1
    fi
}

# Function to check Neo4j availability
check_neo4j() {
    print_status "Checking Neo4j availability..."
    
    # Check if Neo4j is running locally
    if command_exists neo4j; then
        print_status "Neo4j command found"
        # Try to check status
        if neo4j status >/dev/null 2>&1; then
            print_success "Neo4j is running"
        else
            print_warning "Neo4j is installed but not running"
        fi
    else
        print_warning "Neo4j not found locally"
        print_status "You can install Neo4j using:"
        echo "  - Docker: docker run -p 7474:7474 -p 7687:7687 neo4j:latest"
        echo "  - Package manager: sudo apt-get install neo4j"
        echo "  - Download from: https://neo4j.com/download/"
    fi
    
    # Check if we can connect to Neo4j
    if command_exists python3; then
        python3 -c "
import sys
try:
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver('bolt://localhost:7687', auth=('neo4j', 'password'))
    with driver.session() as session:
        session.run('RETURN 1')
    print('✅ Neo4j connection successful')
    driver.close()
except Exception as e:
    print(f'❌ Neo4j connection failed: {e}')
    print('Please ensure Neo4j is running and accessible')
" 2>/dev/null || print_warning "Neo4j connection test failed"
    fi
}

# Function to check model cache
check_model_cache() {
    print_status "Checking HuggingFace model cache..."
    
    CACHE_DIR="$HOME/.cache/huggingface"
    if [ -d "$CACHE_DIR" ]; then
        CACHE_SIZE=$(du -sh "$CACHE_DIR" 2>/dev/null | cut -f1)
        print_status "HuggingFace cache size: $CACHE_SIZE"
        
        # List available models
        print_status "Available cached models:"
        if [ -d "$CACHE_DIR/hub" ]; then
            find "$CACHE_DIR/hub" -name "config.json" -type f | head -10 | while read file; do
                MODEL_PATH=$(dirname "$file" | sed "s|$CACHE_DIR/hub/||")
                print_status "  - $MODEL_PATH"
            done
        fi
    else
        print_warning "HuggingFace cache not found"
    fi
}

# Function to create environment file
create_env_file() {
    print_status "Creating environment configuration..."
    
    cat > .env << EOF
# Python and ML environment
PYTHONPATH=\$PWD:\$PYTHONPATH
CUDA_VISIBLE_DEVICES=0

# HuggingFace cache configuration
HF_HOME=/data/user_data/\$USER/.hf_cache
HF_HUB_CACHE=/data/hf_cache/hub
HF_DATASETS_CACHE=/data/hf_cache/datasets
HF_HUB_OFFLINE=1

# Neo4j configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Model and data paths
MODEL_CACHE_DIR=/data/hf_cache/hub
DATASET_CACHE_DIR=/data/hf_cache/datasets
OUTPUT_DIR=output
LOG_DIR=logs
DATA_DIR=data
EOF
    
    print_success "Environment file created: .env"
}

# Function to create directories
create_directories() {
    print_status "Creating necessary directories..."
    
    mkdir -p data
    mkdir -p cache
    mkdir -p output
    mkdir -p logs
    
    print_success "Directories created"
}

# Function to run comprehensive verification
run_verification() {
    print_status "Running comprehensive environment verification..."
    
    if [ -f "verify_environment.py" ]; then
        python3 verify_environment.py
    else
        print_warning "verify_environment.py not found, skipping verification"
    fi
}

# Main setup process
main() {
    echo "=========================================="
    echo "Knowledge Graph RAG Environment Setup"
    echo "=========================================="
    
    # Check system requirements
    check_python_version
    check_cuda
    check_system_resources
    
    # Upgrade pip and install dependencies
    upgrade_pip
    install_python_dependencies
    
    # Check external services
    check_neo4j
    check_model_cache
    
    # Create environment
    create_directories
    create_env_file
    
    # Run comprehensive verification (auto-installs missing packages)
    run_verification
    
    echo ""
    echo "=========================================="
    print_success "Environment setup completed!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo "1. Start Neo4j database (if not running)"
    echo "2. Configure your models in config/model_configs.yaml"
    echo "3. Run: python -m src.main"
    echo ""
    echo "For more information, see ENVIRONMENT_SETUP.md"
}

# Run main function
main "$@" 