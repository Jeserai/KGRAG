#!/bin/bash

# HuggingFace Cache Environment Setup Script
# Sets up environment variables for using cached HuggingFace models and datasets

set -e

# Colors for output
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

print_header() {
    echo -e "${BLUE}================================${NC}"
    echo -e "${BLUE}  HuggingFace Cache Setup${NC}"
    echo -e "${BLUE}================================${NC}"
}

# Function to show usage
show_usage() {
    cat << EOF
HuggingFace Cache Environment Setup Script

Usage: $0 [OPTIONS]

OPTIONS:
    --user-cache <PATH>      User-specific cache directory (default: /data/user_data/\$USER/.hf_cache)
    --hub-cache <PATH>       Model hub cache directory (default: /data/hf_cache/hub)
    --datasets-cache <PATH>  Datasets cache directory (default: /data/hf_cache/datasets)
    --offline                Enable offline mode (HF_HUB_OFFLINE=1)
    --online                 Disable offline mode (HF_HUB_OFFLINE=0)
    --export                 Export variables to shell (source this script)
    --check                  Check current cache configuration
    --help                   Show this help message

EXAMPLES:
    # Set up cache with default paths
    $0

    # Set up cache with custom paths
    $0 --user-cache /path/to/user/cache --hub-cache /path/to/hub/cache

    # Enable offline mode
    $0 --offline

    # Export variables to current shell
    source \$($0 --export)

    # Check current configuration
    $0 --check

EOF
}

# Default values
USER_CACHE="/data/user_data/$USER/.hf_cache"
HUB_CACHE="/data/hf_cache/hub"
DATASETS_CACHE="/data/hf_cache/datasets"
OFFLINE_MODE=true
EXPORT_MODE=false
CHECK_MODE=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --user-cache)
            USER_CACHE="$2"
            shift 2
            ;;
        --hub-cache)
            HUB_CACHE="$2"
            shift 2
            ;;
        --datasets-cache)
            DATASETS_CACHE="$2"
            shift 2
            ;;
        --offline)
            OFFLINE_MODE=true
            shift
            ;;
        --online)
            OFFLINE_MODE=false
            shift
            ;;
        --export)
            EXPORT_MODE=true
            shift
            ;;
        --check)
            CHECK_MODE=true
            shift
            ;;
        --help)
            show_usage
            exit 0
            ;;
        *)
            print_warning "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Function to check cache configuration
check_cache_config() {
    print_header
    print_info "Checking HuggingFace cache configuration..."
    
    echo ""
    echo "Environment Variables:"
    echo "  HF_HOME: ${HF_HOME:-'Not set'}"
    echo "  HF_HUB_CACHE: ${HF_HUB_CACHE:-'Not set'}"
    echo "  HF_DATASETS_CACHE: ${HF_DATASETS_CACHE:-'Not set'}"
    echo "  HF_HUB_OFFLINE: ${HF_HUB_OFFLINE:-'Not set'}"
    
    echo ""
    echo "Cache Directories:"
    
    # Check user cache
    if [[ -n "$HF_HOME" && -d "$HF_HOME" ]]; then
        size=$(du -sh "$HF_HOME" 2>/dev/null | cut -f1 || echo "Unknown")
        print_success "HF_HOME: $HF_HOME ($size)"
    elif [[ -n "$HF_HOME" ]]; then
        print_warning "HF_HOME: $HF_HOME (directory does not exist)"
    else
        print_warning "HF_HOME: Not configured"
    fi
    
    # Check hub cache
    if [[ -n "$HF_HUB_CACHE" && -d "$HF_HUB_CACHE" ]]; then
        size=$(du -sh "$HF_HUB_CACHE" 2>/dev/null | cut -f1 || echo "Unknown")
        model_count=$(find "$HF_HUB_CACHE" -maxdepth 2 -type d 2>/dev/null | wc -l || echo "0")
        print_success "HF_HUB_CACHE: $HF_HUB_CACHE ($size, ~$model_count models)"
    elif [[ -n "$HF_HUB_CACHE" ]]; then
        print_warning "HF_HUB_CACHE: $HF_HUB_CACHE (directory does not exist)"
    else
        print_warning "HF_HUB_CACHE: Not configured"
    fi
    
    # Check datasets cache
    if [[ -n "$HF_DATASETS_CACHE" && -d "$HF_DATASETS_CACHE" ]]; then
        size=$(du -sh "$HF_DATASETS_CACHE" 2>/dev/null | cut -f1 || echo "Unknown")
        dataset_count=$(find "$HF_DATASETS_CACHE" -maxdepth 2 -type d 2>/dev/null | wc -l || echo "0")
        print_success "HF_DATASETS_CACHE: $HF_DATASETS_CACHE ($size, ~$dataset_count datasets)"
    elif [[ -n "$HF_DATASETS_CACHE" ]]; then
        print_warning "HF_DATASETS_CACHE: $HF_DATASETS_CACHE (directory does not exist)"
    else
        print_warning "HF_DATASETS_CACHE: Not configured"
    fi
    
    echo ""
    echo "Offline Mode:"
    if [[ "$HF_HUB_OFFLINE" == "1" ]]; then
        print_success "HF_HUB_OFFLINE=1 (offline mode enabled)"
    else
        print_warning "HF_HUB_OFFLINE not set to 1 (online mode)"
    fi
}

# Function to create cache directories
create_cache_dirs() {
    print_info "Creating cache directories..."
    
    for dir in "$USER_CACHE" "$HUB_CACHE" "$DATASETS_CACHE"; do
        if [[ ! -d "$dir" ]]; then
            print_info "Creating directory: $dir"
            mkdir -p "$dir"
            if [[ $? -eq 0 ]]; then
                print_success "Created: $dir"
            else
                print_warning "Failed to create: $dir"
            fi
        else
            print_success "Directory exists: $dir"
        fi
    done
}

# Function to set environment variables
set_environment_vars() {
    if [[ "$EXPORT_MODE" == true ]]; then
        # Export mode - output export commands
        echo "export HF_HOME=$USER_CACHE"
        echo "export HF_HUB_CACHE=$HUB_CACHE"
        echo "export HF_DATASETS_CACHE=$DATASETS_CACHE"
        if [[ "$OFFLINE_MODE" == true ]]; then
            echo "export HF_HUB_OFFLINE=1"
        else
            echo "export HF_HUB_OFFLINE=0"
        fi
    else
        # Set variables for current session
        export HF_HOME="$USER_CACHE"
        export HF_HUB_CACHE="$HUB_CACHE"
        export HF_DATASETS_CACHE="$DATASETS_CACHE"
        
        if [[ "$OFFLINE_MODE" == true ]]; then
            export HF_HUB_OFFLINE=1
        else
            export HF_HUB_OFFLINE=0
        fi
        
        print_success "Environment variables set for current session"
        print_info "HF_HOME: $HF_HOME"
        print_info "HF_HUB_CACHE: $HF_HUB_CACHE"
        print_info "HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
        print_info "HF_HUB_OFFLINE: $HF_HUB_OFFLINE"
    fi
}

# Main function
main() {
    if [[ "$CHECK_MODE" == true ]]; then
        check_cache_config
        exit 0
    fi
    
    if [[ "$EXPORT_MODE" == true ]]; then
        set_environment_vars
        exit 0
    fi
    
    print_header
    print_info "Setting up HuggingFace cache environment..."
    
    # Create cache directories
    create_cache_dirs
    
    # Set environment variables
    set_environment_vars
    
    print_success "HuggingFace cache environment setup completed!"
    print_info "To make these settings permanent, add them to your shell profile:"
    echo ""
    echo "  # Add to ~/.bashrc or ~/.zshrc:"
    echo "  export HF_HOME=$USER_CACHE"
    echo "  export HF_HUB_CACHE=$HUB_CACHE"
    echo "  export HF_DATASETS_CACHE=$DATASETS_CACHE"
    if [[ "$OFFLINE_MODE" == true ]]; then
        echo "  export HF_HUB_OFFLINE=1"
    else
        echo "  export HF_HUB_OFFLINE=0"
    fi
    echo ""
    print_info "Or source this script in your shell profile:"
    echo "  source $(pwd)/$0"
}

# Run main function
main "$@" 