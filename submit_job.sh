#!/bin/bash

# Knowledge Graph RAG Job Submission Script
# Handles job submission for different compute environments (local, SLURM, PBS, etc.)

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
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
    echo -e "${PURPLE}$1${NC}"
}

# Default configuration
DEFAULT_CONFIG="config/model_configs.yaml"
DEFAULT_SCRIPT="src/main.py"
DEFAULT_OUTPUT_DIR="output"
DEFAULT_LOG_DIR="logs"
DEFAULT_DATA_DIR="data"

# Job types
JOB_TYPES=("kg_build" "entity_extract" "qa_pipeline" "evaluation" "custom")

# Compute environments
COMPUTE_ENVS=("local" "slurm" "pbs" "docker" "kubernetes")

# Function to show usage
show_usage() {
    cat << EOF
Knowledge Graph RAG Job Submission Script

Usage: $0 [OPTIONS] --job-type <TYPE> --input <PATH>

OPTIONS:
    --job-type <TYPE>        Job type: ${JOB_TYPES[*]}
    --input <PATH>           Input data path (file/directory)
    --output <PATH>          Output directory (default: $DEFAULT_OUTPUT_DIR)
    --config <PATH>          Configuration file (default: $DEFAULT_CONFIG)
    --script <PATH>          Python script to run (default: $DEFAULT_SCRIPT)
    --compute-env <ENV>      Compute environment: ${COMPUTE_ENVS[*]}
    --gpu <NUM>              Number of GPUs to request
    --cpu <NUM>              Number of CPUs to request
    --memory <GB>            Memory in GB
    --time <HOURS>           Time limit in hours
    --name <NAME>            Job name
    --email <EMAIL>          Email for notifications
    --dry-run                Show what would be executed without running
    --help                   Show this help message

JOB TYPES:
    kg_build      - Build knowledge graph from documents
    entity_extract - Extract entities and relationships only
    qa_pipeline   - Run question-answering pipeline
    evaluation    - Run evaluation on test data
    custom        - Run custom script

COMPUTE ENVIRONMENTS:
    local         - Run locally (default)
    slurm         - Submit to SLURM cluster
    pbs           - Submit to PBS/Torque cluster
    docker        - Run in Docker container
    kubernetes    - Submit to Kubernetes cluster

EXAMPLES:
    # Build KG from documents locally
    $0 --job-type kg_build --input data/documents/ --gpu 1

    # Submit entity extraction to SLURM
    $0 --job-type entity_extract --input data/texts/ --compute-env slurm --gpu 2 --time 4

    # Run QA pipeline with custom config
    $0 --job-type qa_pipeline --input data/questions.txt --config config/custom.yaml

    # Dry run to see what would be executed
    $0 --job-type kg_build --input data/ --dry-run

EOF
}

# Function to parse command line arguments
parse_args() {
    # Default values
    JOB_TYPE=""
    INPUT_PATH=""
    OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
    CONFIG_FILE="$DEFAULT_CONFIG"
    SCRIPT_PATH="$DEFAULT_SCRIPT"
    COMPUTE_ENV="local"
    GPU_COUNT=0
    CPU_COUNT=1
    MEMORY_GB=16
    TIME_HOURS=24
    JOB_NAME=""
    EMAIL=""
    DRY_RUN=false

    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --job-type)
                JOB_TYPE="$2"
                shift 2
                ;;
            --input)
                INPUT_PATH="$2"
                shift 2
                ;;
            --output)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            --script)
                SCRIPT_PATH="$2"
                shift 2
                ;;
            --compute-env)
                COMPUTE_ENV="$2"
                shift 2
                ;;
            --gpu)
                GPU_COUNT="$2"
                shift 2
                ;;
            --cpu)
                CPU_COUNT="$2"
                shift 2
                ;;
            --memory)
                MEMORY_GB="$2"
                shift 2
                ;;
            --time)
                TIME_HOURS="$2"
                shift 2
                ;;
            --name)
                JOB_NAME="$2"
                shift 2
                ;;
            --email)
                EMAIL="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            --help)
                show_usage
                exit 0
                ;;
            *)
                print_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done

    # Validate required arguments
    if [[ -z "$JOB_TYPE" ]]; then
        print_error "Job type is required"
        show_usage
        exit 1
    fi

    if [[ -z "$INPUT_PATH" ]]; then
        print_error "Input path is required"
        show_usage
        exit 1
    fi

    # Validate job type
    if [[ ! " ${JOB_TYPES[@]} " =~ " ${JOB_TYPE} " ]]; then
        print_error "Invalid job type: $JOB_TYPE"
        print_info "Valid job types: ${JOB_TYPES[*]}"
        exit 1
    fi

    # Validate compute environment
    if [[ ! " ${COMPUTE_ENVS[@]} " =~ " ${COMPUTE_ENV} " ]]; then
        print_error "Invalid compute environment: $COMPUTE_ENV"
        print_info "Valid environments: ${COMPUTE_ENVS[*]}"
        exit 1
    fi

    # Set default job name if not provided
    if [[ -z "$JOB_NAME" ]]; then
        JOB_NAME="kgrag_${JOB_TYPE}_$(date +%Y%m%d_%H%M%S)"
    fi
}

# Function to validate input path
validate_input() {
    if [[ ! -e "$INPUT_PATH" ]]; then
        print_error "Input path does not exist: $INPUT_PATH"
        exit 1
    fi

    print_success "Input path validated: $INPUT_PATH"
}

# Function to create job-specific Python command
create_python_command() {
    local cmd="python -m src.main"
    
    case "$JOB_TYPE" in
        "kg_build")
            cmd="$cmd --mode build --input $INPUT_PATH --output $OUTPUT_DIR --config $CONFIG_FILE"
            ;;
        "entity_extract")
            cmd="$cmd --mode extract --input $INPUT_PATH --output $OUTPUT_DIR --config $CONFIG_FILE"
            ;;
        "qa_pipeline")
            cmd="$cmd --mode qa --input $INPUT_PATH --output $OUTPUT_DIR --config $CONFIG_FILE"
            ;;
        "evaluation")
            cmd="$cmd --mode evaluate --input $INPUT_PATH --output $OUTPUT_DIR --config $CONFIG_FILE"
            ;;
        "custom")
            cmd="python $SCRIPT_PATH --input $INPUT_PATH --output $OUTPUT_DIR --config $CONFIG_FILE"
            ;;
    esac

    echo "$cmd"
}

# Function to create SLURM job script
create_slurm_script() {
    local python_cmd="$1"
    local script_file="slurm_job_${JOB_NAME}.sh"
    
    cat > "$script_file" << EOF
#!/bin/bash
#SBATCH --job-name=$JOB_NAME
#SBATCH --output=$DEFAULT_LOG_DIR/${JOB_NAME}_%j.out
#SBATCH --error=$DEFAULT_LOG_DIR/${JOB_NAME}_%j.err
#SBATCH --time=$TIME_HOURS:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=$CPU_COUNT
#SBATCH --mem=${MEMORY_GB}G
EOF

    if [[ $GPU_COUNT -gt 0 ]]; then
        cat >> "$script_file" << EOF
#SBATCH --gres=gpu:$GPU_COUNT
EOF
    fi

    if [[ -n "$EMAIL" ]]; then
        cat >> "$script_file" << EOF
#SBATCH --mail-user=$EMAIL
#SBATCH --mail-type=ALL
EOF
    fi

    cat >> "$script_file" << EOF

# Load modules (customize for your cluster)
module load python/3.9
module load cuda/11.8

# Set environment variables
export PYTHONPATH=\$PWD:\$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

# Set HuggingFace cache environment variables
export HF_HOME=/data/user_data/\$USER/.hf_cache
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HUB_OFFLINE=1

# Create output and log directories
mkdir -p $OUTPUT_DIR
mkdir -p $DEFAULT_LOG_DIR

# Print job information
echo "Job started at: \$(date)"
echo "Job ID: \$SLURM_JOB_ID"
echo "Node: \$SLURM_NODELIST"
echo "Working directory: \$PWD"
echo "HF_HOME: \$HF_HOME"
echo "HF_HUB_CACHE: \$HF_HUB_CACHE"
echo "HF_DATASETS_CACHE: \$HF_DATASETS_CACHE"

# Run the command
$python_cmd

# Print completion information
echo "Job completed at: \$(date)"
EOF

    chmod +x "$script_file"
    echo "$script_file"
}

# Function to create PBS job script
create_pbs_script() {
    local python_cmd="$1"
    local script_file="pbs_job_${JOB_NAME}.sh"
    
    cat > "$script_file" << EOF
#!/bin/bash
#PBS -N $JOB_NAME
#PBS -o $DEFAULT_LOG_DIR/${JOB_NAME}_\${PBS_JOBID}.out
#PBS -e $DEFAULT_LOG_DIR/${JOB_NAME}_\${PBS_JOBID}.err
#PBS -l walltime=$TIME_HOURS:00:00
#PBS -l nodes=1:ppn=$CPU_COUNT
#PBS -l mem=${MEMORY_GB}gb
EOF

    if [[ $GPU_COUNT -gt 0 ]]; then
        cat >> "$script_file" << EOF
#PBS -l gpus=$GPU_COUNT
EOF
    fi

    if [[ -n "$EMAIL" ]]; then
        cat >> "$script_file" << EOF
#PBS -M $EMAIL
#PBS -m abe
EOF
    fi

    cat >> "$script_file" << EOF

# Load modules (customize for your cluster)
module load python/3.9
module load cuda/11.8

# Set environment variables
export PYTHONPATH=\$PWD:\$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0

# Set HuggingFace cache environment variables
export HF_HOME=/data/user_data/\$USER/.hf_cache
export HF_HUB_CACHE=/data/hf_cache/hub
export HF_DATASETS_CACHE=/data/hf_cache/datasets
export HF_HUB_OFFLINE=1

# Create output and log directories
mkdir -p $OUTPUT_DIR
mkdir -p $DEFAULT_LOG_DIR

# Print job information
echo "Job started at: \$(date)"
echo "Job ID: \$PBS_JOBID"
echo "Node: \$PBS_NODEFILE"
echo "Working directory: \$PWD"
echo "HF_HOME: \$HF_HOME"
echo "HF_HUB_CACHE: \$HF_HUB_CACHE"
echo "HF_DATASETS_CACHE: \$HF_DATASETS_CACHE"

# Run the command
$python_cmd

# Print completion information
echo "Job completed at: \$(date)"
EOF

    chmod +x "$script_file"
    echo "$script_file"
}

# Function to create Docker run command
create_docker_command() {
    local python_cmd="$1"
    
    local docker_cmd="docker run --rm"
    
    # Add GPU support if requested
    if [[ $GPU_COUNT -gt 0 ]]; then
        docker_cmd="$docker_cmd --gpus all"
    fi
    
    # Add resource limits
    docker_cmd="$docker_cmd --memory=${MEMORY_GB}g --cpus=$CPU_COUNT"
    
    # Add volume mounts
    docker_cmd="$docker_cmd -v \$(pwd):/app -v \$(pwd)/$OUTPUT_DIR:/app/$OUTPUT_DIR"
    docker_cmd="$docker_cmd -v \$(pwd)/$DEFAULT_LOG_DIR:/app/$DEFAULT_LOG_DIR"
    docker_cmd="$docker_cmd -v \$(pwd)/$INPUT_PATH:/app/input_data"
    
    # Add HuggingFace cache volume mounts
    docker_cmd="$docker_cmd -v /data/user_data/\$USER/.hf_cache:/data/user_data/\$USER/.hf_cache"
    docker_cmd="$docker_cmd -v /data/hf_cache/hub:/data/hf_cache/hub"
    docker_cmd="$docker_cmd -v /data/hf_cache/datasets:/data/hf_cache/datasets"
    
    # Add environment variables
    docker_cmd="$docker_cmd -e PYTHONPATH=/app"
    docker_cmd="$docker_cmd -e HF_HOME=/data/user_data/\$USER/.hf_cache"
    docker_cmd="$docker_cmd -e HF_HUB_CACHE=/data/hf_cache/hub"
    docker_cmd="$docker_cmd -e HF_DATASETS_CACHE=/data/hf_cache/datasets"
    docker_cmd="$docker_cmd -e HF_HUB_OFFLINE=1"
    
    # Add image and command
    docker_cmd="$docker_cmd kgrag:latest bash -c 'cd /app && $python_cmd'"
    
    echo "$docker_cmd"
}

# Function to submit job based on compute environment
submit_job() {
    local python_cmd="$1"
    
    case "$COMPUTE_ENV" in
        "local")
            print_info "Running job locally..."
            if [[ "$DRY_RUN" == true ]]; then
                print_info "DRY RUN - Would execute: $python_cmd"
            else
                mkdir -p "$OUTPUT_DIR" "$DEFAULT_LOG_DIR"
                
                # Set HuggingFace cache environment variables for local execution
                export HF_HOME=/data/user_data/$USER/.hf_cache
                export HF_HUB_CACHE=/data/hf_cache/hub
                export HF_DATASETS_CACHE=/data/hf_cache/datasets
                export HF_HUB_OFFLINE=1
                
                print_info "Using HuggingFace cache:"
                print_info "  HF_HOME: $HF_HOME"
                print_info "  HF_HUB_CACHE: $HF_HUB_CACHE"
                print_info "  HF_DATASETS_CACHE: $HF_DATASETS_CACHE"
                
                eval "$python_cmd"
            fi
            ;;
            
        "slurm")
            print_info "Submitting job to SLURM..."
            local script_file=$(create_slurm_script "$python_cmd")
            if [[ "$DRY_RUN" == true ]]; then
                print_info "DRY RUN - Would submit: sbatch $script_file"
                print_info "Script content:"
                cat "$script_file"
            else
                local job_id=$(sbatch "$script_file" | awk '{print $4}')
                print_success "Job submitted to SLURM with ID: $job_id"
                print_info "Monitor with: squeue -j $job_id"
            fi
            ;;
            
        "pbs")
            print_info "Submitting job to PBS..."
            local script_file=$(create_pbs_script "$python_cmd")
            if [[ "$DRY_RUN" == true ]]; then
                print_info "DRY RUN - Would submit: qsub $script_file"
                print_info "Script content:"
                cat "$script_file"
            else
                local job_id=$(qsub "$script_file")
                print_success "Job submitted to PBS with ID: $job_id"
                print_info "Monitor with: qstat $job_id"
            fi
            ;;
            
        "docker")
            print_info "Running job in Docker..."
            local docker_cmd=$(create_docker_command "$python_cmd")
            if [[ "$DRY_RUN" == true ]]; then
                print_info "DRY RUN - Would execute: $docker_cmd"
            else
                eval "$docker_cmd"
            fi
            ;;
            
        "kubernetes")
            print_warning "Kubernetes submission not yet implemented"
            print_info "Would create Kubernetes job manifest and apply it"
            ;;
    esac
}

# Function to show job summary
show_job_summary() {
    print_header "Job Summary"
    echo "Job Type: $JOB_TYPE"
    echo "Input Path: $INPUT_PATH"
    echo "Output Directory: $OUTPUT_DIR"
    echo "Configuration: $CONFIG_FILE"
    echo "Compute Environment: $COMPUTE_ENV"
    echo "Job Name: $JOB_NAME"
    
    if [[ $GPU_COUNT -gt 0 ]]; then
        echo "GPUs: $GPU_COUNT"
    fi
    
    echo "CPUs: $CPU_COUNT"
    echo "Memory: ${MEMORY_GB}GB"
    echo "Time Limit: ${TIME_HOURS}h"
    
    if [[ -n "$EMAIL" ]]; then
        echo "Email: $EMAIL"
    fi
    
    if [[ "$DRY_RUN" == true ]]; then
        print_warning "DRY RUN MODE - No actual job will be submitted"
    fi
}

# Main function
main() {
    print_header "Knowledge Graph RAG Job Submission"
    
    # Parse arguments
    parse_args "$@"
    
    # Validate input
    validate_input
    
    # Show job summary
    show_job_summary
    
    # Create Python command
    local python_cmd=$(create_python_command)
    print_info "Python command: $python_cmd"
    
    # Submit job
    submit_job "$python_cmd"
    
    print_success "Job submission completed!"
}

# Run main function with all arguments
main "$@" 