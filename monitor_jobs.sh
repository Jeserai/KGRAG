#!/bin/bash

# Knowledge Graph RAG Job Monitoring Script
# Monitors and manages jobs across different compute environments

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

# Default values
JOB_FILE=".kgrag_jobs"
LOG_DIR="logs"
OUTPUT_DIR="output"

# Function to show usage
show_usage() {
    cat << EOF
Knowledge Graph RAG Job Monitoring Script

Usage: $0 [COMMAND] [OPTIONS]

COMMANDS:
    list                    List all tracked jobs
    status [JOB_ID]         Show status of specific job or all jobs
    logs [JOB_ID]           Show logs for specific job
    cancel [JOB_ID]         Cancel a running job
    clean [JOB_ID]          Clean up completed jobs
    watch [JOB_ID]          Watch job progress in real-time
    summary                 Show summary of all jobs
    help                    Show this help message

OPTIONS:
    --compute-env <ENV>     Filter by compute environment (slurm, pbs, local)
    --status <STATUS>       Filter by job status (running, completed, failed)
    --job-type <TYPE>       Filter by job type
    --days <N>              Show jobs from last N days
    --verbose               Show detailed information

EXAMPLES:
    # List all jobs
    $0 list

    # Show status of specific job
    $0 status 12345

    # Show logs for job
    $0 logs 12345

    # Cancel running job
    $0 cancel 12345

    # Watch job progress
    $0 watch 12345

    # Show summary
    $0 summary

    # List only SLURM jobs
    $0 list --compute-env slurm

    # Show failed jobs from last 7 days
    $0 status --status failed --days 7

EOF
}

# Function to get job status from compute environment
get_job_status() {
    local job_id="$1"
    local compute_env="$2"
    
    case "$compute_env" in
        "slurm")
            squeue -j "$job_id" 2>/dev/null | tail -n +2 | awk '{print $5}' || echo "COMPLETED"
            ;;
        "pbs")
            qstat "$job_id" 2>/dev/null | tail -n +2 | awk '{print $10}' || echo "C"
            ;;
        "local")
            # For local jobs, check if process is still running
            if ps -p "$job_id" > /dev/null 2>&1; then
                echo "RUNNING"
            else
                echo "COMPLETED"
            fi
            ;;
        *)
            echo "UNKNOWN"
            ;;
    esac
}

# Function to map status codes to readable names
map_status() {
    local status="$1"
    local compute_env="$2"
    
    case "$compute_env" in
        "slurm")
            case "$status" in
                "R") echo "RUNNING" ;;
                "PD") echo "PENDING" ;;
                "CG") echo "COMPLETING" ;;
                "CD") echo "COMPLETED" ;;
                "F") echo "FAILED" ;;
                "TO") echo "TIMEOUT" ;;
                "CA") echo "CANCELLED" ;;
                *) echo "$status" ;;
            esac
            ;;
        "pbs")
            case "$status" in
                "R") echo "RUNNING" ;;
                "Q") echo "QUEUED" ;;
                "C") echo "COMPLETED" ;;
                "E") echo "EXITING" ;;
                "H") echo "HELD" ;;
                "T") echo "MOVED" ;;
                "W") echo "WAITING" ;;
                "S") echo "SUSPENDED" ;;
                *) echo "$status" ;;
            esac
            ;;
        *)
            echo "$status"
            ;;
    esac
}

# Function to list all tracked jobs
list_jobs() {
    local filter_env="$1"
    local filter_status="$2"
    local filter_type="$3"
    local days="$4"
    
    print_header "Tracked Jobs"
    
    if [[ ! -f "$JOB_FILE" ]]; then
        print_warning "No jobs file found. No jobs have been submitted yet."
        return
    fi
    
    # Create temporary file for filtered results
    local temp_file=$(mktemp)
    
    # Apply filters
    if [[ -n "$days" ]]; then
        # Filter by date (last N days)
        local cutoff_date=$(date -d "$days days ago" +%Y-%m-%d)
        awk -v cutoff="$cutoff_date" '$3 >= cutoff' "$JOB_FILE" > "$temp_file"
    else
        cp "$JOB_FILE" "$temp_file"
    fi
    
    if [[ -n "$filter_env" ]]; then
        awk -v env="$filter_env" '$2 == env' "$temp_file" > "${temp_file}.tmp" && mv "${temp_file}.tmp" "$temp_file"
    fi
    
    if [[ -n "$filter_type" ]]; then
        awk -v type="$filter_type" '$5 == type' "$temp_file" > "${temp_file}.tmp" && mv "${temp_file}.tmp" "$temp_file"
    fi
    
    # Display jobs
    if [[ -s "$temp_file" ]]; then
        printf "%-12s %-8s %-12s %-15s %-12s %-10s %-20s\n" "JOB_ID" "ENV" "DATE" "NAME" "TYPE" "STATUS" "INPUT"
        echo "----------------------------------------------------------------------------------------"
        
        while IFS='|' read -r job_id env date name type status input_path; do
            # Get current status if job is still running
            if [[ "$status" == "RUNNING" || "$status" == "PENDING" ]]; then
                current_status=$(get_job_status "$job_id" "$env")
                status=$(map_status "$current_status" "$env")
            fi
            
            printf "%-12s %-8s %-12s %-15s %-12s %-10s %-20s\n" \
                   "$job_id" "$env" "$date" "$name" "$type" "$status" "$(basename "$input_path")"
        done < "$temp_file"
    else
        print_warning "No jobs found matching the specified filters."
    fi
    
    rm -f "$temp_file"
}

# Function to show job status
show_job_status() {
    local job_id="$1"
    local filter_status="$2"
    local days="$3"
    local verbose="$4"
    
    if [[ -n "$job_id" ]]; then
        # Show specific job
        if [[ ! -f "$JOB_FILE" ]]; then
            print_error "No jobs file found."
            return
        fi
        
        local job_info=$(grep "^$job_id|" "$JOB_FILE" || true)
        if [[ -z "$job_info" ]]; then
            print_error "Job $job_id not found."
            return
        fi
        
        IFS='|' read -r job_id env date name type status input_path output_dir config_file <<< "$job_info"
        
        print_header "Job Status: $job_id"
        echo "Job ID: $job_id"
        echo "Environment: $env"
        echo "Date: $date"
        echo "Name: $name"
        echo "Type: $type"
        echo "Input: $input_path"
        echo "Output: $output_dir"
        echo "Config: $config_file"
        
        # Get current status
        current_status=$(get_job_status "$job_id" "$env")
        mapped_status=$(map_status "$current_status" "$env")
        echo "Status: $mapped_status"
        
        # Show log file if exists
        local log_file="$LOG_DIR/${name}_${job_id}.out"
        if [[ -f "$log_file" ]]; then
            echo "Log file: $log_file"
            if [[ "$verbose" == true ]]; then
                echo "Recent log output:"
                tail -20 "$log_file"
            fi
        fi
        
        # Show error file if exists
        local err_file="$LOG_DIR/${name}_${job_id}.err"
        if [[ -f "$err_file" ]]; then
            echo "Error file: $err_file"
            if [[ "$verbose" == true ]]; then
                echo "Recent errors:"
                tail -10 "$err_file"
            fi
        fi
        
    else
        # Show all jobs with optional filters
        list_jobs "" "$filter_status" "" "$days"
    fi
}

# Function to show job logs
show_job_logs() {
    local job_id="$1"
    local lines="${2:-50}"
    
    if [[ -z "$job_id" ]]; then
        print_error "Job ID is required for logs command."
        return
    fi
    
    if [[ ! -f "$JOB_FILE" ]]; then
        print_error "No jobs file found."
        return
    fi
    
    local job_info=$(grep "^$job_id|" "$JOB_FILE" || true)
    if [[ -z "$job_info" ]]; then
        print_error "Job $job_id not found."
        return
    fi
    
    IFS='|' read -r job_id env date name type status input_path output_dir config_file <<< "$job_info"
    
    local log_file="$LOG_DIR/${name}_${job_id}.out"
    local err_file="$LOG_DIR/${name}_${job_id}.err"
    
    print_header "Logs for Job: $job_id ($name)"
    
    if [[ -f "$log_file" ]]; then
        echo "=== STDOUT Log ==="
        tail -n "$lines" "$log_file"
    else
        print_warning "No stdout log file found: $log_file"
    fi
    
    if [[ -f "$err_file" ]]; then
        echo ""
        echo "=== STDERR Log ==="
        tail -n "$lines" "$err_file"
    fi
}

# Function to cancel a job
cancel_job() {
    local job_id="$1"
    
    if [[ -z "$job_id" ]]; then
        print_error "Job ID is required for cancel command."
        return
    fi
    
    if [[ ! -f "$JOB_FILE" ]]; then
        print_error "No jobs file found."
        return
    fi
    
    local job_info=$(grep "^$job_id|" "$JOB_FILE" || true)
    if [[ -z "$job_info" ]]; then
        print_error "Job $job_id not found."
        return
    fi
    
    IFS='|' read -r job_id env date name type status input_path output_dir config_file <<< "$job_info"
    
    print_header "Cancelling Job: $job_id"
    
    case "$env" in
        "slurm")
            scancel "$job_id"
            print_success "Job $job_id cancelled in SLURM"
            ;;
        "pbs")
            qdel "$job_id"
            print_success "Job $job_id cancelled in PBS"
            ;;
        "local")
            kill "$job_id" 2>/dev/null || true
            print_success "Job $job_id cancelled locally"
            ;;
        *)
            print_error "Cannot cancel job in environment: $env"
            ;;
    esac
    
    # Update job status in tracking file
    sed -i "s/^$job_id|.*|RUNNING|/$job_id|$env|$date|$name|$type|CANCELLED|$input_path|$output_dir|$config_file/" "$JOB_FILE"
}

# Function to clean completed jobs
clean_jobs() {
    local job_id="$1"
    
    print_header "Cleaning Jobs"
    
    if [[ -n "$job_id" ]]; then
        # Clean specific job
        if [[ ! -f "$JOB_FILE" ]]; then
            print_error "No jobs file found."
            return
        fi
        
        local job_info=$(grep "^$job_id|" "$JOB_FILE" || true)
        if [[ -z "$job_info" ]]; then
            print_error "Job $job_id not found."
            return
        fi
        
        IFS='|' read -r job_id env date name type status input_path output_dir config_file <<< "$job_info"
        
        # Remove log files
        rm -f "$LOG_DIR/${name}_${job_id}.out" "$LOG_DIR/${name}_${job_id}.err"
        
        # Remove from tracking file
        sed -i "/^$job_id|/d" "$JOB_FILE"
        
        print_success "Cleaned job $job_id"
        
    else
        # Clean all completed jobs
        if [[ ! -f "$JOB_FILE" ]]; then
            print_warning "No jobs file found."
            return
        fi
        
        local cleaned=0
        while IFS='|' read -r job_id env date name type status input_path output_dir config_file; do
            if [[ "$status" == "COMPLETED" || "$status" == "FAILED" || "$status" == "CANCELLED" ]]; then
                # Remove log files
                rm -f "$LOG_DIR/${name}_${job_id}.out" "$LOG_DIR/${name}_${job_id}.err"
                cleaned=$((cleaned + 1))
            fi
        done < "$JOB_FILE"
        
        # Remove completed jobs from tracking file
        sed -i '/|COMPLETED|/d; /|FAILED|/d; /|CANCELLED|/d' "$JOB_FILE"
        
        print_success "Cleaned $cleaned completed jobs"
    fi
}

# Function to watch job progress
watch_job() {
    local job_id="$1"
    
    if [[ -z "$job_id" ]]; then
        print_error "Job ID is required for watch command."
        return
    fi
    
    if [[ ! -f "$JOB_FILE" ]]; then
        print_error "No jobs file found."
        return
    fi
    
    local job_info=$(grep "^$job_id|" "$JOB_FILE" || true)
    if [[ -z "$job_info" ]]; then
        print_error "Job $job_id not found."
        return
    fi
    
    IFS='|' read -r job_id env date name type status input_path output_dir config_file <<< "$job_info"
    
    local log_file="$LOG_DIR/${name}_${job_id}.out"
    
    print_header "Watching Job: $job_id ($name)"
    print_info "Press Ctrl+C to stop watching"
    
    if [[ -f "$log_file" ]]; then
        tail -f "$log_file"
    else
        print_warning "No log file found: $log_file"
        print_info "Job may not have started yet or logs are in a different location."
    fi
}

# Function to show job summary
show_summary() {
    print_header "Job Summary"
    
    if [[ ! -f "$JOB_FILE" ]]; then
        print_warning "No jobs file found. No jobs have been submitted yet."
        return
    fi
    
    local total_jobs=0
    local running_jobs=0
    local completed_jobs=0
    local failed_jobs=0
    local pending_jobs=0
    
    # Count jobs by status
    while IFS='|' read -r job_id env date name type status input_path output_dir config_file; do
        total_jobs=$((total_jobs + 1))
        
        # Get current status for running/pending jobs
        if [[ "$status" == "RUNNING" || "$status" == "PENDING" ]]; then
            current_status=$(get_job_status "$job_id" "$env")
            status=$(map_status "$current_status" "$env")
        fi
        
        case "$status" in
            "RUNNING") running_jobs=$((running_jobs + 1)) ;;
            "COMPLETED") completed_jobs=$((completed_jobs + 1)) ;;
            "FAILED") failed_jobs=$((failed_jobs + 1)) ;;
            "PENDING"|"QUEUED") pending_jobs=$((pending_jobs + 1)) ;;
        esac
    done < "$JOB_FILE"
    
    echo "Total Jobs: $total_jobs"
    echo "Running: $running_jobs"
    echo "Pending: $pending_jobs"
    echo "Completed: $completed_jobs"
    echo "Failed: $failed_jobs"
    
    # Show jobs by type
    echo ""
    print_header "Jobs by Type"
    awk -F'|' '{print $5}' "$JOB_FILE" | sort | uniq -c | while read count type; do
        echo "$type: $count"
    done
    
    # Show jobs by environment
    echo ""
    print_header "Jobs by Environment"
    awk -F'|' '{print $2}' "$JOB_FILE" | sort | uniq -c | while read count env; do
        echo "$env: $count"
    done
}

# Function to track a new job
track_job() {
    local job_id="$1"
    local env="$2"
    local name="$3"
    local type="$4"
    local input_path="$5"
    local output_dir="$6"
    local config_file="$7"
    
    local date=$(date +%Y-%m-%d)
    local status="RUNNING"
    
    echo "$job_id|$env|$date|$name|$type|$status|$input_path|$output_dir|$config_file" >> "$JOB_FILE"
}

# Main function
main() {
    local command="$1"
    shift || true
    
    case "$command" in
        "list")
            local filter_env=""
            local filter_status=""
            local filter_type=""
            local days=""
            
            while [[ $# -gt 0 ]]; do
                case $1 in
                    --compute-env) filter_env="$2"; shift 2 ;;
                    --status) filter_status="$2"; shift 2 ;;
                    --job-type) filter_type="$2"; shift 2 ;;
                    --days) days="$2"; shift 2 ;;
                    *) shift ;;
                esac
            done
            
            list_jobs "$filter_env" "$filter_status" "$filter_type" "$days"
            ;;
            
        "status")
            local job_id=""
            local filter_status=""
            local days=""
            local verbose=false
            
            while [[ $# -gt 0 ]]; do
                case $1 in
                    --status) filter_status="$2"; shift 2 ;;
                    --days) days="$2"; shift 2 ;;
                    --verbose) verbose=true; shift ;;
                    *) job_id="$1"; shift ;;
                esac
            done
            
            show_job_status "$job_id" "$filter_status" "$days" "$verbose"
            ;;
            
        "logs")
            local job_id="$1"
            local lines="$2"
            show_job_logs "$job_id" "$lines"
            ;;
            
        "cancel")
            local job_id="$1"
            cancel_job "$job_id"
            ;;
            
        "clean")
            local job_id="$1"
            clean_jobs "$job_id"
            ;;
            
        "watch")
            local job_id="$1"
            watch_job "$job_id"
            ;;
            
        "summary")
            show_summary
            ;;
            
        "help"|"--help"|"-h"|"")
            show_usage
            ;;
            
        *)
            print_error "Unknown command: $command"
            show_usage
            exit 1
            ;;
    esac
}

# Export track_job function for use by submit_job.sh
export -f track_job

# Run main function
main "$@" 