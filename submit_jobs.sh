#!/bin/bash
# PBS Job Array Mimic with Queue Management and State Persistence
# Handles queue limits and can resume after interruption
#
# Usage: ./submit_jobs.sh [options]
# Options:
#   -m MAX_JOBS    Maximum jobs in queue
#   -w WAIT_TIME   Wait time between checks in seconds (default: 120)
#   -r             Resume from previous state if available
#   -c             Clear state and start fresh
#   -d             Daemon mode (run continuously in background)

# Configuration
HOME_DIR=/grand/insitu/cohanlon
MAX_QUEUED_JOBS=20  # Default max jobs in queue
WAIT_TIME=120       # Seconds between queue checks
DAEMON_MODE=false
RESUME_MODE=false
CLEAR_STATE=false
LOG_FILE="$HOME_DIR/job_submission.log"
STATE_FILE="$HOME_DIR/.pbs_submission_state"
LOCK_FILE="/tmp/pbs_submission.lock"

# Parse command line arguments
while getopts "m:w:p:drch" opt; do
    case $opt in
        m) MAX_QUEUED_JOBS=$OPTARG ;;
        w) WAIT_TIME=$OPTARG ;;
        d) DAEMON_MODE=true ;;
        r) RESUME_MODE=true ;;
        c) CLEAR_STATE=true ;;
        h) echo "Usage: $0 [-m max_jobs] [-w wait_time] [-r] [-c] [-d]"
           echo "  -r: Resume from previous state"
           echo "  -c: Clear state and start fresh"
           exit 0 ;;
        *) echo "Invalid option. Use -h for help."
           exit 1 ;;
    esac
done

# Function to log messages
log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Function to acquire lock (prevent multiple instances)
acquire_lock() {
    local timeout=10
    local elapsed=0
    
    while [ $elapsed -lt $timeout ]; do
        if mkdir "$LOCK_FILE" 2>/dev/null; then
            echo $$ > "$LOCK_FILE/pid"
            log_message "Lock acquired (PID: $$)"
            return 0
        fi
        
        # Check if the process holding the lock is still running
        if [ -f "$LOCK_FILE/pid" ]; then
            local pid=$(cat "$LOCK_FILE/pid" 2>/dev/null)
            if ! kill -0 "$pid" 2>/dev/null; then
                log_message "Removing stale lock from PID $pid"
                rm -rf "$LOCK_FILE"
                continue
            fi
        fi
        
        sleep 1
        ((elapsed++))
    done
    
    log_message "ERROR: Could not acquire lock. Another instance may be running."
    exit 1
}

# Function to release lock
release_lock() {
    rm -rf "$LOCK_FILE"
    log_message "Lock released"
}

# Function to save state
save_state() {
    local run_index=$1
    local status=$2  # submitted, failed, or completed
    local job_id=$3
    
    # Create state file if it doesn't exist
    if [ ! -f "$STATE_FILE" ]; then
        echo "# PBS Submission State File" > "$STATE_FILE"
        echo "# Format: run_index|status|job_id|timestamp" >> "$STATE_FILE"
        echo "TOTAL_RUNS=$num_runs" >> "$STATE_FILE"
        echo "CONFIG_HASH=$(echo "$HOME_DIR/alcf_kan_inr" | md5sum | cut -d' ' -f1)" >> "$STATE_FILE"
    fi
    
    # Check if this run_index already exists (for updates)
    if grep -q "^$run_index|" "$STATE_FILE"; then
        # Update existing entry
        sed -i "/^$run_index|/c\\$run_index|$status|$job_id|$(date '+%Y-%m-%d %H:%M:%S')" "$STATE_FILE"
    else
        # Add new entry
        echo "$run_index|$status|$job_id|$(date '+%Y-%m-%d %H:%M:%S')" >> "$STATE_FILE"
    fi
}

# Function to load state
load_state() {
    if [ ! -f "$STATE_FILE" ]; then
        log_message "No state file found"
        return 1
    fi
    
    log_message "Loading state from $STATE_FILE"
    
    # Read total runs from state
    local saved_total=$(grep "^TOTAL_RUNS=" "$STATE_FILE" | cut -d'=' -f2)
    local saved_hash=$(grep "^CONFIG_HASH=" "$STATE_FILE" | cut -d'=' -f2)
    local current_hash=$(echo "$HOME_DIR/alcf_kan_inr" | md5sum | cut -d' ' -f1)
    
    # Verify configuration hasn't changed
    if [ "$saved_hash" != "$current_hash" ]; then
        log_message "WARNING: Configuration has changed since last run"
        read -p "Continue anyway? (y/n): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
    
    # Get list of already submitted run indices
    submitted_runs=$(grep -E "^[0-9]+\|submitted" "$STATE_FILE" | cut -d'|' -f1 | sort -n)
    failed_runs=$(grep -E "^[0-9]+\|failed" "$STATE_FILE" | cut -d'|' -f1 | sort -n)
    
    local submitted_count=$(echo "$submitted_runs" | grep -c "^[0-9]" || echo 0)
    local failed_count=$(echo "$failed_runs" | grep -c "^[0-9]" || echo 0)
    
    log_message "State loaded: $submitted_count submitted, $failed_count failed"
    
    return 0
}

# Function to check if run was already submitted
was_submitted() {
    local run_index=$1
    
    if [ -f "$STATE_FILE" ]; then
        if grep -q "^$run_index|submitted" "$STATE_FILE"; then
            return 0
        fi
    fi
    
    return 1
}

# Function to get current number of queued jobs
get_queued_jobs() {
    local count
    count=$(qstat -u $USER | tail -n +6 | wc -l)
    echo $count
}

# Function to get job states
get_job_states() {
    qstat -u $USER | tail -n +6 | awk '{print $10}' | sort | uniq -c
}

# Function to submit a single job
submit_job() {
    local run_index=$1
    local job_id
    
    log_message "Submitting run index: $run_index"

    # Submit job and capture job ID
    job_id=$(qsub -v PBS_ARRAY_INDEX=$run_index -N job_$run_index $HOME_DIR/alcf_kan_inr/run_scripts/bm.sh 2>&1)

    if [ $? -eq 0 ]; then
        log_message "Successfully submitted job $job_id for run index $run_index"
        save_state $run_index "submitted" "$job_id"
        return 0
    else
        log_message "ERROR: Failed to submit job for run index $run_index: $job_id"
        save_state $run_index "failed" "NA"
        return 1
    fi
}

# Function to wait for queue slot
wait_for_slot() {
    local current_jobs
    
    while true; do
        current_jobs=$(get_queued_jobs)
        
        if [ $current_jobs -lt $MAX_QUEUED_JOBS ]; then
            return 0
        fi
        
        log_message "Queue full ($current_jobs/$MAX_QUEUED_JOBS jobs). Waiting $WAIT_TIME seconds..."
        sleep $WAIT_TIME
    done
}

# Function to show summary
show_summary() {
    if [ -f "$STATE_FILE" ]; then
        local submitted_count=$(grep -c "^[0-9]*|submitted" "$STATE_FILE" || echo 0)
        local failed_count=$(grep -c "^[0-9]*|failed" "$STATE_FILE" || echo 0)
        local total=$(grep "^TOTAL_RUNS=" "$STATE_FILE" | cut -d'=' -f2)
        
        log_message "=== SUBMISSION SUMMARY ==="
        log_message "Total runs: $total"
        log_message "Submitted: $submitted_count"
        log_message "Failed: $failed_count"
        log_message "Remaining: $((total - submitted_count - failed_count))"
        
        if [ $failed_count -gt 0 ]; then
            log_message "Failed run indices:"
            grep "^[0-9]*|failed" "$STATE_FILE" | cut -d'|' -f1 | tr '\n' ' '
            echo
        fi
    fi
}

# Cleanup function for graceful shutdown
cleanup() {
    log_message "Received interrupt signal. Cleaning up..."
    show_summary
    release_lock
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM EXIT

# Main execution
main() {
    # Acquire lock to prevent multiple instances
    acquire_lock
    
    log_message "Starting PBS job submission manager"
    log_message "Configuration: MAX_JOBS=$MAX_QUEUED_JOBS, WAIT_TIME=$WAIT_TIME"

    # Clear state if requested
    if [ "$CLEAR_STATE" = true ] && [ -f "$STATE_FILE" ]; then
        log_message "Clearing previous state"
        rm "$STATE_FILE"
    fi
    
    # Source conda environment
    source $HOME_DIR/miniconda3/etc/profile.d/conda.sh
    
    # Get total number of runs
    num_runs=$(conda run -n alcf_kan_inr python $HOME_DIR/alcf_kan_inr/benchmark.py -cn config dataset=beechnut count_configs=True)
    
    if [ -z "$num_runs" ] || [ "$num_runs" -eq 0 ]; then
        log_message "ERROR: Could not determine number of runs"
        exit 1
    fi
    
    log_message "Total runs to submit: $num_runs"
    
    # Load previous state if resuming
    if [ "$RESUME_MODE" = true ]; then
        if load_state; then
            log_message "Resuming from previous state"
        else
            log_message "No valid state to resume from, starting fresh"
        fi
    fi
    
    # Track submission progress
    submitted=0
    failed=0
    skipped=0
    
    # Submit jobs with queue management
    for run_index in $(seq 0 $((num_runs - 1))); do
        # Check if already submitted (when resuming)
        if was_submitted $run_index; then
            ((skipped++))
            log_message "Skipping run index $run_index (already submitted)"
            continue
        fi
        
        # Wait for available slot
        wait_for_slot
        
        # Submit job
        if submit_job $run_index; then
            ((submitted++))
        else
            ((failed++))
            log_message "WARNING: Continuing despite failure"
        fi
        
        # Show progress
        if [ $(((submitted + skipped) % 10)) -eq 0 ]; then
            log_message "Progress: $submitted new, $skipped resumed, $failed failed out of $num_runs total"
            log_message "Current queue status:"
            get_job_states | while read line; do
                log_message "  $line"
            done
        fi
        
        # Small delay to avoid overwhelming the scheduler
        sleep 1
    done
    
    log_message "Submission complete: $submitted new submissions, $skipped already submitted, $failed failed"
    show_summary
    
    # Optional: Monitor until all jobs complete
    if [ "$DAEMON_MODE" = true ]; then
        log_message "Daemon mode: Monitoring job completion..."
        
        while true; do
            current_jobs=$(get_queued_jobs)
            
            if [ $current_jobs -eq 0 ]; then
                log_message "All jobs completed!"
                break
            fi
            
            log_message "Jobs still in queue: $current_jobs"
            get_job_states | while read line; do
                log_message "  $line"
            done
            
            sleep $WAIT_TIME
        done
    fi
}

# Run main function
main

# Exit successfully
exit 0