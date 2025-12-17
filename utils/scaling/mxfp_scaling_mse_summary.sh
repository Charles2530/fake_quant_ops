#!/bin/bash
# =============================================================================
# MXFP Scaling MSE Summary Script (Multi-threaded)
# 批量处理指定目录下的所有 tensor 文件
# 对每个 tensor 进行 fp4_e2m1 格式的 MSE 测试（scale vs half_scale）
#
# Usage:
#   ./mxfp_scaling_mse_summary.sh [INPUT_DIR] [OUTPUT_DIR] [JOBS]
#
# Arguments:
#   INPUT_DIR    - Input directory containing tensor files (default: data/bf16)
#   OUTPUT_DIR   - Output directory for logs (default: ./draw/scaling_mse_analysis)
#   JOBS         - Number of parallel jobs (default: number of CPU cores)
#
# Example:
#   ./mxfp_scaling_mse_summary.sh data/bf16 ./draw/scaling_mse_analysis 8
# =============================================================================

# =============================================================================
# Configuration
# =============================================================================

# Parse command line arguments
INPUT_DIR="${1:-data/bf16}"
OUTPUT_BASE_DIR="${2:-./draw/scaling_mse_analysis}"
JOBS="${3:-$(nproc 2>/dev/null || echo 4)}"  # Number of parallel jobs (default: CPU cores or 4)

# Element format to test (fixed to fp4_e2m1)
ELEM_FORMAT="fp4_e2m1"

# Script directory (where mxfp_scaling_mse.py is located)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TEST_SCRIPT="$SCRIPT_DIR/mxfp_scaling_mse.py"

# Default test parameters
SCALE_BITS=8
BLOCK_SIZE=32
AXES=-1
MAX_SCALE_EXP=10
MIN_SCALE_EXP=-10

# =============================================================================
# Validation
# =============================================================================

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "❌ Error: Input directory does not exist: $INPUT_DIR"
    exit 1
fi

# Check if test script exists
if [ ! -f "$TEST_SCRIPT" ]; then
    echo "❌ Error: Test script not found: $TEST_SCRIPT"
    exit 1
fi

# =============================================================================
# Find all tensor files
# =============================================================================

echo "=============================================================================="
echo "MXFP Scaling MSE Summary - Batch Processing (Multi-threaded)"
echo "=============================================================================="
echo "Input directory: $INPUT_DIR"
echo "Output directory: $OUTPUT_BASE_DIR"
echo "Element format: $ELEM_FORMAT"
echo "Parallel jobs: $JOBS"
echo "Block size: $BLOCK_SIZE"
echo "Scale bits: $SCALE_BITS"
echo "=============================================================================="
echo ""

# Find all .pt files in the input directory (recursively)
TENSOR_FILES=($(find "$INPUT_DIR" -type f -name "*.pt" | sort))

if [ ${#TENSOR_FILES[@]} -eq 0 ]; then
    echo "⚠️  Warning: No .pt tensor files found in $INPUT_DIR"
    exit 0
fi

TOTAL_TENSORS=${#TENSOR_FILES[@]}

echo "Found $TOTAL_TENSORS tensor file(s)"
echo "Will run $TOTAL_TENSORS test(s) in total"
echo ""

# Create output directory
mkdir -p "$OUTPUT_BASE_DIR"

# =============================================================================
# Process each tensor (Parallel)
# =============================================================================

cd "$PROJECT_ROOT"

# Create temporary directory for job tracking
TMP_DIR=$(mktemp -d)
trap "rm -rf '$TMP_DIR'" EXIT

# Create output lock file for synchronized output
OUTPUT_LOCK="$TMP_DIR/output.lock"
touch "$OUTPUT_LOCK"

# Check if flock is available
HAS_FLOCK=$(command -v flock >/dev/null 2>&1 && echo "yes" || echo "no")

# Export variables for subprocesses
export PROJECT_ROOT TEST_SCRIPT OUTPUT_BASE_DIR TMP_DIR TOTAL_TENSORS OUTPUT_LOCK HAS_FLOCK
export ELEM_FORMAT SCALE_BITS BLOCK_SIZE AXES MAX_SCALE_EXP MIN_SCALE_EXP

# Build task list
TASK_ID=0
declare -a TASKS

for tensor_file in "${TENSOR_FILES[@]}"; do
    TASK_ID=$((TASK_ID + 1))
    TASKS+=("$tensor_file|$TASK_ID")
done

# Process tasks in parallel
echo "Starting parallel processing with $JOBS concurrent jobs..."
echo ""

SUCCESSFUL_TESTS=0
FAILED_TESTS=0
FAILED_FILES=()

# Process all tasks using job control
for task in "${TASKS[@]}"; do
    IFS='|' read -r tensor_file test_id <<< "$task"
    tensor_name=$(basename "$tensor_file" .pt)
    # Create subdirectory structure: output_dir/tensor_name/
    output_dir="$OUTPUT_BASE_DIR/$tensor_name"
    mkdir -p "$output_dir"
    log_file="$output_dir/${tensor_name}_${ELEM_FORMAT}.log"
    result_file="$TMP_DIR/result_${test_id}.txt"
    
    # Wait if we've reached the job limit
    while [ $(jobs -r | wc -l) -ge $JOBS ]; do
        sleep 0.1
    done
    
    # Start background job
    (
        {
            echo "[$test_id/$TOTAL_TENSORS] Testing: $tensor_name with $ELEM_FORMAT"
            echo "──────────────────────────────────────────────────────────────────────"
            echo "Input file: $tensor_file"
            echo "Output log: $log_file"
            echo ""
            
            # Run the test script
            if python3 "$TEST_SCRIPT" \
                --input_tensor "$tensor_file" \
                --elem-format "$ELEM_FORMAT" \
                --scale-bits "$SCALE_BITS" \
                --block-size "$BLOCK_SIZE" \
                --axes "$AXES" \
                --max-scale-exp "$MAX_SCALE_EXP" \
                --min-scale-exp "$MIN_SCALE_EXP" \
                > "$log_file" 2>&1; then
                echo "SUCCESS|$tensor_name" > "$result_file"
                echo "✅ Success: $tensor_name"
            else
                echo "FAILED|$tensor_name" > "$result_file"
                echo "❌ Failed: $tensor_name"
                echo "   Check log: $log_file"
            fi
            echo ""
        } | {
            # Use flock to synchronize output (if available)
            if [ "$HAS_FLOCK" = "yes" ]; then
                flock -x 200
                cat
            else
                # Fallback: simple output without locking
                cat
            fi
        } 200>"$OUTPUT_LOCK" 2>/dev/null || cat
    ) &
done

# Wait for all background jobs to complete
wait

# Collect results
for task in "${TASKS[@]}"; do
    IFS='|' read -r tensor_file test_id <<< "$task"
    result_file="$TMP_DIR/result_${test_id}.txt"
    tensor_name=$(basename "$tensor_file" .pt)
    
    if [ -f "$result_file" ]; then
        IFS='|' read -r status message <<< "$(cat "$result_file")"
        if [ "$status" = "SUCCESS" ]; then
            SUCCESSFUL_TESTS=$((SUCCESSFUL_TESTS + 1))
        else
            FAILED_TESTS=$((FAILED_TESTS + 1))
            FAILED_FILES+=("$message")
        fi
    else
        FAILED_TESTS=$((FAILED_TESTS + 1))
        FAILED_FILES+=("$tensor_name")
    fi
done

# =============================================================================
# Summary
# =============================================================================

echo "=============================================================================="
echo "FINAL SUMMARY"
echo "=============================================================================="
echo "Total tensors processed: $TOTAL_TENSORS"
echo "Successful: $SUCCESSFUL_TESTS"
echo "Failed: $FAILED_TESTS"
echo ""

if [ $FAILED_TESTS -eq 0 ]; then
    echo "🎉 All tests completed successfully!"
    echo ""
    echo "Results saved to: $OUTPUT_BASE_DIR/"
    echo ""
    echo "Each tensor has its own subdirectory with:"
    echo "  - Log files: {tensor_name}_${ELEM_FORMAT}.log"
    echo ""
    echo "Log files contain:"
    echo "  - Tensor information (shape, dtype, range)"
    echo "  - MSE comparison (scale vs half_scale)"
    echo "  - Per-block selection statistics"
    echo "  - Best MSE results"
    exit 0
else
    echo "⚠️  Some tests failed. Failed tests:"
    for failed in "${FAILED_FILES[@]}"; do
        echo "  - $failed"
    done
    echo ""
    echo "Check individual log files in $OUTPUT_BASE_DIR/ for details."
    exit 1
fi
