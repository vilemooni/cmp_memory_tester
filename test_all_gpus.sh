#!/bin/bash

# ============================================================
# CUDA VRAM Tester - Multi-GPU Test Script
#
# Usage:
#   ./test_all_gpus.sh
#   ./test_all_gpus.sh <VRAM_GiB>
#   ./test_all_gpus.sh <VRAM_GiB> <ROUNDS>
#
# Examples:
#   ./test_all_gpus.sh
#   ./test_all_gpus.sh 60
#   ./test_all_gpus.sh 60 3
#
# Defaults:
#   VRAM_GiB = 60
#   ROUNDS    = 1
# ============================================================


# ------------------------------------------------------------
# Configuration
# ------------------------------------------------------------

VRAM_GB=${1:-60}
ROUNDS=${2:-1}


# ------------------------------------------------------------
# Check that vram_test exists
# ------------------------------------------------------------

if [ ! -x "./vram_test" ]; then

    echo "ERROR: ./vram_test not found or not executable."
    echo ""

    echo "Compile the CUDA tester first:"
    echo ""

    echo "  nvcc -O3 -o vram_test vram_test.cu"

    echo ""

    echo "Then make this script executable:"
    echo ""

    echo "  chmod +x test_all_gpus.sh"

    exit 1
fi


# ------------------------------------------------------------
# Check nvidia-smi
# ------------------------------------------------------------

if ! command -v nvidia-smi >/dev/null 2>&1; then

    echo "ERROR: nvidia-smi not found."

    echo ""
    echo "Make sure the NVIDIA driver is installed and working."

    exit 1
fi


# ------------------------------------------------------------
# Detect NVIDIA GPUs
# ------------------------------------------------------------

GPU_LIST=$(
    nvidia-smi \
        --query-gpu=index \
        --format=csv,noheader \
        2>/dev/null
)


if [ -z "$GPU_LIST" ]; then

    echo "ERROR: No NVIDIA GPUs detected."

    exit 1
fi


# Count GPUs

GPU_COUNT=$(echo "$GPU_LIST" | wc -l)


# ------------------------------------------------------------
# Validate VRAM argument
# ------------------------------------------------------------

if ! [[ "$VRAM_GB" =~ ^[0-9]+([.][0-9]+)?$ ]]; then

    echo "ERROR: VRAM amount must be a number."

    echo ""

    echo "Example:"
    echo "  ./test_all_gpus.sh 60"

    exit 1
fi


# ------------------------------------------------------------
# Validate rounds argument
# ------------------------------------------------------------

if ! [[ "$ROUNDS" =~ ^[0-9]+$ ]] ||
   [ "$ROUNDS" -lt 1 ]; then

    echo "ERROR: Number of rounds must be a positive integer."

    echo ""

    echo "Example:"
    echo "  ./test_all_gpus.sh 60 3"

    exit 1
fi


# ------------------------------------------------------------
# Create log file
# ------------------------------------------------------------

LOG="vram_test_results_$(date +%Y%m%d_%H%M%S).txt"


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

echo "============================================" | tee "$LOG"

echo "       CUDA VRAM MULTI-GPU TEST" | tee -a "$LOG"

echo "============================================" | tee -a "$LOG"

echo "Date:        $(date)" | tee -a "$LOG"

echo "GPU count:   $GPU_COUNT" | tee -a "$LOG"

echo "VRAM test:   ${VRAM_GB} GiB per GPU" | tee -a "$LOG"

echo "Test rounds: $ROUNDS" | tee -a "$LOG"

echo "============================================" | tee -a "$LOG"


# ------------------------------------------------------------
# Show detected GPU information
# ------------------------------------------------------------

echo "" | tee -a "$LOG"

echo "Detected GPUs:" | tee -a "$LOG"

nvidia-smi \
    --query-gpu=index,name,memory.total,pci.bus_id \
    --format=csv \
    | tee -a "$LOG"


echo "" | tee -a "$LOG"


# ------------------------------------------------------------
# Initialize counters
# ------------------------------------------------------------

PASS_COUNT=0

FAIL_COUNT=0


# ------------------------------------------------------------
# Test each GPU
# ------------------------------------------------------------

for GPU in $GPU_LIST; do


    echo "============================================" | tee -a "$LOG"

    echo "TESTING GPU $GPU" | tee -a "$LOG"

    echo "============================================" | tee -a "$LOG"


    # --------------------------------------------------------
    # Get GPU information
    # --------------------------------------------------------

    GPU_INFO=$(
        nvidia-smi \
            --id="$GPU" \
            --query-gpu=index,name,memory.total,pci.bus_id \
            --format=csv,noheader \
            2>/dev/null
    )


    echo "GPU: $GPU_INFO" | tee -a "$LOG"


    GPU_FAILED=0


    # --------------------------------------------------------
    # Run requested number of rounds
    # --------------------------------------------------------

    for ROUND in $(seq 1 "$ROUNDS"); do


        echo "" | tee -a "$LOG"

        echo "Round $ROUND / $ROUNDS" | tee -a "$LOG"

        echo "Testing ${VRAM_GB} GiB..." | tee -a "$LOG"

        echo "" | tee -a "$LOG"


        # ----------------------------------------------------
        # Run VRAM test
        # ----------------------------------------------------

        ./vram_test \
            "$GPU" \
            "$VRAM_GB" \
            2>&1 | tee -a "$LOG"


        RESULT=${PIPESTATUS[0]}


        # ----------------------------------------------------
        # Check result
        # ----------------------------------------------------

        if [ "$RESULT" -ne 0 ]; then


            GPU_FAILED=1


            echo "" | tee -a "$LOG"

            echo "ROUND $ROUND: FAIL" | tee -a "$LOG"


            # Stop testing this GPU

            break


        else


            echo "" | tee -a "$LOG"

            echo "ROUND $ROUND: PASS" | tee -a "$LOG"


        fi


    done


    # --------------------------------------------------------
    # GPU final result
    # --------------------------------------------------------

    echo "" | tee -a "$LOG"


    if [ "$GPU_FAILED" -eq 0 ]; then


        echo "GPU $GPU: PASS" | tee -a "$LOG"


        PASS_COUNT=$(
            (PASS_COUNT + 1)
        )


    else


        echo "GPU $GPU: FAIL" | tee -a "$LOG"


        FAIL_COUNT=$(
            (FAIL_COUNT + 1)
        )


    fi


    echo "" | tee -a "$LOG"


done


# ------------------------------------------------------------
# Final summary
# ------------------------------------------------------------

echo "============================================" | tee -a "$LOG"

echo "             FINAL SUMMARY" | tee -a "$LOG"

echo "============================================" | tee -a "$LOG"


echo "GPUs detected: $GPU_COUNT" | tee -a "$LOG"

echo "VRAM tested:   ${VRAM_GB} GiB per GPU" | tee -a "$LOG"

echo "Test rounds:   $ROUNDS" | tee -a "$LOG"


echo "" | tee -a "$LOG"


echo "PASS: $PASS_COUNT" | tee -a "$LOG"

echo "FAIL: $FAIL_COUNT" | tee -a "$LOG"


echo "" | tee -a "$LOG"


# ------------------------------------------------------------
# Overall result
# ------------------------------------------------------------

if [ "$FAIL_COUNT" -eq 0 ]; then


    echo "OVERALL RESULT: PASS" | tee -a "$LOG"


else


    echo "OVERALL RESULT: FAIL" | tee -a "$LOG"


fi


echo "" | tee -a "$LOG"


echo "Finished: $(date)" | tee -a "$LOG"


echo ""


echo "============================================"

echo "Test complete."

echo ""

echo "Log saved to:"

echo "$LOG"

echo "============================================"
