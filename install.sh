#!/bin/bash

# ============================================================
# CUDA VRAM Tester - Installation Script
#
# This script:
#   1. Checks for NVIDIA GPUs
#   2. Checks for nvidia-smi
#   3. Checks for nvcc
#   4. Checks CUDA compiler version
#   5. Compiles vram_test.cu
#   6. Makes test_all_gpus.sh executable
#
# This script does NOT install or modify NVIDIA drivers.
# ============================================================


set -e


# ------------------------------------------------------------
# Colors
# ------------------------------------------------------------

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'


# ------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------

print_ok()
{
    echo -e "${GREEN}[OK]${NC} $1"
}


print_warning()
{
    echo -e "${YELLOW}[WARNING]${NC} $1"
}


print_error()
{
    echo -e "${RED}[ERROR]${NC} $1"
}


# ------------------------------------------------------------
# Header
# ------------------------------------------------------------

echo ""
echo "============================================"
echo "       CUDA VRAM TESTER INSTALLER"
echo "============================================"
echo ""


# ------------------------------------------------------------
# Check operating system
# ------------------------------------------------------------

if [ "$(uname -s)" != "Linux" ]; then

    print_error "This installer currently supports Linux only."

    exit 1

fi


print_ok "Operating system: Linux"


# ------------------------------------------------------------
# Check nvidia-smi
# ------------------------------------------------------------

if ! command -v nvidia-smi >/dev/null 2>&1; then

    print_error "nvidia-smi was not found."

    echo ""
    echo "Make sure the NVIDIA driver is installed and working."
    echo ""

    exit 1

fi


print_ok "nvidia-smi found"


# ------------------------------------------------------------
# Check NVIDIA GPUs
# ------------------------------------------------------------

GPU_COUNT=$(
    nvidia-smi \
        --query-gpu=index \
        --format=csv,noheader \
        2>/dev/null \
        | wc -l
)


if [ "$GPU_COUNT" -eq 0 ]; then

    print_error "No NVIDIA GPUs detected."

    echo ""
    echo "Please check that:"
    echo "  - The NVIDIA driver is installed"
    echo "  - The GPUs are detected by the system"
    echo ""

    exit 1

fi


print_ok "Detected NVIDIA GPUs: $GPU_COUNT"


# ------------------------------------------------------------
# Show GPUs
# ------------------------------------------------------------

echo ""

nvidia-smi \
    --query-gpu=index,name,memory.total,pci.bus_id \
    --format=csv


echo ""


# ------------------------------------------------------------
# Check nvcc
# ------------------------------------------------------------

if ! command -v nvcc >/dev/null 2>&1; then

    print_error "nvcc was not found."

    echo ""
    echo "The CUDA Toolkit is required to compile the VRAM tester."
    echo ""
    echo "Install the CUDA Toolkit and run this installer again."
    echo ""

    exit 1

fi


print_ok "nvcc found"


# ------------------------------------------------------------
# Show CUDA compiler version
# ------------------------------------------------------------

echo ""

echo "CUDA compiler version:"

nvcc --version

echo ""


# ------------------------------------------------------------
# Check source file
# ------------------------------------------------------------

if [ ! -f "vram_test.cu" ]; then

    print_error "vram_test.cu was not found."

    echo ""
    echo "Run this installer from the project directory."
    echo ""

    exit 1

fi


print_ok "vram_test.cu found"


# ------------------------------------------------------------
# Compile CUDA VRAM tester
# ------------------------------------------------------------

echo ""

echo "Compiling CUDA VRAM tester..."

echo ""

nvcc \
    -O3 \
    -o vram_test \
    vram_test.cu


if [ ! -x "./vram_test" ]; then

    print_error "Compilation failed."

    exit 1

fi


print_ok "vram_test compiled successfully"


# ------------------------------------------------------------
# Make multi-GPU test executable
# ------------------------------------------------------------

if [ -f "test_all_gpus.sh" ]; then

    chmod +x test_all_gpus.sh

    print_ok "test_all_gpus.sh is executable"

else

    print_warning "test_all_gpus.sh was not found."

fi


# ------------------------------------------------------------
# Installation summary
# ------------------------------------------------------------

echo ""

echo "============================================"

echo "       INSTALLATION COMPLETE"

echo "============================================"

echo ""

echo "Detected GPUs: $GPU_COUNT"

echo ""

echo "Available commands:"

echo ""

echo "Test one GPU:"
echo "  ./vram_test 0 60"

echo ""

echo "Test all GPUs:"
echo "  ./test_all_gpus.sh"

echo ""

echo "Test all GPUs with 3 rounds:"
echo "  ./test_all_gpus.sh 60 3"

echo ""

echo "============================================"

echo ""
