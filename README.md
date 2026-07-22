CUDA VRAM Tester

A CUDA-based GPU memory testing tool for Linux and NVIDIA GPUs.

This project is designed to verify that GPU VRAM can be allocated, written to, read back, and validated without memory errors.

It is especially useful for testing GPUs with modified or expanded VRAM configurations, where the reported VRAM capacity needs to be validated with an actual memory test.

Features
Tests GPU VRAM using CUDA
Tests multiple memory patterns
Allocates a user-defined amount of VRAM
Writes test patterns to VRAM
Reads the data back
Validates the data against the expected patterns
Supports testing individual GPUs
Supports automatic detection and testing of all NVIDIA GPUs
Supports multiple test rounds
Saves test results to a log file
Does not modify NVIDIA drivers or GPU firmware
Requirements
Linux
NVIDIA GPU
Working NVIDIA proprietary driver
nvidia-smi
CUDA Toolkit with nvcc
Bash

The CUDA Toolkit is required to compile vram_test.cu.

The NVIDIA driver must support the CUDA runtime requirements of the compiled application.

The CUDA Toolkit version used to compile the program does not necessarily need to match the CUDA version reported by nvidia-smi.

HiveOS users

The NVIDIA driver is normally available in HiveOS, but the CUDA Toolkit and nvcc compiler may not be installed by default.

The CUDA Toolkit is required only to compile vram_test.cu.

After the program has been compiled, the resulting vram_test binary can be run without nvcc.

The installer does not install or modify NVIDIA drivers or the CUDA Toolkit automatically.

If nvcc is not installed, install a compatible CUDA Toolkit for your NVIDIA driver before compiling the program.

Installation

Clone or download this repository and enter the project directory.

git clone https://github.com/vilemooni/cmp_memory_tester.git
cd cmp_memory_tester
Option 1: Automatic setup and compilation

The easiest method is to use the included installation script.

First, make the script executable:

chmod +x install.sh

Then run the installer:

./install.sh

The installation script:

Checks that the system is running Linux
Checks that nvidia-smi is available
Detects NVIDIA GPUs
Checks that nvcc is available
Displays the CUDA compiler version
Compiles vram_test.cu
Makes test_all_gpus.sh executable

The installer does not install or modify NVIDIA drivers or the CUDA Toolkit.

After successful installation, the following commands are available:

./vram_test <GPU_ID> <VRAM_GiB>

and:

./test_all_gpus.sh
Option 2: Manual compilation

If you prefer to compile the program manually, make sure the CUDA Toolkit and nvcc are installed.

Compile the CUDA VRAM tester:

nvcc -O3 -o vram_test vram_test.cu

Make the multi-GPU test script executable:

chmod +x test_all_gpus.sh

The program is now ready to use.

Testing a Single GPU

The VRAM tester takes two arguments:

./vram_test <GPU_ID> <VRAM_GiB>

For example, to test 60 GiB of VRAM on GPU 0:

./vram_test 0 60

The test performs multiple memory pattern tests.

Each test performs:

VRAM allocation
Memory pattern write
CUDA synchronization
Memory readback
Pattern validation

A successful test ends with:

FINAL RESULT: PASS

A failed test ends with:

FINAL RESULT: FAIL
Testing All GPUs

The multi-GPU script automatically detects the number of NVIDIA GPUs using nvidia-smi.

Default test

Run:

./test_all_gpus.sh

The default configuration tests 60 GiB on every detected GPU.

Specify VRAM amount

To specify a different amount of VRAM:

./test_all_gpus.sh 32

This tests 32 GiB on every detected GPU.

Run multiple test rounds

To run multiple test rounds:

./test_all_gpus.sh 60 3

This tests 60 GiB on every detected GPU and performs three complete test rounds per GPU.

If a GPU fails a test round, testing stops for that GPU and the script continues with the next GPU.

Test Results

The multi-GPU test creates a timestamped log file:

vram_test_results_YYYYMMDD_HHMMSS.txt

The final summary reports the number of GPUs that passed and failed.

Example:

============================================
             FINAL SUMMARY
============================================
GPUs detected: 9
VRAM tested:   60 GiB per GPU
Test rounds:   3

PASS: 9
FAIL: 0

OVERALL RESULT: PASS

The test script returns a successful result only when all detected GPUs pass the test.

Important Notes

A successful test demonstrates that the tested VRAM allocation could be:

Allocated by CUDA
Written to
Read back
Validated against the expected data patterns

A successful test does not guarantee that every physical memory cell has been tested or that the GPU will remain stable under every possible workload.

For higher confidence, run multiple test rounds and consider performing the test under different GPU operating conditions.

The amount of VRAM tested should leave some memory available for the CUDA runtime and driver.

For a GPU reporting approximately 64 GB of VRAM, testing 60 GiB is a practical starting point.

Example: 64 GB GPU

For a GPU reporting 64 GB of VRAM:

./vram_test 0 60

To test all detected GPUs:

./test_all_gpus.sh 60

For additional confidence:

./test_all_gpus.sh 60 3
Disclaimer

This software is provided for testing and diagnostic purposes.

The authors make no guarantees regarding the accuracy, completeness, or suitability of test results for any specific purpose.

Use at your own risk.

License

This project is released under the MIT License.
