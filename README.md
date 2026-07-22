# CUDA VRAM Tester

A CUDA-based GPU memory testing tool for Linux and NVIDIA GPUs.

This project is designed to verify that GPU VRAM can be allocated, written to, read back, and validated without memory errors.

It is especially useful for testing GPUs with modified or expanded VRAM configurations, where the reported VRAM capacity needs to be validated with an actual memory test.

## Features

* Tests GPU VRAM using CUDA
* Tests multiple memory patterns
* Allocates a user-defined amount of VRAM
* Writes test patterns to VRAM
* Reads the data back
* Validates the data against the expected patterns
* Supports testing individual GPUs
* Supports automatic detection and testing of all NVIDIA GPUs
* Supports multiple test rounds
* Saves test results to a log file
* Does not modify NVIDIA drivers or GPU firmware

## Requirements

- Linux
- NVIDIA GPU
- Working NVIDIA driver
- `nvidia-smi`
- CUDA Toolkit with `nvcc`
- Bash

### HiveOS users

The NVIDIA driver is normally available in HiveOS, but the CUDA Toolkit and `nvcc` compiler may not be installed by default.

The CUDA Toolkit is required **only to compile** `vram_test.cu`.

After the program has been compiled, the resulting `vram_test` binary can be run without `nvcc`.

If `nvcc` is not installed, install a compatible CUDA Toolkit for your NVIDIA driver and then run:

```bash
./install.sh

## Installation

Clone or download this repository and enter the project directory.

Compile the CUDA VRAM tester:

```bash
nvcc -O3 -o vram_test vram_test.cu
```

Make the multi-GPU test script executable:

```bash
chmod +x test_all_gpus.sh
```

You can also use the installation script:

```bash
chmod +x install.sh
./install.sh
```

## Testing a Single GPU

The VRAM tester takes two arguments:

```text
./vram_test <GPU_ID> <VRAM_GiB>
```

For example, to test 60 GiB of VRAM on GPU 0:

```bash
./vram_test 0 60
```

The test performs multiple memory pattern tests.

Each test performs:

1. VRAM allocation
2. Memory pattern write
3. CUDA synchronization
4. Memory readback
5. Pattern validation

A successful test ends with:

```text
FINAL RESULT: PASS
```

A failed test ends with:

```text
FINAL RESULT: FAIL
```

## Testing All GPUs

The multi-GPU script automatically detects the number of NVIDIA GPUs using `nvidia-smi`.

Run the default test:

```bash
./test_all_gpus.sh
```

The default configuration tests 60 GiB on every detected GPU.

To specify a different amount of VRAM:

```bash
./test_all_gpus.sh 32
```

This tests 32 GiB on every detected GPU.

To run multiple test rounds:

```bash
./test_all_gpus.sh 60 3
```

This tests 60 GiB on every detected GPU and performs three complete test rounds per GPU.

## Test Results

The multi-GPU test creates a timestamped log file:

```text
vram_test_results_YYYYMMDD_HHMMSS.txt
```

The final summary reports the number of GPUs that passed and failed.

Example:

```text
============================================
             FINAL SUMMARY
============================================
GPUs detected: 9
VRAM tested:   60 GiB per GPU
Test rounds:   3

PASS: 9
FAIL: 0

OVERALL RESULT: PASS
```

## Important Notes

A successful test demonstrates that the tested VRAM allocation could be:

* Allocated by CUDA
* Written to
* Read back
* Validated against the expected data patterns

A successful test does **not** guarantee that every physical memory cell has been tested or that the GPU will remain stable under every possible workload.

For higher confidence, run multiple test rounds and consider performing the test under different GPU operating conditions.

The amount of VRAM tested should leave some memory available for the CUDA runtime and driver.

For a GPU reporting approximately 64 GB of VRAM, testing 60 GiB is a practical starting point.

## Example: 64 GB GPU

For a GPU reporting 64 GB of VRAM:

```bash
./vram_test 0 60
```

To test all detected GPUs:

```bash
./test_all_gpus.sh 60
```

For additional confidence:

```bash
./test_all_gpus.sh 60 3
```

## Disclaimer

This software is provided for testing and diagnostic purposes.

The authors make no guarantees regarding the accuracy, completeness, or suitability of test results for any specific purpose.

Use at your own risk.

## License

This project is released under the MIT License.
