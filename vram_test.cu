#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstdint>


#define CUDA_CHECK(call)                                      \
do {                                                          \
    cudaError_t err = (call);                                 \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr,                                         \
                "\nCUDA ERROR: %s\n",                          \
                cudaGetErrorString(err));                     \
        return 1;                                             \
    }                                                         \
} while (0)


__global__ void fill_pattern(
    uint64_t *data,
    size_t elements,
    uint64_t pattern,
    int mode)
{
    size_t idx =
        (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= elements)
        return;

    if (mode == 0) {
        data[idx] = pattern;
    }
    else if (mode == 1) {
        data[idx] = ~pattern;
    }
    else {
        data[idx] =
            pattern ^
            (idx * 0x9E3779B97F4A7C15ULL);
    }
}


__global__ void check_pattern(
    const uint64_t *data,
    size_t elements,
    uint64_t pattern,
    int mode,
    unsigned long long *errors)
{
    size_t idx =
        (size_t)blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= elements)
        return;

    uint64_t expected;

    if (mode == 0) {
        expected = pattern;
    }
    else if (mode == 1) {
        expected = ~pattern;
    }
    else {
        expected =
            pattern ^
            (idx * 0x9E3779B97F4A7C15ULL);
    }

    if (data[idx] != expected) {
        atomicAdd(errors, 1ULL);
    }
}


int main(int argc, char **argv)
{
    // --------------------------------------------------------
    // Check command line arguments
    // --------------------------------------------------------

    if (argc < 3) {

        printf(
            "CUDA VRAM Tester\n\n"
            "Usage:\n"
            "  %s <GPU_ID> <VRAM_GiB>\n\n"
            "Example:\n"
            "  %s 0 60\n",
            argv[0],
            argv[0]
        );

        return 1;
    }


    // --------------------------------------------------------
    // Parse GPU ID
    // --------------------------------------------------------

    char *endptr = nullptr;

    long gpu_long =
        strtol(
            argv[1],
            &endptr,
            10
        );

    if (*endptr != '\0' ||
        gpu_long < 0) {

        fprintf(
            stderr,
            "ERROR: Invalid GPU ID: %s\n",
            argv[1]
        );

        return 1;
    }

    int gpu =
        (int)gpu_long;


    // --------------------------------------------------------
    // Parse requested VRAM amount
    // --------------------------------------------------------

    endptr = nullptr;

    double requested_gb =
        strtod(
            argv[2],
            &endptr
        );

    if (*endptr != '\0' ||
        requested_gb <= 0) {

        fprintf(
            stderr,
            "ERROR: Invalid VRAM amount: %s\n",
            argv[2]
        );

        return 1;
    }


    // --------------------------------------------------------
    // Get CUDA GPU count
    // --------------------------------------------------------

    int gpu_count = 0;

    cudaError_t count_result =
        cudaGetDeviceCount(
            &gpu_count
        );

    if (count_result != cudaSuccess) {

        fprintf(
            stderr,
            "ERROR: Cannot enumerate CUDA GPUs.\n"
        );

        fprintf(
            stderr,
            "CUDA error: %s\n",
            cudaGetErrorString(
                count_result
            )
        );

        return 1;
    }


    if (gpu >= gpu_count) {

        fprintf(
            stderr,
            "ERROR: GPU %d does not exist.\n",
            gpu
        );

        fprintf(
            stderr,
            "Detected CUDA GPUs: %d\n",
            gpu_count
        );

        return 1;
    }


    // --------------------------------------------------------
    // Select GPU
    // --------------------------------------------------------

    CUDA_CHECK(
        cudaSetDevice(
            gpu
        )
    );


    // --------------------------------------------------------
    // Get GPU properties
    // --------------------------------------------------------

    cudaDeviceProp prop;

    CUDA_CHECK(
        cudaGetDeviceProperties(
            &prop,
            gpu
        )
    );


    // --------------------------------------------------------
    // Get current VRAM information
    // --------------------------------------------------------

    size_t free_mem = 0;
    size_t total_mem = 0;

    CUDA_CHECK(
        cudaMemGetInfo(
            &free_mem,
            &total_mem
        )
    );


    // --------------------------------------------------------
    // Calculate requested memory
    // --------------------------------------------------------

    size_t test_bytes =
        (size_t)(
            requested_gb *
            1024.0 *
            1024.0 *
            1024.0
        );


    // --------------------------------------------------------
    // Print GPU information
    // --------------------------------------------------------

    printf(
        "========================================\n"
    );

    printf(
        "CUDA VRAM TEST\n"
    );

    printf(
        "========================================\n"
    );

    printf(
        "GPU ID:       %d\n",
        gpu
    );

    printf(
        "GPU Name:     %s\n",
        prop.name
    );

    printf(
        "PCI Bus ID:   %02X:%02X.0\n",
        prop.pciBusID,
        prop.pciDeviceID
    );

    printf(
        "CUDA GPUs:    %d\n",
        gpu_count
    );

    printf(
        "Total VRAM:   %.2f GiB\n",
        total_mem /
        (1024.0 *
         1024.0 *
         1024.0)
    );

    printf(
        "Free VRAM:    %.2f GiB\n",
        free_mem /
        (1024.0 *
         1024.0 *
         1024.0)
    );

    printf(
        "Test VRAM:    %.2f GiB\n",
        requested_gb
    );

    printf(
        "========================================\n"
    );


    // --------------------------------------------------------
    // Check that requested VRAM is available
    // --------------------------------------------------------

    if (test_bytes >= free_mem) {

        fprintf(
            stderr,
            "\nERROR: Not enough free VRAM.\n"
        );

        fprintf(
            stderr,
            "Requested: %.2f GiB\n",
            requested_gb
        );

        fprintf(
            stderr,
            "Available: %.2f GiB\n",
            free_mem /
            (1024.0 *
             1024.0 *
             1024.0)
        );

        return 1;
    }


    // --------------------------------------------------------
    // Allocate VRAM
    // --------------------------------------------------------

    printf(
        "\nAllocating %.2f GiB...\n",
        requested_gb
    );

    uint64_t *d_data = nullptr;

    cudaError_t alloc_result =
        cudaMalloc(
            (void **)&d_data,
            test_bytes
        );


    if (alloc_result != cudaSuccess) {

        fprintf(
            stderr,
            "\nRESULT: FAIL\n"
        );

        fprintf(
            stderr,
            "CUDA memory allocation failed:\n"
        );

        fprintf(
            stderr,
            "%s\n",
            cudaGetErrorString(
                alloc_result
            )
        );

        return 1;
    }


    // --------------------------------------------------------
    // Calculate number of 64-bit elements
    // --------------------------------------------------------

    size_t elements =
        test_bytes /
        sizeof(uint64_t);


    // --------------------------------------------------------
    // Allocate error counter
    // --------------------------------------------------------

    unsigned long long *d_errors = nullptr;

    cudaError_t error_alloc =
        cudaMalloc(
            (void **)&d_errors,
            sizeof(
                unsigned long long
            )
        );


    if (error_alloc != cudaSuccess) {

        fprintf(
            stderr,
            "\nERROR: Could not allocate error counter.\n"
        );

        fprintf(
            stderr,
            "%s\n",
            cudaGetErrorString(
                error_alloc
            )
        );

        cudaFree(
            d_data
        );

        return 1;
    }


    // --------------------------------------------------------
    // Configure CUDA kernel
    // --------------------------------------------------------

    int threads = 256;

    int blocks =
        (int)(
            (elements +
             threads -
             1) /
            threads
        );


    // --------------------------------------------------------
    // Test patterns
    // --------------------------------------------------------

    uint64_t patterns[3] = {

        0xDEADBEEFCAFEBABEULL,

        0xAAAAAAAA55555555ULL,

        0x0123456789ABCDEFULL
    };


    const char *names[3] = {

        "Constant pattern",

        "Inverted pattern",

        "Address-dependent pattern"
    };


    bool failed = false;


    // --------------------------------------------------------
    // Run all memory tests
    // --------------------------------------------------------

    for (int test = 0;
         test < 3;
         test++) {

        printf(
            "\n[%d/3] %s\n",
            test + 1,
            names[test]
        );


        // Reset error counter

        CUDA_CHECK(
            cudaMemset(
                d_errors,
                0,
                sizeof(
                    unsigned long long
                )
            )
        );


        // Write VRAM

        printf(
            "Writing VRAM...\n"
        );


        fill_pattern<<<
            blocks,
            threads
        >>>(
            d_data,
            elements,
            patterns[test],
            test
        );


        CUDA_CHECK(
            cudaGetLastError()
        );


        CUDA_CHECK(
            cudaDeviceSynchronize()
        );


        // Read and validate VRAM

        printf(
            "Reading and validating...\n"
        );


        check_pattern<<<
            blocks,
            threads
        >>>(
            d_data,
            elements,
            patterns[test],
            test,
            d_errors
        );


        CUDA_CHECK(
            cudaGetLastError()
        );


        CUDA_CHECK(
            cudaDeviceSynchronize()
        );


        // Copy error count back to CPU

        unsigned long long errors = 0;


        CUDA_CHECK(
            cudaMemcpy(
                &errors,
                d_errors,
                sizeof(
                    unsigned long long
                ),
                cudaMemcpyDeviceToHost
            )
        );


        if (errors == 0) {

            printf(
                "PASS - No errors detected\n"
            );

        }
        else {

            printf(
                "FAIL - %llu errors detected\n",
                errors
            );

            failed = true;
        }
    }


    // --------------------------------------------------------
    // Free allocated memory
    // --------------------------------------------------------

    cudaFree(
        d_errors
    );

    cudaFree(
        d_data
    );


    // --------------------------------------------------------
    // Final result
    // --------------------------------------------------------

    printf(
        "\n========================================\n"
    );


    if (failed) {

        printf(
            "FINAL RESULT: FAIL\n"
        );

        printf(
            "Tested VRAM: %.2f GiB\n",
            requested_gb
        );

        printf(
            "Memory errors were detected.\n"
        );

        printf(
            "========================================\n"
        );

        return 1;
    }


    printf(
        "FINAL RESULT: PASS\n"
    );

    printf(
        "Tested VRAM: %.2f GiB\n",
        requested_gb
    );

    printf(
        "All memory patterns passed.\n"
    );

    printf(
        "========================================\n"
    );


    return 0;
}
