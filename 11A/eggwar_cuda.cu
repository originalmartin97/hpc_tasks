#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>
#include <unistd.h>
#include <cuda_runtime.h>

// Field size
#define N 100
#define E 10 // Percentage of eggs to place
// Maximum eggs cannot exceed 99% of field capacity
#define MAX_EGGS (N * N * 99 / 100)
// CUDA configuration
#define BLOCK_SIZE 16

// Function prototypes
void initialize_field(int field[N][N]);
void populate_field(int field[N][N], int num_eggs);
void print_field(int field[N][N]);
int search_eggs_cpu(int field[N][N], int start_row, int end_row, bool verbose);

// CUDA kernel for searching eggs
__global__ void search_eggs_kernel(int *d_field, int *d_results, int field_width, int field_height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check if within field bounds
    if (row < field_height && col < field_width) {
        // If egg found, atomically increment the counter
        if (d_field[row * field_width + col] == 1) {
            atomicAdd(d_results, 1);
        }
    }
}

// CUDA error checking helper
#define cudaCheckError() {                                          \
    cudaError_t e = cudaGetLastError();                             \
    if (e != cudaSuccess) {                                         \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
               cudaGetErrorString(e));                              \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
}

int main(int argc, char *argv[]) {
    int field[N][N];
    int num_eggs = N * N / E;  // Default: populate with 10% eggs
    bool verbose = false;
    int eggs_found = 0;
    
    // Check for verbose flag
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0) {
            verbose = true;
        }
    }
    
    // Seed random number generator
    srand(time(NULL));
    
    // Initialize field with zeros
    initialize_field(field);
    
    // Populate with eggs
    populate_field(field, num_eggs);
    
    // Print field if verbose mode enabled
    if (verbose) {
        print_field(field);
    }
    
    printf("Field of size %dx%d populated with %d Easter eggs\n", N, N, num_eggs);
    printf("CUDA implementation using GPU for searching eggs\n");
    
    // Start timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    
    // Allocate memory on the GPU
    int *d_field, *d_result;
    cudaMalloc((void**)&d_field, N * N * sizeof(int));
    cudaCheckError();
    cudaMalloc((void**)&d_result, sizeof(int));
    cudaCheckError();
    
    // Initialize the result counter to 0
    cudaMemset(d_result, 0, sizeof(int));
    cudaCheckError();
    
    // Copy the field to GPU
    cudaMemcpy(d_field, field, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaCheckError();
    
    // Calculate grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    if (verbose) {
        printf("Launching CUDA kernel with grid(%d, %d) and block(%d, %d)\n", 
               gridDim.x, gridDim.y, blockDim.x, blockDim.y);
    }
    
    // Launch the kernel
    search_eggs_kernel<<<gridDim, blockDim>>>(d_field, d_result, N, N);
    cudaCheckError();
    
    // Synchronize to wait for kernel to finish
    cudaDeviceSynchronize();
    cudaCheckError();
    
    // Copy results back from the GPU
    cudaMemcpy(&eggs_found, d_result, sizeof(int), cudaMemcpyDeviceToHost);
    cudaCheckError();
    
    // Stop timer
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    printf("Total eggs found: %d out of %d\n", eggs_found, num_eggs);
    printf("Search completed in %f seconds with CUDA\n", milliseconds / 1000.0f);
    
    // Free GPU memory
    cudaFree(d_field);
    cudaFree(d_result);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    // Verify results with CPU version if in verbose mode
    if (verbose) {
        int cpu_eggs_found = search_eggs_cpu(field, 0, N, verbose);
        printf("CPU verification: %d eggs\n", cpu_eggs_found);
        if (cpu_eggs_found != eggs_found) {
            printf("Warning: GPU result differs from CPU result!\n");
        } else {
            printf("GPU and CPU results match.\n");
        }
    }
    
    return 0;
}

// CPU version of egg search for verification
int search_eggs_cpu(int field[N][N], int start_row, int end_row, bool verbose) {
    int eggs_found = 0;
    
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < N; j++) {
            if (field[i][j] == 1) {
                eggs_found++;
            }
        }
    }
    
    if (verbose) {
        printf("CPU searched rows %d-%d and found %d eggs total\n", 
               start_row, end_row-1, eggs_found);
    }
    
    return eggs_found;
}

// Initialize the field with zeros
void initialize_field(int field[N][N]) {
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            field[i][j] = 0;
        }
    }
}

// Populate the field with eggs at random positions
void populate_field(int field[N][N], int num_eggs) {
    if (num_eggs > MAX_EGGS) {
        printf("Warning: Requested too many eggs. Limiting to maximum allowed (%d)\n", MAX_EGGS);
        num_eggs = MAX_EGGS;
    }
    
    int eggs_placed = 0;
    while (eggs_placed < num_eggs) {
        // Generate random coordinates
        int i = rand() % N;
        int j = rand() % N;
        
        // Place egg only if position is empty
        if (field[i][j] == 0) {
            field[i][j] = 1;
            eggs_placed++;
        }
    }
}

// Print the field contents with more options
void print_field(int field[N][N]) {
    printf("Field contents (%dx%d):\n", N, N);
    
    // If N is too large, print a summary instead
    if (N > 50) {
        printf("Field is too large for full display.\n");
        printf("Top-left corner (10x10):\n");
        for (int i = 0; i < 10; i++) {
            for (int j = 0; j < 10; j++) {
                printf("%d", field[i][j]);
            }
            printf("\n");
        }
        return;
    }
    
    // Print each row separately
    for (int i = 0; i < N; i++) {
        printf("Row %3d: ", i);
        for (int j = 0; j < N; j++) {
            printf("%d", field[i][j]);
        }
        printf("\n");
    }
}