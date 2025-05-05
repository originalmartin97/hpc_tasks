#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// CUDA error checking helper
#define cudaCheckError() {                                          \
    cudaError_t e = cudaGetLastError();                             \
    if (e != cudaSuccess) {                                         \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
               cudaGetErrorString(e));                              \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
}

// CUDA kernel to calculate vector lengths
__global__ void calculateVectorLengths(float *vector1, float *vector2, float *length1, float *length2, int size) {
    float sum1 = 0.0f;
    float sum2 = 0.0f;
    
    for (int i = 0; i < size; i++) {
        sum1 += vector1[i] * vector1[i];
        sum2 += vector2[i] * vector2[i];
    }
    
    *length1 = sqrt(sum1);
    *length2 = sqrt(sum2);
}

// CUDA kernel to calculate scalar product
__global__ void calculateScalarProduct(float *vector1, float *vector2, float *result, int size) {
    float sum = 0.0f;
    
    for (int i = 0; i < size; i++) {
        sum += vector1[i] * vector2[i];
    }
    
    *result = sum;
}

int main(int argc, char *argv[]) {
    // Check if we have enough arguments (we need at least 6 values for two 3D vectors)
    if (argc < 7) {
        printf("Usage: %s x1 y1 z1 x2 y2 z2\n", argv[0]);
        return 1;
    }
    
    int size = 3; // We're working with 3D vectors
    
    // Allocate memory for vectors on host
    float *h_vector1 = (float *)malloc(size * sizeof(float));
    float *h_vector2 = (float *)malloc(size * sizeof(float));
    float h_length1, h_length2, h_scalar_product;
    
    // Parse arguments
    for (int i = 0; i < size; i++) {
        h_vector1[i] = atof(argv[i+1]);
        h_vector2[i] = atof(argv[i+4]);
    }
    
    // Print input vectors
    printf("Vector 1: (%.2f, %.2f, %.2f)\n", h_vector1[0], h_vector1[1], h_vector1[2]);
    printf("Vector 2: (%.2f, %.2f, %.2f)\n", h_vector2[0], h_vector2[1], h_vector2[2]);
    
    // Allocate memory for vectors on device
    float *d_vector1, *d_vector2;
    float *d_length1, *d_length2, *d_scalar_product;
    
    cudaMalloc((void **)&d_vector1, size * sizeof(float));
    cudaCheckError();
    cudaMalloc((void **)&d_vector2, size * sizeof(float));
    cudaCheckError();
    cudaMalloc((void **)&d_length1, sizeof(float));
    cudaCheckError();
    cudaMalloc((void **)&d_length2, sizeof(float));
    cudaCheckError();
    cudaMalloc((void **)&d_scalar_product, sizeof(float));
    cudaCheckError();
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Start timing
    cudaEventRecord(start);
    
    // Copy vectors from host to device
    cudaMemcpy(d_vector1, h_vector1, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError();
    cudaMemcpy(d_vector2, h_vector2, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaCheckError();
    
    // Launch kernels
    calculateVectorLengths<<<1, 1>>>(d_vector1, d_vector2, d_length1, d_length2, size);
    cudaCheckError();
    calculateScalarProduct<<<1, 1>>>(d_vector1, d_vector2, d_scalar_product, size);
    cudaCheckError();
    
    // Copy results from device to host
    cudaMemcpy(&h_length1, d_length1, sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError();
    cudaMemcpy(&h_length2, d_length2, sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError();
    cudaMemcpy(&h_scalar_product, d_scalar_product, sizeof(float), cudaMemcpyDeviceToHost);
    cudaCheckError();
    
    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Print results
    printf("Vector 1 Length: %.6f\n", h_length1);
    printf("Vector 2 Length: %.6f\n", h_length2);
    printf("Scalar Product: %.6f\n", h_scalar_product);
    printf("Execution Time: %.6f milliseconds\n", milliseconds);
    
    // Free device memory
    cudaFree(d_vector1);
    cudaFree(d_vector2);
    cudaFree(d_length1);
    cudaFree(d_length2);
    cudaFree(d_scalar_product);
    
    // Free host memory
    free(h_vector1);
    free(h_vector2);
    
    // Destroy CUDA events
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}