#include <stdio.h>
#include <stdlib.h>

// CUDA error checking helper
#define cudaCheckError() {                                          \
    cudaError_t e = cudaGetLastError();                             \
    if (e != cudaSuccess) {                                         \
        printf("CUDA error %s:%d: %s\n", __FILE__, __LINE__,        \
               cudaGetErrorString(e));                              \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
}

// CUDA kernel to generate color gradient
__global__ void generateColorGradient(unsigned char* image, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = (y * width + x) * 3; // 3 bytes per pixel (RGB)
        
        // Calculate normalized position from left (0.0) to right (1.0)
        float factor = (float)x / (width - 1);
        
        // Red decreases from left to right
        image[idx] = (unsigned char)(255.0f * (1.0f - factor));
        
        // Green varies (creates a yellow->green->cyan transition)
        image[idx + 1] = (unsigned char)(255.0f * (factor < 0.5f ? factor * 2.0f : (1.0f - factor) * 2.0f));
        
        // Blue increases from left to right
        image[idx + 2] = (unsigned char)(255.0f * factor);
    }
}

int main(int argc, char* argv[]) {
    // Define image dimensions
    int width = 800;
    int height = 600;
    
    // Parse command line arguments if provided
    if (argc > 2) {
        width = atoi(argv[1]);
        height = atoi(argv[2]);
    }
    
    if (width <= 0 || height <= 0) {
        printf("Invalid dimensions. Using defaults (800x600).\n");
        width = 800;
        height = 600;
    }
    
    int imageSize = width * height * 3; // RGB (3 bytes per pixel)
    
    // Allocate memory for image on host
    unsigned char* h_image = (unsigned char*)malloc(imageSize);
    if (!h_image) {
        fprintf(stderr, "Error: Host memory allocation failed\n");
        return 1;
    }
    
    // Allocate memory for image on device
    unsigned char* d_image;
    cudaMalloc((void**)&d_image, imageSize);
    cudaCheckError();
    
    // Define grid and block dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                 (height + blockSize.y - 1) / blockSize.y);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Start timing
    cudaEventRecord(start);
    
    // Launch kernel
    generateColorGradient<<<gridSize, blockSize>>>(d_image, width, height);
    cudaCheckError();
    
    // Stop timing
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy result from device to host
    cudaMemcpy(h_image, d_image, imageSize, cudaMemcpyDeviceToHost);
    cudaCheckError();
    
    // Save image as PPM file
    const char* filename = "gradient.ppm";
    FILE* fp = fopen(filename, "wb");
    if (!fp) {
        fprintf(stderr, "Error: Could not open output file\n");
        free(h_image);
        cudaFree(d_image);
        return 1;
    }
    
    // Write PPM header
    fprintf(fp, "P6\n%d %d\n255\n", width, height);
    
    // Write image data
    fwrite(h_image, 1, imageSize, fp);
    
    // Close file
    fclose(fp);
    
    printf("Color gradient generated successfully:\n");
    printf("- Resolution: %d x %d pixels\n", width, height);
    printf("- Output file: %s\n", filename);
    printf("- Execution time: %.3f milliseconds\n", milliseconds);
    
    // Free memory
    free(h_image);
    cudaFree(d_image);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return 0;
}