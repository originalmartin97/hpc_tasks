# How to compile and run
- To compile the program:
```bash
nvcc -o vector_scalar vector_scalar.cu
```

- To run the program with two 3D vectors:
```bash
./vector_scalar 1.0 2.0 3.0 4.0 5.0 6.0
```
This will calculate the lengths and scalar product of the vectors (1,2,3) and (4,5,6).

# Program explanation
The program performs the following steps:

1. Parse command-line arguments to get the coordinates of two 3D vectors.

2. Allocate memory on both host and device for the vectors and results.

3. Copy the vector data from host to device.

4. Execute two CUDA kernels:
    - calculateVectorLengths computes the lengths of both vectors
    - calculateScalarProduct computes the dot product of the two vectors

5. Copy the results back from device to host.

6. Print the results including the vector lengths, scalar product, and execution time in milliseconds.

7. Free all allocated memory on both device and host.

The program uses CUDA events for precise timing and includes error checking at each CUDA API call.