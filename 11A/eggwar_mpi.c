#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <time.h>
#include <string.h>
#include <mpi.h>
#include <unistd.h> // For sleep function

// Field size
#define N 100
#define E 10 // Percentage of eggs to place
// Maximum eggs cannot exceed 99% of field capacity
#define MAX_EGGS (N * N * 99 / 100)

// Function prototypes
void initialize_field(int field[N][N]);
void populate_field(int field[N][N], int num_eggs);
void print_field(int field[N][N]);
int search_eggs(int field[N][N], int start_row, int end_row);

int main(int argc, char *argv[]) {
    int field[N][N];
    int num_eggs = N * N / E;  // Default: populate with 10% eggs
    bool verbose = false;
    int rank, size, eggs_found = 0, total_eggs_found = 0;
    
    // Initialize MPI environment
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Check for verbose flag
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-v") == 0) {
            verbose = true;
        }
    }
    
    // Master process initializes and distributes the field
    if (rank == 0) {
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
        
        // Start timer for search
        double start_time = MPI_Wtime();
        
        // Distribute field to all processes
        for (int i = 1; i < size; i++) {
            MPI_Send(&field, N * N, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
        
        // Master also searches a portion of the field
        int rows_per_process = N / size;
        int start_row = 0;
        int end_row = rows_per_process;
        
        eggs_found = search_eggs(field, start_row, end_row);
        total_eggs_found = eggs_found;
        
        // Collect results from worker processes
        for (int i = 1; i < size; i++) {
            MPI_Recv(&eggs_found, 1, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_eggs_found += eggs_found;
        }
        
        // Stop timer
        double end_time = MPI_Wtime();
        
        printf("Total eggs found: %d out of %d\n", total_eggs_found, num_eggs);
        printf("Search completed in %f seconds with %d processes\n", 
               end_time - start_time, size);
    }
    else { // Worker processes
        // Receive field from master
        MPI_Recv(&field, N * N, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Determine which portion of the field to search
        int rows_per_process = N / size;
        int start_row = rank * rows_per_process;
        int end_row = (rank == size - 1) ? N : start_row + rows_per_process;
        
        // Search for eggs
        eggs_found = search_eggs(field, start_row, end_row);
        
        // Report results back to master
        MPI_Send(&eggs_found, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
    
    MPI_Finalize();
    return 0;
}

// Search for eggs in a portion of the field
int search_eggs(int field[N][N], int start_row, int end_row) {
    int eggs_found = 0;
    
    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < N; j++) {
            // Simulate some processing time
            usleep(1000); // 1ms delay
            
            if (field[i][j] == 1) {
                eggs_found++;
            }
        }
    }
    
    printf("Process searched rows %d-%d and found %d eggs\n", 
           start_row, end_row-1, eggs_found);
    
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