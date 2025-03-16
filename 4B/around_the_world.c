#define _GNU_SOURCE
#include <stdio.h>
#include <sched.h>
#include <unistd.h>
#include <mpi.h>
#include <string.h>

#define MAX_CHAR_OF_NAME 100

int main(int argc, char ** argv)
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Get cpu number and node name
    int cpu = sched_getcpu();
    char node_name[MAX_CHAR_OF_NAME];
    gethostname(node_name, MAX_CHAR_OF_NAME);
    
    if (rank == 0) {
        // Rank 0 collects and prints info from all ranks (including itself)
        char node_name_of_rank[MAX_CHAR_OF_NAME];
        int cpu_of_rank;
        
        // Print rank 0's
        printf("Hello world from: Rank %d, cpu #%d, node %s \n", 0, cpu, node_name);
        
        // Then receive and print info from each rank in order
        for (int src = 1; src < size; src++) {
            MPI_Recv(node_name_of_rank, 100, MPI_CHAR, src, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Recv(&cpu_of_rank, 1, MPI_INT, src, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("Hello world from: Rank %d, cpu #%d, node %s \n", src, cpu_of_rank, node_name_of_rank);
        }
    } else {
        // All other ranks send their info to rank 0
        MPI_Send(node_name, 100, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
        MPI_Send(&cpu, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
    }
    
    MPI_Finalize();
    return 0;
}