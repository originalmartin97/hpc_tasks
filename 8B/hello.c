#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>
#include <unistd.h> // For gethostname
#include <sched.h>

int main(int argc, char *argv[]) {
    int rank, size, provided;
    char hostname[256];
    
    // Initialize MPI with thread support
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    // Get hostname of the machine
    gethostname(hostname, sizeof(hostname));
    
    // OpenMP parallel region
    #pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        int num_threads = omp_get_num_threads();
        int cpu_id = sched_getcpu();
        
        // Use critical section to avoid garbled output
        #pragma omp critical
        {
            printf("Hello World from host %s, MPI rank %d/%d, OpenMP thread %d/%d, CPU ID %d\n", 
                   hostname, rank, size, thread_id, num_threads, cpu_id);
        }
    }
    
    // Synchronize all processes before finalizing
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    return 0;
}
