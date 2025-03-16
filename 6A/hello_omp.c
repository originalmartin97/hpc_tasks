#define _GNU_SOURCE
#include <stdio.h>
#include <omp.h>
#include <sched.h>
#include <unistd.h>

int main(int argc, char ** argv)
{
    char h[100];
    gethostname( h, 100 );
    #pragma omp parallel
    {
        int i = omp_get_thread_num();
        int n = omp_get_num_threads();
        int c = sched_getcpu();
        printf("Hello World from thread %d of %d running on cpu # %d from %s\n", i, n, c, h);
    }
    return 0;
}
