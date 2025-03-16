#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

typedef double myf;
typedef int myi;

double get_time(){ return omp_get_wtime(); }

myf fun(myf x) {return 4.0/(1.0+x*x);}

myf rectangle_method(myf a, myf b, myi n) {
    myf deltax = (b-a)/n;
    myf result = 0.0;
    
    #pragma omp parallel
    {
        myf local_sum = 0.0;
        
        #pragma omp single
        {
            // Determine chunk size for tasks
            myi chunk_size = 100;  // Adjust based on problem size
            myi num_chunks = (n + chunk_size - 1) / chunk_size;
            
            for(myi chunk = 0; chunk < num_chunks; chunk++) {
                myi start = chunk * chunk_size;
                myi end = (start + chunk_size < n) ? start + chunk_size : n;
                
                #pragma omp task firstprivate(start, end, deltax)
                {
                    myf task_sum = 0.0;
                    for(myi i = start; i < end; i++) {
                        task_sum += deltax * fun(a + deltax*0.5 + i*deltax);
                    }
                    
                    #pragma omp critical
                    {
                        result += task_sum;
                    }
                }
            }
        } // end single - implicit barrier waits for all tasks
    } // end parallel
    
    return result;
}

void usage(char * argv0) {
    printf("Usage:\n");
    printf("     %s <num_of_intervals>\n\n", argv0);
    printf("Integrates 4/(1+x^2) from 0 to 1 with rectangle method with given number of subintervals");
}

int main(int argc, char**argv) {
    if(argc != 2) {
        usage(argv[0]);
        return -1;
    }
    
    myi nint = (myi) atoll(argv[1]);
    double start = get_time();
    myf result = rectangle_method(0, 1, nint);
    double end = get_time();
    
    printf("Result: %.14f\n", result);
    printf("Error: %.14f\n", fabs(result - M_PI));
    printf("Elapsed time: %10.6f\n", (end - start));
    return 0;
}
