#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <omp.h>
// use M_PIl for pi

typedef double myf;
typedef int myi;

double get_time(){ return omp_get_wtime(); } // Using OpenMP timer for better precision

myf fun(myf x) {return 4.0/(1.0+x*x);}

myf rectangle_method(myf a, myf b, myi n){
    myf deltax = (b-a)/n;
    myf res = 0.0;
    
    #pragma omp parallel for reduction(+:res)
    for(myi i=0; i<n; i++){
        // Fixed bug: i*n -> i*deltax
        res += deltax*fun(a + deltax*0.5 + i*deltax);
    }
    
    return res;
}

void usage(char * argv0){
    printf("Usage:\n");
    printf("     %s <num_of_intervals>\n\n", argv0);
    printf("Integrates 4/(1+x^2) from 0 to 1 with rectangle method with given number of subintervals");
}

int main(int argc, char**argv){
    if(argc != 2){
        usage(argv[0]);
        return -1;
    }
    
    myi nint = (myi) atoll(argv[1]);
    double start = get_time();
    myf result = rectangle_method(0, 1, nint);
    double end = get_time();
    
    printf("Result: %.14f\n", result);
    printf("Error: %.14f\n", fabs(result - M_PI));  // Using fabs instead of abs for floating point
    printf("Elapsed time: %10.6f\n", (end - start));
    
    return 0;
}
