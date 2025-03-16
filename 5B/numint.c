#define __USE_GNU
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>
#include <omp.h>

// Use M_PIl for pi
#define M_PIl 3.141592653589793238462643383279502884L

typedef double myf;
typedef int myi;

double get_time() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (double)tv.tv_usec / 1000000.0 + (double)tv.tv_sec;
}

myf fun(myf x) {
    return 4.0 / (1.0 + x * x);
}

myf rectangle_method(myf a, myf b, myi n) {
    myf deltax = (b - a) / n;
    myf res = 0.0;
    myi i;

    for (i = 0; i < n; i++) {
        res += deltax * fun(a + deltax * 0.5 + i * deltax);
    }
    return res;
}

myf simpson_method(myf a, myf b, myi n) {
    if (n % 2 != 0) {
        n++;
    }
    myf h = (b - a) / n;
    myf sum = fun(a) + fun(b);
    myi i;

    for (i = 1; i < n; i++) {
        myf x = a + i * h;
        if (i % 2 == 0) {
            sum += 2 * fun(x);
        } else {
            sum += 4 * fun(x);
        }
    }

    myf res = (h / 3.0) * sum;
    return res;
}

myf trapezoid_method(myf a, myf b, myi n) {
    myf h = (b - a) / n;
    myf sum = fun(a) + fun(b);
    myi i;

    for (i = 1; i < n; i++) {
        sum += 2 * fun(a + i * h);
    }

    return (h / 2.0) * sum;
}

void write_results(const char *filename, myi nint, myf result, double elapsed_time) {
    FILE *fp = fopen(filename, "a");
    if (fp == NULL) {
        perror("Error opening file");
        return;
    }

    fprintf(fp, "%d,", nint);
    fprintf(fp, "%16.14f,", (double)result);
    fprintf(fp, "%16.14f,", (double)fabs(result - (myf)M_PIl));
    fprintf(fp, "%10.6f\n", elapsed_time);

    fclose(fp);
}

void run_rectangle_method(myi nint) {
    double start = get_time();
    myf result = rectangle_method(0, 1, nint);
    double end = get_time();

    write_results("results_rectangle.txt", nint, result, end - start);

    printf("Rectangle Method\n");
    printf("Result: %16.14f\n", (double)result);
    printf("Error: %16.14f\n", (double)fabs(result - (myf)M_PIl));
    printf("Elapsed time: %10.6f\n\n", (end - start));
}

void run_simpson_method(myi nint) {
    double start = get_time();
    myf result = simpson_method(0, 1, nint);
    double end = get_time();

    write_results("results_simpson.txt", nint, result, end - start);

    printf("Simpson Method\n");
    printf("Result: %16.14f\n", (double)result);
    printf("Error: %16.14f\n", (double)fabs(result - (myf)M_PIl));
    printf("Elapsed time: %10.6f\n\n", (end - start));
}

void run_trapezoid_method(myi nint) {
    double start = get_time();
    myf result = trapezoid_method(0, 1, nint);
    double end = get_time();

    write_results("results_trapezoid.txt", nint, result, end - start);

    printf("Trapezoid Method\n");
    printf("Result: %16.14f\n", (double)result);
    printf("Error: %16.14f\n", (double)fabs(result - (myf)M_PIl));
    printf("Elapsed time: %10.6f\n\n", (end - start));
}

void run_methods(myi nint) {
    // 1. Function Pointer Array:
    //    We create an array of function pointers called 'methods'.
    //    This array stores the addresses of our integration method functions.
    //    This lets us treat the functions like data, making it easier to loop through them.
    void (*methods[])(myi) = {run_rectangle_method, run_simpson_method, run_trapezoid_method};

    // 2. Calculate Number of Methods:
    //    We determine the number of methods in our array.
    //    This makes the code more flexible if we add or remove methods later.
    int num_methods = sizeof(methods) / sizeof(methods[0]);

    // 3. Start a Parallel Region:
    //    '#pragma omp parallel' creates a team of threads to execute the code inside.
    //    This means the code within this block can run concurrently on multiple cores.
    #pragma omp parallel
    {
        // 4. Single Thread Task Creation:
        //    '#pragma omp single' ensures that only one thread in the team executes
        //    the code inside. This is important because we only want to create
        //    the tasks once.
        #pragma omp single
        {
            // 5. Loop Through Methods and Create Tasks:
            //    We loop through the 'methods' array, creating a task for each method.
            //    '#pragma omp task' creates a new task that can be executed by any available thread.
            //    'methods[i](nint)' calls the function pointed to by 'methods[i]', passing 'nint'.
            for (int i = 0; i < num_methods; ++i) {
                #pragma omp task
                methods[i](nint);
            }
        }
        // 6. Wait for All Tasks to Complete:
        //    '#pragma omp taskwait' creates a barrier. The threads in the team will wait
        //    here until all the tasks created within the 'single' region have finished.
        //    This ensures that all integration methods complete before the program continues.
        #pragma omp taskwait
    }
}

void usage(char *argv0) {
    printf("Usage:\n");
    printf("      %s <num_of_intervals>\n\n");
    printf("Integrates 4/(1+x^2) from 0 to 1 with all methods.\n");
}

int main(int argc, char **argv) {
    if (argc != 2) {
        usage(argv[0]);
        return -1;
    }

    myi nint = (myi)atoll(argv[1]);

    run_methods(nint);

    return 0;
}
