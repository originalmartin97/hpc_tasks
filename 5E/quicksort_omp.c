#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <time.h>
#include <locale.h>

void print_array( int n, int * a )
{
    // [ 1, 2, 3, ... ]
    int i;
    if( n == 0 )
    {
        printf("[ ]\n");
        return;
    }
    printf("[");
    for( i = 0; i<n-1; i++ )
        printf(" %'d;",a[i]);
    printf(" %'d ]\n", a[n-1]);
}

void check_sorted( int n, int *a )
{
    // check if array is ascending: 1, 2, 2, 3, 3, 3,
    int i;
    int flag = 0;
    for( i = 0; i < n-1; i++ )
    {
        if( a[i] > a[i+1] )
        {
            printf("Bad order at index %d and %d: %d > %d.\n",
                   i, i+1, a[i], a[i+1] );
            flag = 1;
        }
    }
    if(flag)
        printf("Array is not sorted.\n");
    else
        printf("Array is sorted.\n");
}

void generate_array( int n, int * a)
{
    // suppose *a is already allocated!
    int i;
    srand(time(NULL));
    for( i = 0; i < n; i++ )
        a[i] = rand()%1000;
}

void quicksort(int *arr, int left, int right) {
    if (left < right) {
        int pivot = arr[right];
        int i = left - 1, j;

        for ( j = left; j < right; j++) {
            if (arr[j] < pivot) {
                i++;
                int temp = arr[i];
                arr[i] = arr[j];
                arr[j] = temp;
            }
        }

        int temp = arr[i + 1];
        arr[i + 1] = arr[right];
        arr[right] = temp;

        int partition_index = i + 1;

        #pragma omp task shared(arr) if (right - left > 1000)
        {quicksort(arr, left, partition_index - 1);}

        #pragma omp task shared(arr) if (right - left > 1000)
        {quicksort(arr, partition_index + 1, right);}

        #pragma omp taskwait
    }
}

int main(int argc, char **argv) {
    setlocale(LC_ALL, "");
    if (argc != 4) {
        printf("Usage: %s <array_size> <num_threads> <verbosity>.\n", argv[0]);
        return -1;
    }

    int N = atoi(argv[1]);
    if (N < 2) {
        printf("Invalid array length.\n");
        return -1;
    }

    int P = atoi(argv[2]);
    omp_set_num_threads(P);

    int *array = (int *)malloc(N * sizeof(int));
    generate_array(N, array);
    if( argv[3][0] == 'v') {
        print_array(N, array);
    }
    double start = omp_get_wtime();

    #pragma omp parallel
    {
        #pragma omp single
        quicksort(array, 0, N - 1);
    }

    double end = omp_get_wtime();

    check_sorted(N, array);
    if( argv[3][0] == 'v') {
        print_array(N, array);
    }

    printf("Elapsed time: %8.6lf\n", end - start);

    free(array);
    return 0;
}
