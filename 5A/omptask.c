#include <stdio.h>
#include <omp.h>

int main( int argc, char **argv ){
        #pragma omp single {
                #pragma omp task {
                        printf("The ");
                }

                #pragma omp task {
                        printf("The ");
                }

                #pragma omp task {
                        printf("king ");
                }

                #pragma omp task {
                        printf("lion ");
                }

                #pragma omp task {
                        printf("sentence ");
                }

                #pragma omp task {
                        printf("not ");
                }

                #pragma omp task {
                        printf("words ");
                }

                #pragma omp task {
                        printf("chase ");
                }

                #pragma omp task {
                        printf("black ");
                }

                #pragma omp task {
                        printf("cat. ");
                }

        }
        return 0;
}
