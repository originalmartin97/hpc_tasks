#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

int main() {
    const char* words[] = {
        "The", "cat", "sat", "on", "the", "mat,", "watching", "the", "birds", "outside."
    };
    const int num_words = sizeof(words) / sizeof(words[0]);

    // inits the parallel region
    #pragma omp parallel
    {
        // utilize one available thread at a time
        #pragma omp single
        {
            for (int i = 0; i < num_words; ++i) {
                // identify the block of code to be executed explicitly in the prallel region
                #pragma omp task
                {
                    // tells omp to execute only one thread at a time
                    #pragma omp critical
                    {
                        printf("%s ", words[i]);
                    }
                }
            }
        }
        #pragma omp taskwait // Wait for all tasks to complete
    }

    // Optional: Reassemble the sentence (not required, but useful)
    printf("\nReassembled (original order): ");
    for (int i = 0; i < num_words; i++){
        printf("%s ", words[i]);
    }
    printf("\n");

    return 0;
}
