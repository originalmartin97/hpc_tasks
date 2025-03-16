/*
 * The first task depends on the initial value of dependency 0.
 * After first completes it updates the value of dependency to 1.
 * The second task depends on dependency with value 1 and so on.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

int main() {
    const char* words[] = {
        "The", "cat", "sat", "on", "the", "mat,", "watching", "the", "birds", "outside."
    };
    const int num_words = sizeof(words) / sizeof(words[0]);

    #pragma omp parallel
    {
        #pragma omp single
        {
            // Dummy variable to create dependencies
            int dependency = 0;

            for (int i = 0; i < num_words; ++i) {
                #pragma omp task depend(inout: dependency)
                // inout means that the task reads and writes the dependency variable
                // creates a dependency chain: each task depends on previous task's modification
                {
                    #pragma omp critical // ensures only one thread prints at a time
                    {
                        printf("%s ", words[i]);
                    }
                    dependency = i + 1; // Update dependency for the next task
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
