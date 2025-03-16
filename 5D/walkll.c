#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>

struct Node {
    double value;
    struct Node* next;
};

double generateRand() {
    return rand() / (double)RAND_MAX;
}

struct Node* buildList(int N) {
    struct Node* head = NULL;
    struct Node* current = NULL;

    for (int i = 0; i < N; i++) {
        struct Node* new_node = malloc(sizeof(struct Node));
        new_node->value = generateRand();
        new_node->next = NULL;

        if (head == NULL) {
            head = new_node;
            current = head;
        } else {
            current->next = new_node;
            current = new_node;
        }
    }

    return head;
}

double sum(struct Node* head) {
    double sum = 0.0;

    #pragma omp parallel
    {
        #pragma omp single
        {
            struct Node* current = head;
            while (current != NULL) {
                #pragma omp task firstprivate(current)
                {
                    double sqrt_val = sqrt(current->value);
                    #pragma omp critical
                    sum += sqrt_val;
                }
                current = current->next;
            }
        }
    }

    return sum;
}

void freeList(struct Node* head) {
    struct Node* current = head;
    while (current != NULL) {
        struct Node* temp = current;
        current = current->next;
        free(temp);
    }
}

void printList(struct Node* head) {
    struct Node* current = head;
    printf("Linked List: ");
    while (current != NULL) {
        printf("%.8f -> ", current->value);
        current = current->next;
    }
    printf("end\n");
}

int main(int argc, char *argv[]) {
    int N = atoi(argv[1]);
    srand(time(NULL));
    struct Node* head = buildList(N);
    printList(head);
    double result = sum(head);
    printf("Res: %.8f\n", result);
    freeList(head);
    return 0;
}
