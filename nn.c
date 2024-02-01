#ifndef NN_H
#define NN_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "mnist.h"

#define INPUT_SIZE 784
#define OUTPUT_SIZE 10

#define RAND() ((float) rand() / (float) RAND_MAX)

typedef struct {
    float W[INPUT_SIZE][OUTPUT_SIZE];
    float b[OUTPUT_SIZE];
} neural_network_t;

neural_network_t *initialize_network() {
    neural_network_t *nn = malloc(sizeof(neural_network_t));
    for (int i = 0; i < INPUT_SIZE; i++){
        for (int j = 0; j < OUTPUT_SIZE; j++){
            
        }
    }
}

int main(void) {

    mnist_dataset_t *train_dataset = build_train_dataset();
    
    srand(time(0));
    int i = rand() % NUMBER_IMAGES_TRAIN;

    printf("%d\n", train_dataset->labels[i]); 
    print_digit(train_dataset->images[i]);
   
    free_dataset(train_dataset);
    return 0;
}

#endif
