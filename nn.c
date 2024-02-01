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

neural_network_t *initialize_network(void) {
    neural_network_t *nn = malloc(sizeof(neural_network_t));
    for (int i = 0; i < INPUT_SIZE; i++){
        for (int j = 0; j < OUTPUT_SIZE; j++){
            nn->W[i][j] = RAND();
        }
    }
    for (int i = 0; i < OUTPUT_SIZE; i++){
        nn->b[i] = RAND();
    }
    return nn;
}

void forward(neural_network_t *nn, mnist_image_t* x, float y[OUTPUT_SIZE]) {
    for (int j = 0; j < OUTPUT_SIZE; j++) {
        y[j] = nn->b[j];
        for (int i = 0; i < INPUT_SIZE; i++) {
            y[j] += x->pixels[i] * nn->W[i][j];
        }
    }
}

void print_y(float y[OUTPUT_SIZE]) {
    printf("[ ");
    for (int i = 0; i < OUTPUT_SIZE; i++)
        printf("%.2f, ", y[i]);
    printf(" ]\n");
}

int main(void) {

    mnist_dataset_t *train_dataset = build_train_dataset();
    neural_network_t *nn = initialize_network();
    
    srand(time(0));
    int i = rand() % NUMBER_IMAGES_TRAIN;

    float y[OUTPUT_SIZE];
    print_y(y);
    forward(nn, &train_dataset->images[i], y);
    print_y(y);
    free_dataset(train_dataset);
    free(nn);
    return 0;
}

#endif
