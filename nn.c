#ifndef NN_H
#define NN_H

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <float.h>

#include "mnist.h"

#define INPUT_SIZE 784
#define OUTPUT_SIZE 10

#define RAND() ((float) rand() / (float) RAND_MAX)
#define NORM(x) ((float) x / 255.0)

typedef struct {
    float W[INPUT_SIZE][OUTPUT_SIZE];
    float b[OUTPUT_SIZE];
} neural_network_t;

neural_network_t *initialize_network(void) {
    neural_network_t *nn = malloc(sizeof(neural_network_t));
    /* Initialize weight values uniformly from [-k,k] */
    float k = sqrt(1/INPUT_SIZE);
    for (int i = 0; i < INPUT_SIZE; i++){
        for (int j = 0; j < OUTPUT_SIZE; j++){
            nn->W[i][j] = (2.*k * RAND() - k);
        }
    }
    for (int i = 0; i < OUTPUT_SIZE; i++){
        nn->b[i] = 0.0;
    }
    return nn;
}

void forward(neural_network_t *nn, mnist_image_t* x, float y[OUTPUT_SIZE]) {
    for (int j = 0; j < OUTPUT_SIZE; j++) {
        y[j] = nn->b[j];
        for (int i = 0; i < INPUT_SIZE; i++) {
            y[j] += NORM(x->pixels[i]) * nn->W[i][j];
        }
    }
}

void print_y(float y[OUTPUT_SIZE]) {
    printf("[ ");
    for (int i = 0; i < OUTPUT_SIZE; i++)
        printf("%.2f, ", y[i]);
    printf(" ]\n");
}

float max(float y[OUTPUT_SIZE]) {
    float m = FLT_MIN;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        if (y[i] > m)
            m = y[i];
    }
    return m;
}

float cross_entropy_loss(float y[OUTPUT_SIZE], int i_ground_truth) {
    float m = max(y);
    float loss = 0.;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        loss += exp(y[i] - m);
    }
    loss = log(loss);
    loss -= y[i_ground_truth] - m;
    return loss;
}


int main(void) {

    mnist_dataset_t *train_dataset = build_train_dataset();
    mnist_dataset_t *test_dataset = build_test_dataset();
    neural_network_t *nn = initialize_network();
    
    srand(time(0));
    // int i = rand() % NUMBER_IMAGES_TEST;

    float y[OUTPUT_SIZE];

    float loss = 0.0;

    for (int j = 0; j < test_dataset->size; j++) {
        forward(nn, &test_dataset->images[j], y);
        loss += cross_entropy_loss(y, test_dataset->labels[j]);
    }
    loss /= test_dataset->size;
    printf("Avg. loss on test set: %.4f\n", loss); 
    printf("-log(1/OUTPUT_SIZE): %.4f\n", -log(1.0 / OUTPUT_SIZE));

    free_dataset(train_dataset);
    free_dataset(test_dataset);
    free(nn);
    return 0;
}

#endif
