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

typedef neural_network_t neural_network_grad_t;

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

void zero_grad(neural_network_grad_t *grad) {
    for (int j = 0; j < OUTPUT_SIZE; j++) {
        for (int i = 0; i < INPUT_SIZE; i++) {
            grad->W[i][j] = 0.;
        }
        grad->b[j] = 0.;
    }
}

void print_grad(neural_network_grad_t *grad) {
    printf("--- W ---\n");
    for (int i = 0; i < INPUT_SIZE; i++) {
        for (int j = 0; j < OUTPUT_SIZE; j++) {
            printf("%.2f ", grad->W[i][j]);
        }
        printf("\n");
    }
    printf("--- b ---\n");
    for (int j = 0; j < OUTPUT_SIZE; j++) {
        printf("%.2f ", grad->b[j]);
    }
    printf("\n");
}

float max(float y[OUTPUT_SIZE]) {
    float m = FLT_MIN;
    for (int i = 0; i < OUTPUT_SIZE; i++) {
        if (y[i] > m)
            m = y[i];
    }
    return m;
}

int get_i_max(float y[OUTPUT_SIZE]) {
    int i_max = 0;
    for (int i = 1; i < OUTPUT_SIZE; i++) {
        if (y[i] > y[i_max])
            i_max = i;
    }
    return i_max;
}

void forward(neural_network_t *nn, mnist_image_t *x, float y[OUTPUT_SIZE]) {
    for (int j = 0; j < OUTPUT_SIZE; j++) {
        y[j] = nn->b[j];
        for (int i = 0; i < INPUT_SIZE; i++) {
            y[j] += NORM(x->pixels[i]) * nn->W[i][j];
        }
    }
}

void backward(neural_network_grad_t *grad, mnist_image_t *x, float y[OUTPUT_SIZE], int i_ground_truth) {
    float m = max(y);
    /* Precompute the denominator */
    float d = 0.0;
    for (int i = 0; i < OUTPUT_SIZE; i++)
        d += exp(y[i] - m);

    for (int j = 0; j < OUTPUT_SIZE; j++) {
        grad->b[j] = exp(y[j] - m) / d;
        if (j == i_ground_truth)
            grad->b[j] -= 1.0;
        for (int i = 0; i < INPUT_SIZE; i++) {
            grad->W[i][j] = grad->b[j] * NORM(x->pixels[i]);
        }
    }
}

void print_y(float y[OUTPUT_SIZE]) {
    printf("[ ");
    for (int i = 0; i < OUTPUT_SIZE; i++)
        printf("%.2f, ", y[i]);
    printf(" ]\n");
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

float validate(neural_network_t *nn, mnist_dataset_t *test_dataset) {
    float y[OUTPUT_SIZE];

    float loss = 0.0;
    for (int i = 0; i < test_dataset->size; i++) {
        forward(nn, &test_dataset->images[i], y);
        loss += cross_entropy_loss(y, test_dataset->labels[i]);
    }
    loss /= test_dataset->size;
    
    return loss;
}

float validate_acc(neural_network_t *nn, mnist_dataset_t *test_dataset) {
    float y[OUTPUT_SIZE];

    float acc = 0.0;
    int i_max;
    for (int i = 0; i < test_dataset->size; i++) {
        forward(nn, &test_dataset->images[i], y);
        i_max = get_i_max(y);
        if (test_dataset->labels[i] == i_max)
            acc += 1.0;
    }
    acc /= test_dataset->size;
    
    return acc;
}

void step(neural_network_t *nn, neural_network_grad_t *grad, float step_size) {
    for (int j = 0; j < OUTPUT_SIZE; j++) {
        nn->b[j] -= step_size * grad->b[j];
        for (int i = 0; i < INPUT_SIZE; i++) {
            nn->W[i][j] -= step_size * grad->W[i][j];
        }
    }
}

int main(void) {

    mnist_dataset_t *train_dataset = build_train_dataset();
    mnist_dataset_t *test_dataset = build_test_dataset();
    neural_network_t *nn = initialize_network();
    neural_network_grad_t *grad = malloc(sizeof(neural_network_grad_t));
    /* Log training data */
    FILE *fp = fopen("logs/acc.txt", "w");
    if (!fp) {
        fprintf(stderr, "Couldn't open file!");
        fclose(fp);
        return 0;
    }

    /* Sanity check: the loss at initialization should be ~ -log(1/n_classes) */
    float loss = validate(nn, test_dataset);
    printf("Avg. loss on test set: %.4f\n", loss); 
    printf("-log(1/OUTPUT_SIZE): %.4f\n", -log(1.0 / OUTPUT_SIZE));

    /* Training loop */
    srand(time(0));
    int j;
    float acc = 0;
    float y[OUTPUT_SIZE];
    for (int i = 0; i < 10000; i++){
        j = rand() % train_dataset->size;
        forward(nn, &train_dataset->images[j], y); 
        zero_grad(grad);
        backward(grad, &train_dataset->images[j], y, train_dataset->labels[j]);
        step(nn, grad, 0.001);

        if (i % 100 == 0) {
            acc = validate_acc(nn, test_dataset);
            printf("Test accuracy on i=%d: %.4f\n", i, acc); 
            fprintf(fp, "%d\t%.4f\n", i, acc);
        }
    }

    free_dataset(train_dataset);
    free_dataset(test_dataset);
    free(nn);
    fclose(fp);
    return 0;
}

#endif
