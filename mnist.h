#ifndef MNIST_H
#define MNIST_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define TRAIN_IMAGES "./data/train-images-idx3-ubyte"
#define TRAIN_LABELS "./data/train-labels-idx1-ubyte"
#define TEST_IMAGES "./data/t10k-images-idx3-ubyte"
#define TEST_LABELS "./data/t10k-labels-idx1-ubyte"

#define MAGIC_NUMBER_IMAGES 2051
#define MAGIC_NUMBER_LABELS 2049
#define NUMBER_IMAGES_TRAIN 60000
#define NUMBER_IMAGES_TEST 10000
#define MNIST_IMAGE_WIDTH 28
#define MNIST_IMAGE_HEIGHT 28
#define MNIST_IMAGE_SIZE MNIST_IMAGE_WIDTH * MNIST_IMAGE_HEIGHT

typedef struct {
    uint8_t pixels[MNIST_IMAGE_SIZE];
} mnist_image_t;

typedef struct {
    mnist_image_t *images;
    uint8_t *labels;
    uint32_t size;
} mnist_dataset_t;

int reverse_int(int i) {
    /* Change the endianness of a 32-bit integer */
    int converted = 0;
    converted |= ((0xff & i) << 24);
    converted |= (((0xff << 8) & i) << 8);
    converted |= (((0xff << 16) & i) >> 8);
    converted |= (((0xff << 24) & i) >> 24);
    return converted;
}

void print_digit(mnist_image_t image) {
    for (int i = 0; i < MNIST_IMAGE_HEIGHT; i++) {
        for (int j = 0; j < MNIST_IMAGE_WIDTH; j++) {
            printf("%3d ", image.pixels[MNIST_IMAGE_HEIGHT * i + j]);
        }
        printf("\n");
    }
}

mnist_image_t *read_images(char* image_path) {
    
    int magic_number, n_images, n_rows, n_cols;

    printf("Reading file %s...\n", image_path);

    FILE *fp = fopen(image_path, "rb");
    if (!fp) {
        fprintf(stderr, "Couldn't open file!");
        fclose(fp);
        return NULL;
    }

    fread(&magic_number, sizeof(magic_number), 1, fp);
    fread(&n_images, sizeof(n_images), 1, fp);
    fread(&n_rows, sizeof(n_rows), 1, fp);
    fread(&n_cols, sizeof(n_cols), 1, fp);

    magic_number = reverse_int(magic_number);
    n_images = reverse_int(n_images);
    n_rows = reverse_int(n_rows);
    n_cols = reverse_int(n_cols);

    if (magic_number != MAGIC_NUMBER_IMAGES) {
        fprintf(stderr, "The magic number doesn't match. Are you using the correct file?");
        return NULL;
    }

    printf("Number of images: %d\n", n_images);
    printf("Rows: %d\n", n_rows);
    printf("Cols: %d\n", n_cols);

    mnist_image_t *images = malloc(n_images * sizeof(mnist_image_t));
    fread(images, sizeof(mnist_image_t), n_images, fp);

    fclose(fp);

    return images;
}

uint8_t *read_labels(char* labels_path) {
    int magic_number, n_labels;

    printf("Opening file %s...\n", labels_path);
    FILE *fp = fopen(labels_path, "rb");
    if (!fp) {
        fprintf(stderr, "Couldn't open file!");
        fclose(fp);
        return NULL;
    }

    fread(&magic_number, sizeof(magic_number), 1, fp);
    fread(&n_labels, sizeof(n_labels), 1, fp);

    magic_number = reverse_int(magic_number);
    n_labels = reverse_int(n_labels);

    if (magic_number != MAGIC_NUMBER_LABELS) {
        fprintf(stderr, "The magic number doesn't match. Are you using the correct file?");
        return NULL;
    }

    printf("Number of labels: %d\n", n_labels);

    uint8_t *labels = malloc(n_labels * sizeof(uint8_t));
    fread(labels, sizeof(uint8_t), n_labels, fp);
    
    fclose(fp);
    return labels;
}

mnist_dataset_t *build_train_dataset(void) {
    mnist_dataset_t *train_dataset = malloc(sizeof(mnist_dataset_t));

    mnist_image_t *images = read_images(TRAIN_IMAGES);
    uint8_t *labels = read_labels(TRAIN_LABELS);
    
    train_dataset->images = images;
    train_dataset->labels = labels;

    return train_dataset;
}

mnist_dataset_t *build_test_dataset(void) {
    mnist_dataset_t *test_dataset = malloc(sizeof(test_dataset));

    mnist_image_t *images = read_images(TEST_IMAGES);
    uint8_t *labels = read_labels(TEST_LABELS);
    
    test_dataset->size = NUMBER_IMAGES_TEST;
    test_dataset->images = images;
    test_dataset->labels = labels;

    return test_dataset;
}

void free_dataset(mnist_dataset_t *dataset) {
    free(dataset->labels);
    free(dataset->images);
    free(dataset);
}

#endif
