#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <stdint.h>

#define TRAIN_IMAGES "./data/train-images-idx3-ubyte"
#define TRAIN_LABELS "./data/train-labels-idx1-ubyte"

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

int main(void) {
    int magic_number, n_images, n_rows, n_cols;
    /* fd -> file descriptor */
    int fd = open(TRAIN_IMAGES, O_RDONLY); 
    if (fd == -1) {
        fprintf(stderr, "Couldn't open file!");
        return -1;
    }

    read(fd, (char *)&magic_number, sizeof(magic_number));
    magic_number = reverse_int(magic_number);
    if (magic_number != 2051) {
        fprintf(stderr, "The magic number doesn't match. Are you using the correct file?");
    }

    read(fd, (char *)&n_images, sizeof(n_images));
    n_images = reverse_int(n_images);

    read(fd, (char *)&n_rows, sizeof(n_rows));
    n_rows = reverse_int(n_rows);

    read(fd, (char *)&n_cols, sizeof(n_cols));
    n_cols = reverse_int(n_cols);

    printf("Magic number: %d\n", magic_number);
    printf("Number of images: %d\n", n_images);
    printf("Rows: %d\n", n_rows);
    printf("Cols: %d\n", n_cols);

    mnist_image_t *images = malloc(n_images * sizeof(mnist_image_t));

    close(fd);

    return 0;
}

