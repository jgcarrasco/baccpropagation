#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "mnist.h"

int main(void) {

    mnist_dataset_t *train_dataset = build_train_dataset();
    
    srand(time(0));
    int i = rand() % NUMBER_IMAGES_TRAIN;

    printf("%d\n", train_dataset->labels[i]); 
    print_digit(train_dataset->images[i]);
   
    free_dataset(train_dataset);
    return 0;
}
