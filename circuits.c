#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include <time.h>


double random() {
    return (double)rand() / (double)RAND_MAX;
}

double forwardMultiplyGate(double x, double y) {
    return x * y;
}

int main(void) {
    /* Initial values */
    double x = -2.;
    double y =  3.;

    /* Try changing x,y randomly small amounts and keep track of what works best */
    double tweak_amount = 0.01;
    double best_out = -DBL_MAX;
    double best_x = x;
    double best_y = y;

    for (int k = 0; k < 100; k++) {
        double x_try = x + tweak_amount * (random()*2 - 1);
        double y_try = y + tweak_amount * (random()*2 - 1);
        double out = forwardMultiplyGate(x_try, y_try);

        if (out > best_out) {
            best_out = out;
            best_x = x_try;
            best_y = y_try;
        }
    }


    printf("x: %f, y: %f, best_out: %f", best_x, best_y, best_out);

    return 0;
}

