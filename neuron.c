#include <stdio.h>
#include <stdlib.h>
#include <math.h>


typedef struct unit {
    double value;
    double grad;
    /* Pointer to pointers to the previous units */
    struct unit** prevs;
} unit;

unit* new_unit(double value, double grad) {
    unit* u = malloc(sizeof(unit));
    
    u->value = value;
    u->grad  = grad;
    /* For now, hardcode the size as we will be
     * working with 2 inputs as a max           */
    u->prevs = malloc(2 * sizeof(unit*));

    return u;
}

void free_unit(unit* u) {
    if (u->prevs[0])
        free_unit(u->prevs[0]);
    if (u->prevs[1])
        free_unit(u->prevs[1]);
    free(u);
}

void print_unit(unit* u) {
    printf("value= %.4f, grad= %.4f\n", u->value, u->grad);
}

unit* forward_multiply_gate(unit* u0, unit* u1) {
    unit* utop = new_unit(u0->value * u1->value, 0.0); 
    utop->prevs[0] = u0;
    utop->prevs[1] = u1;
    return utop;
}

void backward_multiply_gate(unit* utop) {
    utop->prevs[0]->grad += utop->prevs[1]->value * utop->grad;
    utop->prevs[1]->grad += utop->prevs[0]->value * utop->grad;
}

unit* forward_add_gate(unit* u0, unit* u1) {
    unit* utop = new_unit(u0->value + u1->value, 0.0);
    utop->prevs[0] = u0;
    utop->prevs[1] = u1;
    return utop;
}

void backward_add_gate(unit* utop) {
    utop->prevs[0]->grad += 1.0 * utop->grad;
    utop->prevs[1]->grad += 1.0 * utop->grad;
}

double sig(double x) {
    return 1.0 / (1.0 + exp(-x));
}

unit* forward_sigmoid(unit* u0) {
    unit* utop = new_unit(sig(u0->value), 0.0);
    utop->prevs[0] = u0;
    return utop;
}

void backward_sigmoid(unit* utop) {
    double s = sig(utop->prevs[0]->value);
    utop->prevs[0]->grad += (s * (1-s)) * utop->grad;
}

double forward_neuron(unit* x, unit* y, unit* a, unit* b, unit* c) {
    return 1.0 / (1.0 + exp(-(a->value*x->value + b->value*y->value + c->value)));
}

int main(void) {

    unit* x = new_unit(-1.0, 0.0);
    unit* y = new_unit(3.0, 0.0);
    unit* a = new_unit(1.0, 0.0);
    unit* b = new_unit(2.0, 0.0);
    unit* c = new_unit(-3.0, 0.0);
    
    double step_size = 0.01;
   
    /* forward pass */
    unit* ax = forward_multiply_gate(a, x);
    unit* by = forward_multiply_gate(b, y);
    unit* axpby = forward_add_gate(ax, by);
    unit* axpbypc = forward_add_gate(axpby, c);
    unit* s = forward_sigmoid(axpbypc);

    printf("out: "); print_unit(s);

    /* backward pass */
    s->grad = 1.0;
    backward_sigmoid(s);
    backward_add_gate(axpbypc);
    backward_add_gate(axpby);
    backward_multiply_gate(by);
    backward_multiply_gate(ax);

    double h = 0.0001;
    unit* aph = new_unit(a->value + h, 0.0);
    double a_grad = (forward_neuron(x, y, aph, b, c) - forward_neuron(x, y, a, b, c)) / h;
    printf("Numerical gradient of a: %.2f\n", a_grad);
    printf("Backprop gradient of a: %.2f\n", a->grad); 


    free_unit(s);

    return 0;
}
