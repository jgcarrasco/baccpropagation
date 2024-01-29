#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>

struct value;
typedef struct value value;

typedef void(*backwardfun)(value*);

struct value {
    double data;
    double grad;
    /* List of children of that value */
    int n_prev;
    struct value** prev;
    /* Backprop function */
    backwardfun _backward;
    /* Boolean used to traverse the graph */
    bool visited;
};

typedef struct {
    int count;
    value** values;
} topo;

value* new_value(double data) {
    value* v = malloc(sizeof(value));
    v->data = data;
    v->grad = 0.0;
    v->n_prev = 0;
    v->prev = NULL;
    v->_backward = NULL;
    v->visited = false;
    return v;
}

void free_value(value* v) {
    for (int i = 0; i < v->n_prev; i++) {
        free_value(v->prev[i]);
    }
    free(v->prev);
    free(v);
}

topo* new_topo() {
    topo* t = malloc(sizeof(topo));
    t->count = 0;
    t->values = NULL;
    return t;
}

void free_topo(topo* t) {
    free(t->values);
    free(t);
}

topo* topo_add(topo* t, value* v) {
    t->count++;
    t->values = realloc(t->values, t->count*sizeof(value*));
    t->values[t->count-1] = v;
    return t;
}

void _build_topo(topo*, value*);
topo* build_topo(value* v) {
    topo* t = new_topo();
    _build_topo(t, v);
    return t;
}

void _build_topo(topo* t, value* v) {
    if (!v->visited) {
        v->visited = true;
        topo_add(t, v);
        for (int i = 0; i < v->n_prev; i++) {
            _build_topo(t, v->prev[i]);    
        }
    }
}

void print_topo(topo* t) {
    for (int i = 0; i < t->count; i++) {
        printf("(data=%.2f, grad=%.2f) ", t->values[i]->data, t->values[i]->grad);
    }
    printf("\n");
}


void print_value(value* v, char* name) {
    printf("%s(data=%.4f, grad=%.4f)\n", name, v->data, v->grad);
}

void _value_add_backward(value* out) {
    out->prev[0]->grad += out->grad;
    out->prev[1]->grad += out->grad;
}

value* value_add(value* u, value* v) {
    value* out = new_value(u->data + v->data);
    
    out->n_prev = 2;
    out->prev = realloc(out->prev, 2*sizeof(value*));
    out->prev[0] = u; out->prev[1] = v;

    out->_backward = _value_add_backward;

    return out;
}

void _value_mul_backward(value* out) {
    out->prev[0]->grad += out->prev[1]->data * out->grad;
    out->prev[1]->grad += out->prev[0]->data * out->grad;
}

value* value_mul(value* u, value* v) {
    value* out = new_value(u->data * v->data);

    out->n_prev = 2;
    out->prev = realloc(out->prev, 2*sizeof(value*));
    out->prev[0] = u; out->prev[1] = v;

    out->_backward = _value_mul_backward;

    return out;
}

void backward(value* v) {
    v->grad = 1.0;
    topo* t = build_topo(v);
    for (int i = 0; i < t->count; i++) {
        if (t->values[i]->_backward) 
            t->values[i]->_backward(t->values[i]);
    }
    free_topo(t);
}

int main(void) {
    value* x = new_value(-2);
    value* y = new_value(5);
    value* z = new_value(-4);
    value* q = value_add(x, y);
    value* f = value_mul(q, z);

    backward(f);

    print_value(f, "f");
    print_value(q, "q");
    print_value(z, "z");
    print_value(x, "x");
    print_value(y, "y");


    free_value(f);
    return 0;
}
