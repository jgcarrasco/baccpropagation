#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

struct value;
typedef struct value value;

typedef void(*backwardfun)(value*);

enum {OP_LEAF, OP_ADD, OP_MUL};

struct value {
    double data;
    double grad;
    /* List of children of that value */
    int n_prev;
    struct value** prev;
    int op;
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
    v->op = OP_LEAF;
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

value* value_add(value* u, value* v) {
    value* out = new_value(u->data + v->data);
    
    out->n_prev = 2;
    out->prev = realloc(out->prev, 2*sizeof(value*));
    out->prev[0] = u; out->prev[1] = v;

    out->op = OP_ADD;

    return out;
}

value* value_mul(value* u, value* v) {
    value* out = new_value(u->data * v->data);

    out->n_prev = 2;
    out->prev = realloc(out->prev, 2*sizeof(value*));
    out->prev[0] = u; out->prev[1] = v;

    out->op = OP_MUL;

    return out;
}

void reset_visited(value* v) {
    v->visited = false;
    for (int i = 0; i < v->n_prev; i++)
        reset_visited(v->prev[i]);
}

void _backward(value* v) {
    switch (v->op) {
        case OP_LEAF: break;
        case OP_ADD:
            v->prev[0]->grad += v->grad;
            v->prev[1]->grad += v->grad;
            break;
        case OP_MUL:
            v->prev[0]->grad += v->prev[1]->data * v->grad;
            v->prev[1]->grad += v->prev[0]->data * v->grad;
            break;
    }
}

void backward(value* v) {
    topo* t = build_topo(v);
    for (int i = 0; i < t->count; i++) {
        _backward(t->values[i]);
    }
    reset_visited(v);
    free_topo(t);
}

void _forward(value* v) {
    switch (v->op) {
        case OP_LEAF: break;
        case OP_ADD:
            v->data = v->prev[0]->data + v->prev[1]->data;
            break;
        case OP_MUL:
            v->data = v->prev[0]->data * v->prev[1]->data;
            break;
    }
}

void forward(value* v) {
    topo* t = build_topo(v);
    for (int i = t->count - 1; i >= 0; i--) {
        _forward(t->values[i]);
    }
    reset_visited(v);
    free_topo(t);
}

void zero_grad(value* v) {
    v->grad = 0.0;
    for (int i = 0; i < v->n_prev; i++) {
        zero_grad(v->prev[i]);
    }
}


/* SVM EXAMPLE */
int main(void) {
    srand(time(0));
    double step_size = 0.01;
    /* Dataset */
    double data[6][2] = {
        { 1.2,  0.7},
        {-0.3, -0.5},
        { 3.0,  0.1},
        {-0.1, -1.0},
        {-1.0,  1.1},
        { 2.1, -3.0},
    };

    int labels[6] = {1, -1, 1, -1, -1, 1};

    /* Random initialization */
    value* a = new_value(1.0);
    value* b = new_value(-2.0);
    value* c = new_value(-1.0);
    value* x = new_value(1.2);
    value* y = new_value(0.7);

    value* out = value_add(value_add(value_mul(a, x), value_mul(b, y)), c);
    
    /* Training loop */ 
    for (int iter = 0; iter < 400; iter++) {
        int i = rand() % 6;
        x->data = data[i][0];
        y->data = data[i][1];
        int label = labels[i];
        
        forward(out);
        zero_grad(out);
        if (label == 1 && out->data < 1)
            out->grad = 1.0;
        if (label == -1 && out->data > -1)
            out->grad = -1.0;
        backward(out);
        /* Add regularization */
        a->grad += -a->data;
        b->grad += -b->data;
        /* Update the weights */
        a->data += step_size * a->grad;
        b->data += step_size * b->grad;
        c->data += step_size * c->grad;
        
        if (iter % 25 == 0) {
            int num_correct = 0;
            for (int j = 0; j < 6; j++) {
                x->data = data[j][0];
                y->data = data[j][1];
                int true_label = labels[j];
                forward(out);
                int pred_label = (out->data > 0.0) ? 1 : -1;
                if (pred_label == true_label) {
                    num_correct++;
                }
            }
            printf("training accuracy at iter %d: %.4f\n", iter, ((double) num_correct) / 6.0);
        }

    }

    free_value(out);
    return 0;
}
