#include "basis/stage4_learning/optim.h"
#include <math.h>

Adam* adam_new(size_t rows, size_t cols, double lr) {
    Adam* opt = (Adam*)malloc(sizeof(Adam));
    opt->lr = lr;
    opt->beta1 = 0.9;
    opt->beta2 = 0.999;
    opt->eps = 1e-8;
    opt->t = 0;
    opt->m = tensor_new(rows, cols);
    opt->v = tensor_new(rows, cols);
    tensor_fill(opt->m, 0.0);
    tensor_fill(opt->v, 0.0);
    return opt;
}

void adam_step(Adam* opt, Tensor* W) {
    opt->t++;
    for (size_t i = 0; i < W->rows * W->cols; i++) {
        double g = W->data[i]->grad;
        opt->m->data[i]->data = opt->beta1 * opt->m->data[i]->data + (1.0 - opt->beta1) * g;
        opt->v->data[i]->data = opt->beta2 * opt->v->data[i]->data + (1.0 - opt->beta2) * g * g;
        double m_hat = opt->m->data[i]->data / (1.0 - pow(opt->beta1, opt->t));
        double v_hat = opt->v->data[i]->data / (1.0 - pow(opt->beta2, opt->t));
        W->data[i]->data -= opt->lr * m_hat / (sqrt(v_hat) + opt->eps);
        W->data[i]->grad = 0.0;
    }
}

void adam_free(Adam* opt) {
    tensor_free(opt->m);
    tensor_free(opt->v);
    free(opt);
}
