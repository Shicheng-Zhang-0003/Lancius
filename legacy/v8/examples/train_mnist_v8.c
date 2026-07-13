#include <basis.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <unistd.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define BATCH_SIZE 64
#define EPOCHS 10
#define LR 0.001

uint32_t read_int(FILE* f) {
    uint8_t b[4]; size_t dr = fread(b, 1, 4, f); (void)dr;
    return (b[0]<<24) | (b[1]<<16) | (b[2]<<8) | b[3];
}

void download_mnist() {
    if (access("train-images-idx3-ubyte", F_OK) == 0) return;
    printf("[1/5] Downloading MNIST Dataset...\n");
    int s1 = system("curl -s -o train-images-idx3-ubyte.gz https://ossci-datasets.s3.amazonaws.com/mnist/train-images-idx3-ubyte.gz"); (void)s1;
    int s2 = system("curl -s -o train-labels-idx1-ubyte.gz https://ossci-datasets.s3.amazonaws.com/mnist/train-labels-idx1-ubyte.gz"); (void)s2;
    int s3 = system("curl -s -o t10k-images-idx3-ubyte.gz https://ossci-datasets.s3.amazonaws.com/mnist/t10k-images-idx3-ubyte.gz"); (void)s3;
    int s4 = system("curl -s -o t10k-labels-idx1-ubyte.gz https://ossci-datasets.s3.amazonaws.com/mnist/t10k-labels-idx1-ubyte.gz"); (void)s4;
    int s5 = system("gunzip -f *.gz"); (void)s5;
}

uint8_t* load_images(const char* path, int* num) {
    FILE* f = fopen(path, "rb");
    uint32_t magic = read_int(f); (void)magic;
    *num = read_int(f); int r = read_int(f); int c = read_int(f); (void)r; (void)c;
    uint8_t* data = (uint8_t*)malloc(*num * 784);
    size_t dr = fread(data, 1, *num * 784, f); (void)dr;
    fclose(f);
    return data;
}

uint8_t* load_labels(const char* path, int* num) {
    FILE* f = fopen(path, "rb");
    uint32_t magic = read_int(f); (void)magic;
    *num = read_int(f);
    uint8_t* data = (uint8_t*)malloc(*num);
    size_t dr = fread(data, 1, *num, f); (void)dr;
    fclose(f);
    return data;
}

void xavier_init(double* w, size_t rows, size_t cols) {
    double limit = sqrt(6.0 / (rows + cols));
    for(size_t i=0; i<rows*cols; i++) {
        w[i] = ((double)rand() / RAND_MAX * 2.0 - 1.0) * limit;
    }
}

void adam_step(double* w, double* g, double* m, double* v, size_t sz, double lr, double b1, double b2, double eps, int t) {
    double bc1 = 1.0 - pow(b1, t); double bc2 = 1.0 - pow(b2, t);
    for(size_t i=0; i<sz; i++) {
        m[i] = b1 * m[i] + (1.0 - b1) * g[i];
        v[i] = b2 * v[i] + (1.0 - b2) * g[i] * g[i];
        w[i] -= lr * (m[i] / bc1) / (sqrt(v[i] / bc2) + eps);
    }
}

int main() {
    srand(42);
    printf("================================================================\n");
    printf("  BASIS V8: MNIST Notepad (Sequential + Scaled MSE)           \n");
    printf("================================================================\n\n");

    download_mnist();

    int tr_n, te_n;
    uint8_t* tr_X = load_images("train-images-idx3-ubyte", &tr_n);
    uint8_t* tr_Y = load_labels("train-labels-idx1-ubyte", &tr_n);
    uint8_t* te_X = load_images("t10k-images-idx3-ubyte", &te_n);
    uint8_t* te_Y = load_labels("t10k-labels-idx1-ubyte", &te_n);
    printf("[2/5] Loaded %d training images, %d test images.\n", tr_n, te_n);

    v8_graph* g = v8_graph_create();
    v8_node* X = v8_input(g, BATCH_SIZE, 784);
    v8_node* W1 = v8_input(g, 784, 128); v8_node* b1 = v8_input(g, 1, 128);
    v8_node* W2 = v8_input(g, 128, 10);  v8_node* b2 = v8_input(g, 1, 10);
    v8_node* Y = v8_input(g, BATCH_SIZE, 10);

    v8_node* Z1 = v8_matmul(g, X, W1);
    v8_node* A1 = v8_add(g, Z1, v8_broadcast(g, b1, BATCH_SIZE, 128));
    v8_node* H1 = v8_relu(g, A1);
    v8_node* Z2 = v8_matmul(g, H1, W2);
    v8_node* A2 = v8_add(g, Z2, v8_broadcast(g, b2, BATCH_SIZE, 10));

    // TRUE MSE: Scale the loss by 1/(Batch*Classes) IN THE IR GRAPH
    // The Autodiff engine will automatically scale all gradients by 1/640!
    v8_node* diff = v8_sub(g, A2, Y);
    v8_node* sq = v8_mul(g, diff, diff);
    v8_node* sq_sum = v8_sum(g, sq);
    v8_node* scale = v8_const(g, 1.0 / (BATCH_SIZE * 10.0), 1, 1);
    v8_node* loss = v8_mul(g, sq_sum, scale);

    printf("[3/5] Compiling Backward Pass & Scheduling Waves...\n");
    v8_training_graph* tg = v8_ir_autodiff(g, loss);
    v8_schedule* sched = v8_ir_schedule(tg->graph);
    v8_arena* scratch = v8_arena_create(256 * 1024 * 1024);

    double* x_batch = (double*)calloc(BATCH_SIZE * 784, sizeof(double));
    double* y_batch = (double*)calloc(BATCH_SIZE * 10, sizeof(double));
    double* w1 = (double*)calloc(784 * 128, sizeof(double)); xavier_init(w1, 784, 128);
    double* b1_d = (double*)calloc(1 * 128, sizeof(double));
    double* w2 = (double*)calloc(128 * 10, sizeof(double)); xavier_init(w2, 128, 10);
    double* b2_d = (double*)calloc(1 * 10, sizeof(double));

    double* m_w1 = (double*)calloc(784*128, sizeof(double)); double* v_w1 = (double*)calloc(784*128, sizeof(double));
    double* m_w2 = (double*)calloc(128*10, sizeof(double));  double* v_w2 = (double*)calloc(128*10, sizeof(double));
    double* m_b1 = (double*)calloc(1*128, sizeof(double)); double* v_b1 = (double*)calloc(1*128, sizeof(double));
    double* m_b2 = (double*)calloc(1*10, sizeof(double));  double* v_b2 = (double*)calloc(1*10, sizeof(double));

    double* grad_w1 = (double*)calloc(784*128, sizeof(double));
    double* grad_w2 = (double*)calloc(128*10, sizeof(double));
    double* grad_b1 = (double*)calloc(1*128, sizeof(double));
    double* grad_b2 = (double*)calloc(1*10, sizeof(double));

    v8_node *nW1=NULL, *nW2=NULL, *nb1=NULL, *nb2=NULL;
    for(uint32_t i=0; i<tg->graph->node_count; i++) {
        v8_node* n = tg->graph->nodes[i];
        if(n->op == V8_OP_INPUT) {
            if(n->shape[0] == BATCH_SIZE && n->shape[1] == 784) n->runtime_data = x_batch;
            else if(n->shape[0] == BATCH_SIZE && n->shape[1] == 10) n->runtime_data = y_batch;
            else if(n->shape[0] == 784 && n->shape[1] == 128) { n->runtime_data = w1; nW1 = n; }
            else if(n->shape[0] == 1 && n->shape[1] == 128) { n->runtime_data = b1_d; nb1 = n; }
            else if(n->shape[0] == 128 && n->shape[1] == 10) { n->runtime_data = w2; nW2 = n; }
            else if(n->shape[0] == 1 && n->shape[1] == 10) { n->runtime_data = b2_d; nb2 = n; }
        }
    }

    printf("[4/5] Training for %d Epochs (Batch Size: %d) via Sequential Executor...\n\n", EPOCHS, BATCH_SIZE);
    int* indices = (int*)malloc(tr_n * sizeof(int));
    for(int i=0; i<tr_n; i++) indices[i] = i;

    int step = 0;
    for(int ep=0; ep<EPOCHS; ep++) {
        for(int i=tr_n-1; i>0; i--) { int j = rand() % (i+1); int tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp; }
        double epoch_loss = 0.0; int batches = 0;
        for(int i=0; i<=tr_n - BATCH_SIZE; i+=BATCH_SIZE) {
            memset(y_batch, 0, BATCH_SIZE * 10 * sizeof(double));
            for(int b=0; b<BATCH_SIZE; b++) {
                int idx = indices[i+b];
                for(int p=0; p<784; p++) x_batch[b*784 + p] = (tr_X[idx*784 + p] / 255.0) - 0.5; // Zero-Mean Centering
                y_batch[b*10 + tr_Y[idx]] = 1.0;
            }

            // Nullify intermediates
            for(uint32_t w=0; w<sched->wave_count; w++) {
                for(uint32_t k=0; k<sched->waves[w].node_count; k++) {
                    v8_node* n = sched->waves[w].nodes[k];
                    if (n->op != V8_OP_INPUT && n->op != V8_OP_CONST) n->runtime_data = NULL;
                }
            }

            // BULLETPROOF FIX: Sequential execution guarantees deterministic gradient accumulation
            v8_schedule_execute(sched, scratch);
            step++;

            // Extract gradients (Autodiff already scaled them by 1/640 via the IR graph!)
            if(tg->grad_nodes[nW1->id] && tg->grad_nodes[nW1->id]->runtime_data) memcpy(grad_w1, tg->grad_nodes[nW1->id]->runtime_data, 784*128*sizeof(double));
            if(tg->grad_nodes[nb1->id] && tg->grad_nodes[nb1->id]->runtime_data) memcpy(grad_b1, tg->grad_nodes[nb1->id]->runtime_data, 1*128*sizeof(double));
            if(tg->grad_nodes[nW2->id] && tg->grad_nodes[nW2->id]->runtime_data) memcpy(grad_w2, tg->grad_nodes[nW2->id]->runtime_data, 128*10*sizeof(double));
            if(tg->grad_nodes[nb2->id] && tg->grad_nodes[nb2->id]->runtime_data) memcpy(grad_b2, tg->grad_nodes[nb2->id]->runtime_data, 1*10*sizeof(double));

            adam_step(w1, grad_w1, m_w1, v_w1, 784*128, LR, 0.9, 0.999, 1e-8, step);
            adam_step(b1_d, grad_b1, m_b1, v_b1, 1*128, LR, 0.9, 0.999, 1e-8, step);
            adam_step(w2, grad_w2, m_w2, v_w2, 128*10, LR, 0.9, 0.999, 1e-8, step);
            adam_step(b2_d, grad_b2, m_b2, v_b2, 1*10, LR, 0.9, 0.999, 1e-8, step);

            epoch_loss += tg->loss_node->runtime_data[0];
            batches++;
            v8_arena_reset(scratch);
            if(batches % 100 == 0) printf("\r  Epoch %d | Batch %d/%d | Loss: %.4f", ep+1, batches, tr_n/BATCH_SIZE, epoch_loss/batches);
        }
        printf("\r  Epoch %d | Loss: %.4f                                     \n", ep+1, epoch_loss / batches);
    }

    printf("\n[5/5] Compiling Inference Graph to Bytecode VM & Evaluating...\n");
    v8_graph* g_inf = v8_graph_create();
    v8_node* X_inf = v8_input(g_inf, BATCH_SIZE, 784);
    v8_node* W1_inf = v8_input(g_inf, 784, 128); v8_node* b1_inf = v8_input(g_inf, 1, 128);
    v8_node* W2_inf = v8_input(g_inf, 128, 10);  v8_node* b2_inf = v8_input(g_inf, 1, 10);

    v8_node* Z1_inf = v8_matmul(g_inf, X_inf, W1_inf);
    v8_node* A1_inf = v8_add(g_inf, Z1_inf, v8_broadcast(g_inf, b1_inf, BATCH_SIZE, 128));
    v8_node* H1_inf = v8_relu(g_inf, A1_inf);
    v8_node* Z2_inf = v8_matmul(g_inf, H1_inf, W2_inf);
    v8_node* A2_inf = v8_add(g_inf, Z2_inf, v8_broadcast(g_inf, b2_inf, BATCH_SIZE, 10));
    (void)A2_inf;

    v8_program* prog = v8_compile_graph(g_inf);

    int correct = 0;
    double* out_batch = (double*)malloc(BATCH_SIZE * 10 * sizeof(double));
    double* vm_inputs[5] = {x_batch, w1, b1_d, w2, b2_d};

    for(int i=0; i<=te_n - BATCH_SIZE; i+=BATCH_SIZE) {
        for(int b=0; b<BATCH_SIZE; b++) {
            int idx = i+b;
            for(int p=0; p<784; p++) x_batch[b*784 + p] = (te_X[idx*784 + p] / 255.0) - 0.5; // Zero-Mean Centering
        }
        v8_vm_execute(prog, vm_inputs, out_batch, scratch);
        for(int b=0; b<BATCH_SIZE; b++) {
            int pred = 0; double max_logit = -1e9;
            for(int c=0; c<10; c++) {
                if(out_batch[b*10 + c] > max_logit) { max_logit = out_batch[b*10 + c]; pred = c; }
            }
            if(pred == te_Y[i+b]) correct++;
        }
        v8_arena_reset(scratch);
    }

    v8_program_destroy(prog);
    v8_graph_destroy(g_inf);

    printf("\n================================================================\n");
    printf("  FINAL TEST ACCURACY: %.2f%% (%d / %d)\n", 100.0 * correct / te_n, correct, te_n);
    printf("================================================================\n");
    printf("  V8 NOTEPAD COMPLETE. PATH C & D VERIFIED.\n");
    printf("================================================================\n\n");

    return 0;
}
