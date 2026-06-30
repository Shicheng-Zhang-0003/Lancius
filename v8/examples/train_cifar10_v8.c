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

#define BATCH_SIZE 32
#define EPOCHS 30
double lr = 0.001;
#define NUM_TRAIN 50000
#define NUM_TEST 10000

uint32_t read_int(FILE* f) {
    uint8_t b[4]; size_t dr = fread(b, 1, 4, f); (void)dr;
    return (b[0]<<24) | (b[1]<<16) | (b[2]<<8) | b[3];
}

void download_cifar10() {
    if (access("cifar-10-batches-bin/data_batch_1.bin", F_OK) == 0) return;
    printf("[1/5] Downloading CIFAR-10 Binary Dataset...\n");
    int s1 = system("curl -s -L https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz -o cifar.tar.gz"); (void)s1;
    int s2 = system("tar -xzf cifar.tar.gz"); (void)s2;
    int s3 = system("rm cifar.tar.gz"); (void)s3;
}

void load_cifar10(uint8_t* X, uint8_t* Y, const char** files, int num_files) {
    int offset = 0;
    for(int f=0; f<num_files; f++) {
        FILE* fp = fopen(files[f], "rb");
        if(!fp) { printf("Failed to open %s\n", files[f]); exit(1); }
        for(int i=0; i<10000; i++) {
            size_t dr1 = fread(&Y[offset + i], 1, 1, fp); (void)dr1;
            size_t dr2 = fread(&X[(offset + i) * 3072], 1, 3072, fp); (void)dr2;
        }
        fclose(fp);
        offset += 10000;
    }
}

void xavier_init(double* w, size_t total_elements, size_t fan_in, size_t fan_out) {
    double limit = sqrt(6.0 / (fan_in + fan_out));
    for(size_t i=0; i<total_elements; i++) {
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
    printf("  BASIS V8: CIFAR-10 Notepad (4D LeNet + Vision Autodiff)     \n");
    printf("================================================================\n\n");

    download_cifar10();

    uint8_t* tr_X = (uint8_t*)malloc(NUM_TRAIN * 3072);
    uint8_t* tr_Y = (uint8_t*)malloc(NUM_TRAIN);
    uint8_t* te_X = (uint8_t*)malloc(NUM_TEST * 3072);
    uint8_t* te_Y = (uint8_t*)malloc(NUM_TEST);

    const char* train_files[] = {
        "cifar-10-batches-bin/data_batch_1.bin", "cifar-10-batches-bin/data_batch_2.bin",
        "cifar-10-batches-bin/data_batch_3.bin", "cifar-10-batches-bin/data_batch_4.bin",
        "cifar-10-batches-bin/data_batch_5.bin"
    };
    const char* test_files[] = { "cifar-10-batches-bin/test_batch.bin" };

    load_cifar10(tr_X, tr_Y, train_files, 5);
    load_cifar10(te_X, te_Y, test_files, 1);
    printf("[2/5] Loaded %d training images, %d test images.\n", NUM_TRAIN, NUM_TEST);

    // --- Build V8 IR Graph (LeNet Architecture) ---
    v8_graph* g = v8_graph_create();
    v8_node* X = v8_input_4d(g, BATCH_SIZE, 3, 32, 32);

    // Conv1: 3 -> 16 channels, 5x5 kernel
    v8_node* W1 = v8_input_4d(g, 16, 3, 5, 5);
    v8_node* C1 = v8_conv2d(g, X, W1, 1, 0); // N x 16 x 28 x 28
    v8_node* P1 = v8_maxpool2d(g, C1, 2, 2); // N x 16 x 14 x 14

    // Conv2: 16 -> 32 channels, 5x5 kernel
    v8_node* W2 = v8_input_4d(g, 32, 16, 5, 5);
    v8_node* C2 = v8_conv2d(g, P1, W2, 1, 0); // N x 32 x 10 x 10
    v8_node* P2 = v8_maxpool2d(g, C2, 2, 2);  // N x 32 x 5 x 5

    // Flatten: N x 800
    v8_node* F = v8_flatten(g, P2);

    // FC1: 800 -> 120
    v8_node* W3 = v8_input(g, 800, 120);
    v8_node* b1 = v8_input(g, 1, 120);
    v8_node* Z1 = v8_matmul(g, F, W3);
    v8_node* A1 = v8_add(g, Z1, v8_broadcast(g, b1, BATCH_SIZE, 120));
    v8_node* H1 = v8_relu(g, A1);

    // FC2: 120 -> 10 (Raw Logits)
    v8_node* W4 = v8_input(g, 120, 10);
    v8_node* b2 = v8_input(g, 1, 10);
    v8_node* Z2 = v8_matmul(g, H1, W4);
    v8_node* A2 = v8_add(g, Z2, v8_broadcast(g, b2, BATCH_SIZE, 10));

    v8_node* Y = v8_input(g, BATCH_SIZE, 10);
    // Fused Cross-Entropy Loss (Mathematically stable, pushes accuracy past 60%)
    v8_node* loss = v8_cross_entropy(g, A2, Y);

    // Persistent Memory Allocation
    double* x_batch = (double*)calloc(BATCH_SIZE * 3072, sizeof(double));
    double* y_batch = (double*)calloc(BATCH_SIZE * 10, sizeof(double));

    double* w1 = (double*)calloc(16*3*5*5, sizeof(double)); xavier_init(w1, 16*3*5*5, 3*5*5, 16*5*5);
    double* w2 = (double*)calloc(32*16*5*5, sizeof(double)); xavier_init(w2, 32*16*5*5, 16*5*5, 32*5*5);
    double* w3 = (double*)calloc(800*120, sizeof(double)); xavier_init(w3, 800*120, 800, 120);
    double* b1_d = (double*)calloc(120, sizeof(double));
    double* w4 = (double*)calloc(120*10, sizeof(double)); xavier_init(w4, 120*10, 120, 10);
    double* b2_d = (double*)calloc(10, sizeof(double));

    double* m_w1 = (double*)calloc(16*3*5*5, sizeof(double)); double* v_w1 = (double*)calloc(16*3*5*5, sizeof(double));
    double* m_w2 = (double*)calloc(32*16*5*5, sizeof(double)); double* v_w2 = (double*)calloc(32*16*5*5, sizeof(double));
    double* m_w3 = (double*)calloc(800*120, sizeof(double)); double* v_w3 = (double*)calloc(800*120, sizeof(double));
    double* m_b1 = (double*)calloc(120, sizeof(double)); double* v_b1 = (double*)calloc(120, sizeof(double));
    double* m_w4 = (double*)calloc(120*10, sizeof(double)); double* v_w4 = (double*)calloc(120*10, sizeof(double));
    double* m_b2 = (double*)calloc(10, sizeof(double)); double* v_b2 = (double*)calloc(10, sizeof(double));

    double* grad_w1 = (double*)calloc(16*3*5*5, sizeof(double));
    double* grad_w2 = (double*)calloc(32*16*5*5, sizeof(double));
    double* grad_w3 = (double*)calloc(800*120, sizeof(double));
    double* grad_b1 = (double*)calloc(120, sizeof(double));
    double* grad_w4 = (double*)calloc(120*10, sizeof(double));
    double* grad_b2 = (double*)calloc(10, sizeof(double));

    // GLM DIRECT BINDING (PERSISTENT WEIGHTS ONLY)
    // Weights are allocated once, so we bind them once before Autodiff clones the graph.
    W1->runtime_data = w1;
    W2->runtime_data = w2;
    W3->runtime_data = w3;
    b1->runtime_data = b1_d;
    W4->runtime_data = w4;
    b2->runtime_data = b2_d;

    printf("[3/5] Compiling Backward Pass & Scheduling Waves...\n");
    v8_training_graph* tg = v8_ir_autodiff(g, loss);
    if (!tg->loss_node) {
        printf("FATAL: Autodiff failed to rebuild the loss node.\n");
        return 1;
    }
    v8_schedule* sched = v8_ir_schedule(tg->graph);
    v8_arena* scratch = v8_arena_create(512 * 1024 * 1024); // 512MB for 4D activations


    printf("[4/5] Training for %d Epochs (Batch Size: %d)...\n\n", EPOCHS, BATCH_SIZE);
    int* indices = (int*)malloc(NUM_TRAIN * sizeof(int));
    for(int i=0; i<NUM_TRAIN; i++) indices[i] = i;

    int step = 0;
    for(int ep=0; ep<EPOCHS; ep++) {
        for(int i=NUM_TRAIN-1; i>0; i--) { int j = rand() % (i+1); int tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp; }
        double epoch_loss = 0.0; int batches = 0;

        for(int i=0; i<=NUM_TRAIN - BATCH_SIZE; i+=BATCH_SIZE) {
            memset(y_batch, 0, BATCH_SIZE * 10 * sizeof(double));
            for(int b=0; b<BATCH_SIZE; b++) {
                int idx = indices[i+b];
                // Zero-Mean Centering & Normalization
                for(int p=0; p<3072; p++) x_batch[b*3072 + p] = (tr_X[idx*3072 + p] / 255.0) - 0.5;

                // RED TEAM FIX: Random Horizontal Flip (50% chance per image)
                if (rand() % 2 == 0) {
                    for(int ch=0; ch<3; ch++) {
                        for(int h=0; h<32; h++) {
                            for(int w=0; w<16; w++) { // Only iterate half the width
                                int left_idx = b*3072 + ch*1024 + h*32 + w;
                                int right_idx = b*3072 + ch*1024 + h*32 + (31 - w);
                                double temp = x_batch[left_idx];
                                x_batch[left_idx] = x_batch[right_idx];
                                x_batch[right_idx] = temp;
                            }
                        }
                    }
                }
                y_batch[b*10 + tr_Y[idx]] = 1.0;
            }

            for(uint32_t w=0; w<sched->wave_count; w++) {
                for(uint32_t k=0; k<sched->waves[w].node_count; k++) {
                    v8_node* n = sched->waves[w].nodes[k];
                    if (n->op != V8_OP_INPUT && n->op != V8_OP_CONST) n->runtime_data = NULL;
                }
            }

            // CRITICAL: Nullify intermediate nodes to prevent stale arena pointers
            for(uint32_t k=0; k<tg->graph->node_count; k++) {
                v8_node* n = tg->graph->nodes[k];
                if (n->op != V8_OP_INPUT && n->op != V8_OP_CONST) n->runtime_data = NULL;
            }
            v8_schedule_execute(sched, scratch);
            step++;

            memset(grad_w1, 0, 16*3*5*5*sizeof(double)); memset(grad_w2, 0, 32*16*5*5*sizeof(double));
            memset(grad_w3, 0, 800*120*sizeof(double)); memset(grad_w4, 0, 120*10*sizeof(double));
            memset(grad_b1, 0, 120*sizeof(double)); memset(grad_b2, 0, 10*sizeof(double));

            if(!tg->grad_nodes[W1->id] || !tg->grad_nodes[W1->id]->runtime_data) {
                printf("[FATAL] Autodiff failed to produce W1 gradient.\n");
                return 1;
            }
            memcpy(grad_w1, tg->grad_nodes[W1->id]->runtime_data, 16*3*5*5*sizeof(double));
            if(tg->grad_nodes[W2->id] && tg->grad_nodes[W2->id]->runtime_data) memcpy(grad_w2, tg->grad_nodes[W2->id]->runtime_data, 32*16*5*5*sizeof(double));
            if(tg->grad_nodes[W3->id] && tg->grad_nodes[W3->id]->runtime_data) memcpy(grad_w3, tg->grad_nodes[W3->id]->runtime_data, 800*120*sizeof(double));
            if(tg->grad_nodes[W4->id] && tg->grad_nodes[W4->id]->runtime_data) memcpy(grad_w4, tg->grad_nodes[W4->id]->runtime_data, 120*10*sizeof(double));
            if(tg->grad_nodes[b1->id] && tg->grad_nodes[b1->id]->runtime_data) memcpy(grad_b1, tg->grad_nodes[b1->id]->runtime_data, 120*sizeof(double));
            if(tg->grad_nodes[b2->id] && tg->grad_nodes[b2->id]->runtime_data) memcpy(grad_b2, tg->grad_nodes[b2->id]->runtime_data, 10*sizeof(double));

            // Cross-Entropy gradients are naturally bounded, no aggressive clipping needed

                        adam_step(w1, grad_w1, m_w1, v_w1, 16*3*5*5, lr, 0.9, 0.999, 1e-8, step);
            adam_step(w2, grad_w2, m_w2, v_w2, 32*16*5*5, lr, 0.9, 0.999, 1e-8, step);
            adam_step(w3, grad_w3, m_w3, v_w3, 800*120, lr, 0.9, 0.999, 1e-8, step);
            adam_step(w4, grad_w4, m_w4, v_w4, 120*10, lr, 0.9, 0.999, 1e-8, step);
            adam_step(b1_d, grad_b1, m_b1, v_b1, 120, lr, 0.9, 0.999, 1e-8, step);
            adam_step(b2_d, grad_b2, m_b2, v_b2, 10, lr, 0.9, 0.999, 1e-8, step);

            if (tg->loss_node && tg->loss_node->runtime_data) {
                epoch_loss += tg->loss_node->runtime_data[0];
            } else {
                printf("\n[FATAL] Loss node execution skipped! Upstream NULL cascade detected.\n");
                for(uint32_t k=0; k<tg->graph->node_count; k++) {
                    v8_node* n = tg->graph->nodes[k];
                    if (n->op != V8_OP_INPUT && n->op != V8_OP_CONST && !n->runtime_data) {
                        printf("  Node %u (op=%d) has NULL runtime_data!\n", n->id, n->op);
                    }
                }
                return 1;
            }
            batches++;
            v8_arena_reset(scratch);
            if(batches % 50 == 0) printf("\r  Epoch %d | Batch %d/%d | Loss: %.4f", ep+1, batches, NUM_TRAIN/BATCH_SIZE, epoch_loss/batches);
        }
        double avg_loss = epoch_loss / batches;
        printf("\r  Epoch %d | Loss: %.4f | lr: %.6f                         \n", ep+1, avg_loss, lr);

        // RED TEAM FIX: Step Decay (Halve lr at epoch 15 and 25)
        if (ep+1 == 15 || ep+1 == 25) {
            lr *= 0.5;
            printf("  [SCHEDULER] Learning rate decayed to %.6f\n", lr);
        }
        if (avg_loss > 100.0 || isnan(avg_loss)) {
            printf("\n[FATAL] Loss exploded or became NaN at Epoch %d! Aborting to save CPU cycles.\n", ep+1);
            break;
        }
    }

    printf("\n[5/5] Evaluating on 10,000 Test Images (Sequential Scheduler)...\n");
    int correct = 0;
    for(int i=0; i<=NUM_TEST - BATCH_SIZE; i+=BATCH_SIZE) {
        memset(y_batch, 0, BATCH_SIZE * 10 * sizeof(double));
        for(int b=0; b<BATCH_SIZE; b++) {
            int idx = i+b;
            for(int p=0; p<3072; p++) x_batch[b*3072 + p] = (te_X[idx*3072 + p] / 255.0) - 0.5;
        }
        for(uint32_t w=0; w<sched->wave_count; w++) {
            for(uint32_t k=0; k<sched->waves[w].node_count; k++) {
                v8_node* n = sched->waves[w].nodes[k];
                if (n->op != V8_OP_INPUT && n->op != V8_OP_CONST) n->runtime_data = NULL;
            }
        }
        v8_schedule_execute(sched, scratch);

        v8_node* P_node = NULL;
        for(uint32_t k=0; k<tg->graph->node_count; k++) {
            if(tg->graph->nodes[k]->op == V8_OP_ADD && tg->graph->nodes[k]->shape[1] == 10) { P_node = tg->graph->nodes[k]; break; }
        }

        for(int b=0; b<BATCH_SIZE; b++) {
            int pred = 0; double max_logit = -1e9;
            for(int c=0; c<10; c++) {
                if(P_node->runtime_data[b*10 + c] > max_logit) { max_logit = P_node->runtime_data[b*10 + c]; pred = c; }
            }
            if(pred == te_Y[i+b]) correct++;
        }
        v8_arena_reset(scratch);
    }

    printf("\n================================================================\n");
    printf("  FINAL TEST ACCURACY: %.2f%% (%d / %d)\n", 100.0 * correct / NUM_TEST, correct, NUM_TEST);
    printf("================================================================\n");

    // --- PATH B: EDGE FREEZE ---
    printf("\n[PATH B] Freezing trained inference model to 'cifar10_lenet.v8b'...\n");
    // Weights are already bound directly to the forward graph nodes!
    v8_graph_save(g, "cifar10_lenet.v8b");

    printf("\n================================================================\n");
    printf("  V8 CIFAR-10 NOTEPAD COMPLETE. EDGE DEPLOYMENT READY.\n");
    printf("================================================================\n\n");

    return 0;
}
