#include <lancius.h>
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
double lr = 0.0003; // V12 Safe LR // V10: Bumped LR for He Init
#define NUM_TRAIN 50000
#define NUM_TEST 10000

/* Despot truth: unchecked fread/system (was: truncated bins trained on
 * uninit bytes, failed downloads proceeded anyway). */
uint32_t read_int(FILE* f) {
    uint8_t b[4];
    if (!f || fread(b, 1, 4, f) != 4) { fprintf(stderr, "FATAL: truncated read\n"); exit(1); }
    return ((uint32_t)b[0] << 24) | ((uint32_t)b[1] << 16) | ((uint32_t)b[2] << 8) | (uint32_t)b[3];
}

static int cifar_step(const char* cmd) {
    int rc = system(cmd);
    if (rc != 0) { fprintf(stderr, "FATAL: step failed (%d): %s\n", rc, cmd); return 0; }
    return 1;
}

void download_cifar10() {
    if (access("cifar-10-batches-bin/data_batch_1.bin", F_OK) == 0) return;
    printf("[1/5] Downloading CIFAR-10 Binary Dataset...\n");
    if (!cifar_step("curl -s -L https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz -o cifar.tar.gz")) exit(1);
    /* Despot V6 truth: tar-slip validated (was blind extract of MITM archive). */
    if (!cifar_step("tar -tzf cifar.tar.gz > /tmp/opencode/tarlist.txt")) exit(1);
    {
        FILE* lf = fopen("/tmp/opencode/tarlist.txt", "r");
        if (!lf) { fprintf(stderr, "FATAL: cannot list archive\n"); exit(1); }
        char line[1024];
        while (fgets(line, sizeof(line), lf)) {
            size_t L = strlen(line);
            while (L && (line[L-1] == '\n' || line[L-1] == '\r')) line[--L] = '\0';
            if (L == 0) continue;
            if (line[0] == '/' || strstr(line, "..") != NULL) { fprintf(stderr, "FATAL: unsafe tar member: %s\n", line); fclose(lf); exit(1); }
        }
        fclose(lf);
    }
    if (!cifar_step("tar -xzf cifar.tar.gz")) exit(1);
    if (!cifar_step("rm cifar.tar.gz")) exit(1);
    if (access("cifar-10-batches-bin/data_batch_1.bin", F_OK) != 0) { fprintf(stderr, "FATAL: CIFAR-10 download incomplete\n"); exit(1); }
}

void load_cifar10(uint8_t* X, uint8_t* Y, const char** files, int num_files) {
    int offset = 0;
    for(int f=0; f<num_files; f++) {
        FILE* fp = fopen(files[f], "rb");
        if(!fp) { printf("Failed to open %s\n", files[f]); exit(1); }
        for(int i=0; i<10000; i++) {
            if (fread(&Y[offset + i], 1, 1, fp) != 1) { fprintf(stderr, "FATAL: truncated %s label %d\n", files[f], i); fclose(fp); exit(1); }
            if (fread(&X[(offset + i) * 3072], 1, 3072, fp) != 3072) { fprintf(stderr, "FATAL: truncated %s image %d\n", files[f], i); fclose(fp); exit(1); }
        }
        fclose(fp);
        offset += 10000;
    }
}

// V10: He Initialization (Kaiming) for ReLU networks
void he_init(double* w, size_t total_elements, size_t fan_in) {
    double std_dev = sqrt(2.0 / fan_in);
    for(size_t i=0; i<total_elements; i++) {
        double u1 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
        double u2 = ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
        double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);
        w[i] = z * std_dev;
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
    printf("  Lancius: CIFAR-10 Notepad (Same Padding + He Init)         \n");
    printf("================================================================\n\n");

    download_cifar10();

    /* Despot truth: every malloc below is checked (was: OOM derefed). */
#define CIFAR_NEED(ptr, what) do { if (!(ptr)) { fprintf(stderr, "FATAL: OOM %s\n", what); return 1; } } while(0)
    uint8_t* tr_X = (uint8_t*)malloc(NUM_TRAIN * 3072); CIFAR_NEED(tr_X, "tr_X");
    uint8_t* tr_Y = (uint8_t*)malloc(NUM_TRAIN); CIFAR_NEED(tr_Y, "tr_Y");
    uint8_t* te_X = (uint8_t*)malloc(NUM_TEST  * 3072); CIFAR_NEED(te_X, "te_X");
    uint8_t* te_Y = (uint8_t*)malloc(NUM_TEST); CIFAR_NEED(te_Y, "te_Y");

    const char* train_files[] = {
        "cifar-10-batches-bin/data_batch_1.bin", "cifar-10-batches-bin/data_batch_2.bin",
        "cifar-10-batches-bin/data_batch_3.bin", "cifar-10-batches-bin/data_batch_4.bin",
        "cifar-10-batches-bin/data_batch_5.bin"
    };
    const char* test_files[] = { "cifar-10-batches-bin/test_batch.bin" };

    load_cifar10(tr_X, tr_Y, train_files, 5);
    load_cifar10(te_X, te_Y, test_files, 1);
    printf("[2/5] Loaded %d training images, %d test images.\n", NUM_TRAIN, NUM_TEST);

    lancius_graph* g = lancius_graph_create();
    lancius_node* X = lancius_input_4d(g, BATCH_SIZE, 3, 32, 32);

    // V10: Same Padding (pad=2) preserves spatial dimensions
    lancius_node* W1 = lancius_input_4d(g, 16, 3, 5, 5);
    lancius_node* C1 = lancius_conv2d(g, X, W1, 1, 2); // N x 16 x 32 x 32
    lancius_node* R1 = lancius_relu(g, C1);
    lancius_node* P1 = lancius_maxpool2d(g, R1, 2, 2); // N x 16 x 16 x 16

    lancius_node* W2 = lancius_input_4d(g, 32, 16, 5, 5);
    lancius_node* C2 = lancius_conv2d(g, P1, W2, 1, 2); // N x 32 x 16 x 16
    lancius_node* R2 = lancius_relu(g, C2);
    lancius_node* P2 = lancius_maxpool2d(g, R2, 2, 2);  // N x 32 x 8 x 8

    lancius_node* F = lancius_flatten(g, P2); // N x 2048

    // V10: Expanded FC1 (2048 -> 120)
    lancius_node* W3 = lancius_input(g, 2048, 120);
    lancius_node* b1 = lancius_input(g, 1, 120);
    lancius_node* Z1 = lancius_matmul(g, F, W3);
    lancius_node* A1 = lancius_add(g, Z1, lancius_broadcast(g, b1, BATCH_SIZE, 120));
    lancius_node* H1 = lancius_relu(g, A1);

    lancius_node* W4 = lancius_input(g, 120, 10);
    lancius_node* b2 = lancius_input(g, 1, 10);
    lancius_node* Z2 = lancius_matmul(g, H1, W4);
    lancius_node* A2 = lancius_add(g, Z2, lancius_broadcast(g, b2, BATCH_SIZE, 10));

    lancius_node* Y = lancius_input(g, BATCH_SIZE, 10);
    lancius_node* loss = lancius_cross_entropy(g, A2, Y);

    double* x_batch = (double*)calloc(BATCH_SIZE * 3072, sizeof(double)); CIFAR_NEED(x_batch, "x_batch");
    double* y_batch = (double*)calloc(BATCH_SIZE * 10, sizeof(double)); CIFAR_NEED(y_batch, "y_batch");

    double* w1 = (double*)calloc(16*3*5*5, sizeof(double)); CIFAR_NEED(w1, "w1"); he_init(w1, 16*3*5*5, 3*5*5);
    double* w2 = (double*)calloc(32*16*5*5, sizeof(double)); CIFAR_NEED(w2, "w2"); he_init(w2, 32*16*5*5, 16*5*5);
    double* w3 = (double*)calloc(2048*120, sizeof(double)); CIFAR_NEED(w3, "w3"); he_init(w3, 2048*120, 2048);
    double* b1_d = (double*)calloc(120, sizeof(double)); CIFAR_NEED(b1_d, "b1");
    double* w4 = (double*)calloc(120*10, sizeof(double)); CIFAR_NEED(w4, "w4"); he_init(w4, 120*10, 120);
    double* b2_d = (double*)calloc(10, sizeof(double)); CIFAR_NEED(b2_d, "b2");

    double* m_w1 = (double*)calloc(16*3*5*5, sizeof(double)); CIFAR_NEED(m_w1, "m_w1"); double* v_w1 = (double*)calloc(16*3*5*5, sizeof(double)); CIFAR_NEED(v_w1, "v_w1");
    double* m_w2 = (double*)calloc(32*16*5*5, sizeof(double)); CIFAR_NEED(m_w2, "m_w2"); double* v_w2 = (double*)calloc(32*16*5*5, sizeof(double)); CIFAR_NEED(v_w2, "v_w2");
    double* m_w3 = (double*)calloc(2048*120, sizeof(double)); CIFAR_NEED(m_w3, "m_w3"); double* v_w3 = (double*)calloc(2048*120, sizeof(double)); CIFAR_NEED(v_w3, "v_w3");
    double* m_b1 = (double*)calloc(120, sizeof(double)); CIFAR_NEED(m_b1, "m_b1"); double* v_b1 = (double*)calloc(120, sizeof(double)); CIFAR_NEED(v_b1, "v_b1");
    double* m_w4 = (double*)calloc(120*10, sizeof(double)); CIFAR_NEED(m_w4, "m_w4"); double* v_w4 = (double*)calloc(120*10, sizeof(double)); CIFAR_NEED(v_w4, "v_w4");
    double* m_b2 = (double*)calloc(10, sizeof(double)); CIFAR_NEED(m_b2, "m_b2"); double* v_b2 = (double*)calloc(10, sizeof(double)); CIFAR_NEED(v_b2, "v_b2");

    double* grad_w1 = (double*)calloc(16*3*5*5, sizeof(double)); CIFAR_NEED(grad_w1, "grad_w1");
    double* grad_w2 = (double*)calloc(32*16*5*5, sizeof(double)); CIFAR_NEED(grad_w2, "grad_w2");
    double* grad_w3 = (double*)calloc(2048*120, sizeof(double)); CIFAR_NEED(grad_w3, "grad_w3");
    double* grad_b1 = (double*)calloc(120, sizeof(double)); CIFAR_NEED(grad_b1, "grad_b1");
    double* grad_w4 = (double*)calloc(120*10, sizeof(double)); CIFAR_NEED(grad_w4, "grad_w4");
    double* grad_b2 = (double*)calloc(10, sizeof(double)); CIFAR_NEED(grad_b2, "grad_b2");

    X->runtime_data = x_batch;
    Y->runtime_data = y_batch;
    W1->runtime_data = w1;
    W2->runtime_data = w2;
    W3->runtime_data = w3;
    b1->runtime_data = b1_d;
    W4->runtime_data = w4;
    b2->runtime_data = b2_d;

    printf("[3/5] Compiling Backward Pass & Scheduling Waves...\n");
    printf("[V10S] Running Graph Optimizations...\n");
    lancius_optimize_fusion(g);

    lancius_training_graph* tg = lancius_ir_autodiff(g, loss);
    if (!tg) { fprintf(stderr, "FATAL: autodiff failed (see stderr)\n"); return 1; }
    lancius_schedule* sched = lancius_ir_schedule(tg->graph);
    if (!sched) { fprintf(stderr, "FATAL: schedule failed\n"); return 1; }
    lancius_arena* scratch = lancius_arena_create(512 * 1024 * 1024);
    if (!scratch) { fprintf(stderr, "FATAL: OOM scratch arena\n"); return 1; }

    printf("[4/5] Training for %d Epochs (Batch Size: %d)...\n\n", EPOCHS, BATCH_SIZE);
    int* indices = (int*)malloc(NUM_TRAIN * sizeof(int));
    CIFAR_NEED(indices, "indices");
    for(int i=0; i<NUM_TRAIN; i++) indices[i] = i;

    int step = 0;
    for(int ep=0; ep<EPOCHS; ep++) {
        for(int i=NUM_TRAIN-1; i>0; i--) { int j = rand() % (i+1); int tmp = indices[i]; indices[i] = indices[j]; indices[j] = tmp; }
        double epoch_loss = 0.0; int batches = 0;

        for(int i=0; i<=NUM_TRAIN - BATCH_SIZE; i+=BATCH_SIZE) {
            memset(y_batch, 0, BATCH_SIZE * 10 * sizeof(double));
            for(int b=0; b<BATCH_SIZE; b++) {
                int idx = indices[i+b];
                // Hostile fix: align with train_1_epoch.py Normalize((0.5,),(0.5,)) => (x-0.5)/0.5 in [-1,1].
                // Was (x/255-0.5) in [-0.5,0.5] (half scale, LR mistransfer + parity gap).
                for(int p=0; p<3072; p++) x_batch[b*3072 + p] = ((tr_X[idx*3072 + p] / 255.0) - 0.5) / 0.5;

                if (rand() % 2 == 0) {
                    for(int ch=0; ch<3; ch++) {
                        for(int h=0; h<32; h++) {
                            for(int w=0; w<16; w++) {
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
                    lancius_node* n = sched->waves[w].nodes[k];
                    if (n->op != LANCIUS_OP_INPUT && n->op != LANCIUS_OP_CONST) n->runtime_data = NULL;
                }
            }
            for(uint32_t k=0; k<tg->graph->node_count; k++) {
                lancius_node* n = tg->graph->nodes[k];
                if (n->op != LANCIUS_OP_INPUT && n->op != LANCIUS_OP_CONST) n->runtime_data = NULL;
            }

            lancius_schedule_execute(sched, scratch);
            step++;

            memset(grad_w1, 0, 16*3*5*5*sizeof(double)); memset(grad_w2, 0, 32*16*5*5*sizeof(double));
            memset(grad_w3, 0, 2048*120*sizeof(double)); memset(grad_w4, 0, 120*10*sizeof(double));
            memset(grad_b1, 0, 120*sizeof(double));  memset(grad_b2, 0, 10*sizeof(double));

            if(tg->grad_nodes[W1->id] && tg->grad_nodes[W1->id]->runtime_data) memcpy(grad_w1, tg->grad_nodes[W1->id]->runtime_data, 16*3*5*5*sizeof(double));
            if(tg->grad_nodes[W2->id] && tg->grad_nodes[W2->id]->runtime_data) memcpy(grad_w2, tg->grad_nodes[W2->id]->runtime_data, 32*16*5*5*sizeof(double));
            if(tg->grad_nodes[W3->id] && tg->grad_nodes[W3->id]->runtime_data) memcpy(grad_w3, tg->grad_nodes[W3->id]->runtime_data, 2048*120*sizeof(double));
            if(tg->grad_nodes[W4->id] && tg->grad_nodes[W4->id]->runtime_data) memcpy(grad_w4, tg->grad_nodes[W4->id]->runtime_data, 120*10*sizeof(double));
            if(tg->grad_nodes[b1->id] && tg->grad_nodes[b1->id]->runtime_data) memcpy(grad_b1, tg->grad_nodes[b1->id]->runtime_data, 120*sizeof(double));
            if(tg->grad_nodes[b2->id] && tg->grad_nodes[b2->id]->runtime_data) memcpy(grad_b2, tg->grad_nodes[b2->id]->runtime_data, 10*sizeof(double));

            // V12 GLOBAL NORM CLIPPING (Preserves Gradient Direction)
            double global_norm = 0.0;
            for(size_t i=0; i<16*3*5*5; i++) global_norm += grad_w1[i] * grad_w1[i];
            for(size_t i=0; i<32*16*5*5; i++) global_norm += grad_w2[i] * grad_w2[i];
            for(size_t i=0; i<2048*120; i++) global_norm += grad_w3[i] * grad_w3[i];
            for(size_t i=0; i<120*10; i++) global_norm += grad_w4[i] * grad_w4[i];
            for(size_t i=0; i<120; i++) global_norm += grad_b1[i] * grad_b1[i];
            for(size_t i=0; i<10; i++) global_norm += grad_b2[i] * grad_b2[i];
            global_norm = sqrt(global_norm);
            
            double max_norm = 1.0;
            if (global_norm > max_norm) {
                double scale = max_norm / (global_norm + 1e-6);
                for(size_t i=0; i<16*3*5*5; i++) grad_w1[i] *= scale;
                for(size_t i=0; i<32*16*5*5; i++) grad_w2[i] *= scale;
                for(size_t i=0; i<2048*120; i++) grad_w3[i] *= scale;
                for(size_t i=0; i<120*10; i++) grad_w4[i] *= scale;
                for(size_t i=0; i<120; i++) grad_b1[i] *= scale;
                for(size_t i=0; i<10; i++) grad_b2[i] *= scale;
            }

            adam_step(w1, grad_w1, m_w1, v_w1, 16*3*5*5, lr, 0.9, 0.999, 1e-8, step);
            adam_step(w2, grad_w2, m_w2, v_w2, 32*16*5*5, lr, 0.9, 0.999, 1e-8, step);
            adam_step(w3, grad_w3, m_w3, v_w3, 2048*120, lr, 0.9, 0.999, 1e-8, step);
            adam_step(w4, grad_w4, m_w4, v_w4, 120*10, lr, 0.9, 0.999, 1e-8, step);
            adam_step(b1_d, grad_b1, m_b1, v_b1, 120, lr, 0.9, 0.999, 1e-8, step);
            adam_step(b2_d, grad_b2, m_b2, v_b2, 10, lr, 0.9, 0.999, 1e-8, step);

            /* Despot truth: track raw loss honestly; clamped average is censored.
             * Accumulate raw for the abort gate, clamped only for display. */
            if (tg->loss_node && tg->loss_node->runtime_data) {
                double l_raw = tg->loss_node->runtime_data[0];
                double l = l_raw;
                if (l < 0.0 || isnan(l) || l > 1000.0) l = 2.3025;
                if (l != l_raw) fprintf(stderr, "[TRAIN] clamped loss raw=%g -> %g\n", l_raw, l);
                epoch_loss += l;
                /* Raw gate: NaN or explosion aborts immediately, never hidden. */
                if (isnan(l_raw) || l_raw > 1000.0 || l_raw < 0.0) {
                    printf("\n[FATAL] Raw loss diverged (raw=%g) at Epoch %d batch %d! Aborting.\n", l_raw, ep+1, batches+1);
                    free(indices);
                    return 1;
                }
            }
            batches++;
            lancius_arena_reset(scratch);
            if(batches % 50 == 0)  printf("\r  Epoch %d | Batch %d/%d | Loss: %.4f", ep+1, batches, NUM_TRAIN/BATCH_SIZE, epoch_loss/batches);
        }
        double avg_loss = epoch_loss / batches;
        printf("\r  Epoch %d | Loss: %.4f | lr: %.6f                         \n", ep+1, avg_loss, lr);

        if (ep+1 == 15 || ep+1 == 25) {
            lr *= 0.5;
            printf("  [SCHEDULER] Learning rate decayed to %.6f\n", lr);
        }
        if (avg_loss > 100.0 || isnan(avg_loss)) {
            printf("\n[FATAL] Loss exploded or became NaN at Epoch %d! Aborting.\n", ep+1);
            break;
        }
    }

    printf("\n[5/5] Evaluating on 10,000 Test Images...\n");
    int correct = 0, evaluated = 0;
    // Hostile fix: track A2 (logits) by pointer identity via loss inputs[0].
    // A2 is fwd_to_full[orig A2 id]; loss_node is full CE; its inputs[0] is full A2.
    lancius_node* A2_full = (tg->loss_node && tg->loss_node->input_count > 0) ? (lancius_node*)tg->loss_node->inputs[0] : NULL;
    for(int i=0; i<=NUM_TEST - BATCH_SIZE; i+=BATCH_SIZE) {
        memset(y_batch, 0, BATCH_SIZE * 10 * sizeof(double));
        for(int b=0; b<BATCH_SIZE; b++) {
            int idx = i+b;
            for(int p=0; p<3072; p++) x_batch[b*3072 + p] = ((te_X[idx*3072 + p] / 255.0) - 0.5) / 0.5;
        }
        for(uint32_t w=0; w<sched->wave_count; w++) {
            for(uint32_t k=0; k<sched->waves[w].node_count; k++) {
                lancius_node* n = sched->waves[w].nodes[k];
                if (n->op != LANCIUS_OP_INPUT && n->op != LANCIUS_OP_CONST) n->runtime_data = NULL;
            }
        }
        lancius_schedule_execute(sched, scratch);

        lancius_node* P_node = A2_full;

        /* Despot truth: break on missing logits still counted stale pred=0 as
         * correct when label was 0 (was: inflated accuracy). Skip instead. */
        if (!P_node || !P_node->runtime_data) continue;
        for(int b=0; b<BATCH_SIZE; b++) {
            int pred = 0; double max_logit = -1e9;
            for(int c=0; c<10; c++) {
                if(P_node->runtime_data[b*10 + c] > max_logit) { max_logit = P_node->runtime_data[b*10 + c]; pred = c; }
            }
            if(pred == te_Y[i+b]) correct++;
            evaluated++;
        }
        lancius_arena_reset(scratch);
    }

    if (evaluated <= 0) { fprintf(stderr, "[TRAIN] FATAL: no test batches evaluated\n"); return 1; }
    printf("\n================================================================\n");
    printf("  FINAL TEST ACCURACY: %.2f%% (%d / %d)\n", 100.0 * correct / evaluated, correct, evaluated);
    printf("================================================================\n");

    /* Despot truth: trainers must earn exit 0. Chance is 10%%; anything at or
     * below chance means no learning (or a broken eval path). */
    if (correct * 10 <= evaluated) {
        fprintf(stderr, "[TRAIN] FATAL: accuracy %.2f%% <= chance; refusing green exit.\n",
            100.0 * correct / evaluated);
        return 1;
    }

    printf("\n[PATH B] Freezing & Quantizing model to INT8...\n");
    lancius_quantize_graph(g);
    /* Despot truth: failed save was reported as success. */
    if (lancius_graph_save(g, "cifar10_lenet.lancius") != 0) {
        fprintf(stderr, "[TRAIN] FATAL: model save failed\n");
        return 1;
    }

    printf("\n================================================================\n");
    printf("  V10S CIFAR-10 NOTEPAD COMPLETE. EDGE DEPLOYMENT READY.\n");
    printf("================================================================\n\n");

    return 0;
}
