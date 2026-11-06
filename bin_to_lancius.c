/* bin_to_lancius.c -- convert train_i6 checkpoint (.bin) to Lancius .lancius model.
 *
 * Reads an I6 checkpoint (magic LVC6) and writes a .lancius v2 model with the
 * same weights, so run_big_i6.py can evaluate it head-to-head vs the I3 ALU pool.
 *
 * Build:
 *   gcc -Wall -Wextra -O3 -mavx2 -mfma -fopenmp -std=c11 -I./include \
 *       -o build/bin/bin_to_lancius bin_to_lancius.c build/lib/liblancius.a -lm -fopenmp -lpthread
 *
 * Usage:
 *   build/bin/bin_to_lancius <checkpoint.bin> <output.lancius> [H1 H2]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>

#include <lancius.h>

#define CKPT_MAGIC 0x4C564336u /* "LVC6" */

static void die(const char* m){ fprintf(stderr, "FATAL: %s\n", m); exit(2); }

int main(int argc, char** argv){
    if (argc < 3) {
        fprintf(stderr, "usage: bin_to_lancius <checkpoint.bin> <output.lancius> [H1 H2]\n");
        return 2;
    }
    const char* ckpt_path = argv[1];
    const char* out_path  = argv[2];
    size_t H1 = 4096, H2 = 2560;
    if (argc > 3) H1 = strtoul(argv[3], NULL, 10);
    if (argc > 4) H2 = strtoul(argv[4], NULL, 10);

    size_t FEAT = 65, NCLASS = 5;

    FILE* f = fopen(ckpt_path, "rb");
    if (!f) die("cannot open checkpoint");

    uint32_t magic = 0;
    if (fread(&magic, sizeof magic, 1, f) != 1 || magic != CKPT_MAGIC)
        die("bad checkpoint magic");
    size_t dims[5];
    if (fread(dims, sizeof dims, 1, f) != 1) die("short dims");
    if (dims[0] != FEAT || dims[1] != NCLASS || dims[2] != H1 || dims[3] != H2)
        die("checkpoint dims mismatch");
    long step = 0; double best_score = 0; uint64_t rng_state = 0;
    if (fread(&step, sizeof step, 1, f) != 1 ||
        fread(&best_score, sizeof best_score, 1, f) != 1 ||
        fread(&rng_state, sizeof rng_state, 1, f) != 1) die("short header");
    double mu[65], sd[65];
    if (fread(mu, sizeof(double), FEAT, f) != FEAT ||
        fread(sd, sizeof(double), FEAT, f) != FEAT) die("short mu/sd");

    /* Allocate parameter buffers. */
    double *W1 = malloc((size_t)FEAT*H1*sizeof(double));
    double *b1 = malloc(H1*sizeof(double));
    double *W2 = malloc((size_t)H1*H2*sizeof(double));
    double *b2 = malloc(H2*sizeof(double));
    double *W3 = malloc((size_t)H2*NCLASS*sizeof(double));
    double *b3 = malloc(NCLASS*sizeof(double));
    if (!W1||!b1||!W2||!b2||!W3||!b3) die("oom");

    if (fread(W1, sizeof(double), (size_t)FEAT*H1, f) != (size_t)FEAT*H1 ||
        fread(b1, sizeof(double), H1, f) != H1 ||
        fread(W2, sizeof(double), (size_t)H1*H2, f) != (size_t)H1*H2 ||
        fread(b2, sizeof(double), H2, f) != H2 ||
        fread(W3, sizeof(double), (size_t)H2*NCLASS, f) != (size_t)H2*NCLASS ||
        fread(b3, sizeof(double), NCLASS, f) != NCLASS) die("short weights");
    /* Skip momentum arrays. */
    fclose(f);

    /* Build Lancius graph. */
    lancius_graph* g = lancius_graph_create();
    if (!g) die("graph");
    lancius_node* Xn  = lancius_input(g, 1, FEAT);
    lancius_node* W1n = lancius_input(g, FEAT, H1);
    lancius_node* b1n = lancius_input(g, 1, H1);
    lancius_node* W2n = lancius_input(g, H1, H2);
    lancius_node* b2n = lancius_input(g, 1, H2);
    lancius_node* W3n = lancius_input(g, H2, NCLASS);
    lancius_node* b3n = lancius_input(g, 1, NCLASS);
    if (!Xn||!W1n||!b1n||!W2n||!b2n||!W3n||!b3n) die("builders");
    lancius_node_bind_external(W1n, W1);
    lancius_node_bind_external(b1n, b1);
    lancius_node_bind_external(W2n, W2);
    lancius_node_bind_external(b2n, b2);
    lancius_node_bind_external(W3n, W3);
    lancius_node_bind_external(b3n, b3);
    lancius_node* h1 = lancius_tanh(g, lancius_add(g, lancius_matmul(g, Xn, W1n), b1n));
    lancius_node* h2 = lancius_tanh(g, lancius_add(g, lancius_matmul(g, h1, W2n), b2n));
    lancius_node* z3 = lancius_add(g, lancius_matmul(g, h2, W3n), b3n);
    if (!h1 || !h2 || !z3) die("op builder");

    if (lancius_graph_save(g, out_path) != 0) {
        fprintf(stderr, "lancius_graph_save failed: %s\n",
                lancius_error_string(lancius_get_error()));
        lancius_graph_destroy(g);
        return -1;
    }
    printf("checkpoint %s -> %s\n", ckpt_path, out_path);
    printf("  step=%ld best_score=%.6f arch: 65->%zu->%zu->5\n",
           step, best_score, H1, H2);
    lancius_graph_destroy(g);
    free(W1); free(b1); free(W2); free(b2); free(W3); free(b3);
    return 0;
}
