/* Lancius OS: unified operator — CLI + ANSI TUI.
 *
 * Verbs:
 *   datasets | train | run | info | convert | export | generate | doctor | tui
 *
 * Design notes (binding):
 * - Generic framework operations only. No truth-algebra opcodes, no verifier
 *   doctrine in src/. Training wraps existing example binaries; inference
 *   uses the public IR/scheduler API directly.
 * - No ncurses dependency: ANSI escapes + stdin only (bare-metal friendly).
 * - Every verb fails loud (nonzero exit) on error; TUI loops instead.
 */
#include <lancius.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/stat.h>
#include <unistd.h>

static void print_version(void) {
    printf("lancius %s (public tag reserved for github releases)\n", LANCIUS_VERSION_STRING);
}

static void print_usage(void) {
    printf("lancius — Lancius runtime operator\n");
    printf("usage:\n");
    printf("  lancius doctor\n");
    printf("  lancius datasets list|pull <name|vision|math|logic|all>|status|distill [-- args]\n");
    printf("  lancius train mnist|cifar10|verifier [--dry]\n");
    printf("  lancius run <model.lancius> [--input f.bin] [--mode wave|static] [--fill random|zero|one] [--topk K] [--show N]\n");
    printf("  lancius info <model.lancius> [--nodes]\n");
    printf("  lancius convert onnx2lancius <in.onnx> <out.lancius>\n");
    printf("  lancius convert lancius2onnx <in.lancius> <out.onnx>\n");
    printf("  lancius export pytorch <in.lancius> <out.py> [--onnx out.onnx]\n");
    printf("  lancius generate\n");
    printf("  lancius tui\n");
}

static int file_exists(const char *p) {
    struct stat st;
    return p && stat(p, &st) == 0;
}

static long file_size(const char *p) {
    struct stat st;
    if (!p || stat(p, &st) != 0) return -1;
    return (long)st.st_size;
}

static int run_shell(const char *cmd) {
    if (!cmd) return 1;
    printf("+ %s\n", cmd);
    fflush(stdout);
    int rc = system(cmd);
    if (rc != 0) {
        printf("FAIL: command exited with status %d\n", rc);
        return 1;
    }
    return 0;
}

static const char *op_name(int op) {
    switch (op) {
        case 0: return "NOP";
        case 1: return "INPUT";
        case 2: return "CONST";
        case 3: return "ADD";
        case 4: return "SUB";
        case 5: return "MUL";
        case 6: return "MATMUL";
        case 7: return "RELU";
        case 8: return "SOFTMAX";
        case 9: return "SUM";
        case 10: return "BROADCAST";
        case 11: return "TRANSPOSE";
        case 12: return "RELU_BWD";
        case 13: return "SOFTMAX_BWD";
        case 14: return "SUM_AXIS0";
        case 15: return "SUM_AXIS1";
        case 16: return "CONV2D";
        case 17: return "MAXPOOL2D";
        case 18: return "FLATTEN";
        case 19: return "CONV2D_RELU_FUSED";
        case 20: return "CROSS_ENTROPY";
        case 21: return "CROSS_ENTROPY_BWD";
        case 22: return "PERMUTE";
        case 23: return "MATMUL_BATCHED";
        case 24: return "CONV2D_BWD";
        case 25: return "CONV2D_BWD_W";
        case 26: return "MAXPOOL2D_BWD";
        case 27: return "RESHAPE";
        case 28: return "EMBEDDING";
        case 29: return "LAYERNORM";
        case 30: return "GELU";
        case 31: return "ROPE";
        case 32: return "ATTENTION";
        case 33: return "KV_CACHE_READ";
        case 34: return "KV_CACHE_WRITE";
        case 35: return "RMSNORM";
        case 36: return "SWIGLU";
        case 37: return "GQA";
        case 38: return "TANH";
        case 39: return "TANH_BWD";
        case 40: return "MSE";
        case 41: return "MSE_BWD";
        default: return "?";
    }
}

/* ---------------- doctor ---------------- */

static int cmd_doctor(void) {
    int bad = 0;
    printf("== lancius doctor ==\n");
    printf("version: %s\n", LANCIUS_VERSION_STRING);
    const char *bins[] = {"./train_mnist", "./train_cifar10", "./train_verifier_head",
        "./run_edge", "./generate_text", "./distill_prm800k", NULL};
    for (int i = 0; bins[i]; i++) {
        printf("  %-24s %s\n", bins[i], file_exists(bins[i]) ? "OK" : "MISSING (make)");
        if (!file_exists(bins[i])) bad = 1;
    }
    const char *pys[] = {"manage_datasets.py", "onnx_to_lancius.py",
        "export_lancius_onnx.py", "export_lancius_pytorch.py", NULL};
    for (int i = 0; pys[i]; i++) {
        printf("  %-24s %s\n", pys[i], file_exists(pys[i]) ? "OK" : "MISSING");
        if (!file_exists(pys[i])) bad = 1;
    }
    int rc = system("python3 -c \"import onnx,numpy\" 2>/dev/null");
    printf("  %-24s %s\n", "python3+onnx+numpy", rc == 0 ? "OK" : "MISSING (pip install onnx numpy)");
    if (rc != 0) bad = 1;
    rc = system("python3 -c \"import torch\" 2>/dev/null");
    printf("  %-24s %s\n", "python3+torch", rc == 0 ? "OK (pytorch export available)" : "absent (pytorch export disabled)");
    const char *data[] = {"train-images-idx3-ubyte", "cifar-10-batches-bin",
        "data_text/prm800k_phase1_train.jsonl", NULL};
    for (int i = 0; data[i]; i++) {
        long sz = file_size(data[i]);
        if (sz >= 0) printf("  %-24s OK (%ld bytes)\n", data[i], sz);
        else printf("  %-24s absent (lancius datasets pull ...)\n", data[i]);
    }
    printf(bad ? "doctor: INCOMPLETE (see MISSING above)\n" : "doctor: READY\n");
    return bad ? 1 : 0;
}

/* ---------------- datasets ---------------- */

static int cmd_datasets(int argc, char **argv) {
    if (argc < 1) {
        printf("usage: lancius datasets list|pull <t>|status|distill [-- args]\n");
        return 2;
    }
    if (strcmp(argv[0], "list") == 0) {
        printf("vision (C-trainable): mnist cifar10\n");
        printf("math/logic (python-side, distill to vectors): gsm8k math prm800k svamp minif2f proofwriter ruletaker\n");
        printf("groups: vision math logic all\n");
        return 0;
    }
    if (strcmp(argv[0], "status") == 0) {
        const char *paths[] = {"train-images-idx3-ubyte", "t10k-images-idx3-ubyte",
            "cifar-10-batches-bin/data_batch_1.bin", "data_text/gsm8k_train.jsonl",
            "data_text/prm800k_phase1_train.jsonl", "data_text/svamp.json",
            "data_text/miniF2F-v1", NULL};
        for (int i = 0; paths[i]; i++) {
            long sz = file_size(paths[i]);
            if (sz >= 0) printf("  PRESENT %s (%ld bytes)\n", paths[i], sz);
            else printf("  ABSENT  %s\n", paths[i]);
        }
        return 0;
    }
    if (strcmp(argv[0], "pull") == 0) {
        if (argc < 2) {
            printf("usage: lancius datasets pull <name|vision|math|logic|all>\n");
            return 2;
        }
        char cmd[512];
        snprintf(cmd, sizeof(cmd), "python3 manage_datasets.py download %s", argv[1]);
        return run_shell(cmd);
    }
    if (strcmp(argv[0], "distill") == 0) {
        /* passthrough: lancius datasets distill -- --in ... --out ... */
        char cmd[2048];
        snprintf(cmd, sizeof(cmd), "./distill_prm800k --selftest");
        if (run_shell(cmd) != 0) return 1;
        if (argc > 1) {
            size_t off = 0;
            off += (size_t)snprintf(cmd + off, sizeof(cmd) - off, "./distill_prm800k");
            for (int i = 1; i < argc && off < sizeof(cmd) - 64; i++)
                off += (size_t)snprintf(cmd + off, sizeof(cmd) - off, " %s", argv[i]);
            return run_shell(cmd);
        }
        printf("distill selftest OK. Passthrough args forwarded to ./distill_prm800k.\n");
        return 0;
    }
    printf("unknown datasets subcommand '%s'\n", argv[0]);
    return 2;
}

/* ---------------- train ---------------- */

static int cmd_train(int argc, char **argv) {
    if (argc < 1) {
        printf("usage: lancius train mnist|cifar10|verifier [--dry]\n");
        return 2;
    }
    int dry = 0;
    for (int i = 1; i < argc; i++) if (strcmp(argv[i], "--dry") == 0) dry = 1;
    const char *bin = NULL;
    if (strcmp(argv[0], "mnist") == 0) bin = "./train_mnist";
    else if (strcmp(argv[0], "cifar10") == 0) bin = "./train_cifar10";
    else if (strcmp(argv[0], "verifier") == 0) bin = "./train_verifier_head";
    else {
        printf("unknown train target '%s' (mnist|cifar10|verifier)\n", argv[0]);
        return 2;
    }
    if (!file_exists(bin)) {
        printf("FAIL: %s missing — run `make` first.\n", bin);
        return 1;
    }
    if (dry) {
        printf("dry-run: would exec %s\n", bin);
        return 0;
    }
    return run_shell(bin);
}

/* ---------------- info ---------------- */

static int cmd_info(int argc, char **argv) {
    int show_nodes = 0;
    const char *model = NULL;
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--nodes") == 0) show_nodes = 1;
        else if (argv[i][0] != '-' && !model) model = argv[i];
    }
    if (!model) {
        printf("usage: lancius info <model.lancius> [--nodes]\n");
        return 2;
    }
    FILE *f = fopen(model, "rb");
    if (!f) {
        printf("FAIL: cannot open %s\n", model);
        return 1;
    }
    uint8_t hdr[48];
    size_t nr = fread(hdr, 1, 48, f);
    fclose(f);
    if (nr != 48) {
        printf("FAIL: %s too small (%zu bytes)\n", model, nr);
        return 1;
    }
    uint32_t magic, version, flags, ncount;
    uint32_t checksum;
    memcpy(&magic, hdr + 0, 4);
    memcpy(&version, hdr + 4, 4);
    memcpy(&flags, hdr + 8, 4);
    memcpy(&ncount, hdr + 12, 4);
    memcpy(&checksum, hdr + 40, 4);
    printf("model: %s (%ld bytes)\n", model, file_size(model));
    printf("  magic=0x%08X version=%u flags=%u nodes=%u crc=%08x\n",
        magic, version, flags, ncount, checksum);
    if (magic != 0x32434E41u) {
        printf("FAIL: bad magic (want 0x32434E41)\n");
        return 1;
    }
    lancius_graph *g = lancius_graph_load(model);
    if (!g) {
        printf("FAIL: loader rejected model (see stderr; CRC/integrity?)\n");
        return 1;
    }
    printf("  loader: OK (CRC/integrity verified), %u nodes resident\n", g->node_count);
    if (show_nodes) {
        for (uint32_t i = 0; i < g->node_count; i++) {
            lancius_node *n = g->nodes[i];
            size_t elems = 0;
            (void)lancius_node_elements_checked(n, &elems);
            printf("  id=%u op=%s ndim=%u shape=[%zu,%zu,%zu,%zu] elems=%zu dtype=%d in=[",
                n->id, op_name((int)n->op), n->ndim,
                n->shape[0], n->shape[1], n->shape[2], n->shape[3],
                elems, (int)n->dtype);
            for (uint32_t k = 0; k < n->input_count; k++)
                printf("%s%u", k ? "," : "", n->inputs[k]->id);
            printf("]%s\n", n->runtime_data ? " W" : "");
        }
    }
    lancius_graph_destroy(g);
    return 0;
}

/* ---------------- run (generic inference) ---------------- */

static int cmd_run(int argc, char **argv) {
    const char *model = NULL;
    const char *input_path = NULL;
    const char *mode = "wave";
    const char *fill = "random";
    int topk = 5;
    int show = 16;
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--input") == 0 && i + 1 < argc) input_path = argv[++i];
        else if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) mode = argv[++i];
        else if (strcmp(argv[i], "--fill") == 0 && i + 1 < argc) fill = argv[++i];
        else if (strcmp(argv[i], "--topk") == 0 && i + 1 < argc) topk = atoi(argv[++i]);
        else if (strcmp(argv[i], "--show") == 0 && i + 1 < argc) show = atoi(argv[++i]);
        else if (argv[i][0] != '-' && !model) model = argv[i];
        else {
            printf("unknown arg '%s'\n", argv[i]);
            return 2;
        }
    }
    if (!model) {
        printf("usage: lancius run <model.lancius> [--input f.bin] [--mode wave|static] [--fill random|zero|one] [--topk K] [--show N]\n");
        return 2;
    }
    if (topk < 1) topk = 1;
    if (show < 1) show = 1;
    lancius_graph *g = lancius_graph_load(model);
    if (!g) {
        printf("FAIL: cannot load %s\n", model);
        return 1;
    }
    lancius_schedule *sched = lancius_ir_schedule(g);
    if (!sched) {
        printf("FAIL: cannot schedule %s\n", model);
        lancius_graph_destroy(g);
        return 1;
    }

    /* feed inputs = INPUT nodes with no bound data (weights are pre-bound) */
    lancius_node *feeds[256];
    size_t feed_elems[256];
    int nfeeds = 0;
    for (uint32_t i = 0; i < g->node_count && nfeeds < 256; i++) {
        lancius_node *n = g->nodes[i];
        if (n->op == LANCIUS_OP_INPUT && n->runtime_data == NULL) {
            size_t e = 0;
            if (!lancius_node_elements_checked(n, &e) || e == 0) {
                printf("FAIL: bad feed shape id=%u\n", n->id);
                lancius_schedule_destroy(sched);
                lancius_graph_destroy(g);
                return 1;
            }
            feeds[nfeeds] = n;
            feed_elems[nfeeds] = e;
            nfeeds++;
        }
    }
    if (nfeeds == 0) {
        printf("note: no feed inputs (fully-weighted graph).\n");
    }

    double *owned[256] = {0};
    if (input_path && nfeeds != 1) {
        printf("FAIL: --input needs exactly 1 feed input, graph has %d\n", nfeeds);
        lancius_schedule_destroy(sched);
        lancius_graph_destroy(g);
        return 1;
    }
    srand(42);
    for (int i = 0; i < nfeeds; i++) {
        owned[i] = (double *)calloc(feed_elems[i], sizeof(double));
        if (!owned[i]) {
            printf("FAIL: OOM\n");
            for (int j = 0; j < i; j++) free(owned[j]);
            lancius_schedule_destroy(sched);
            lancius_graph_destroy(g);
            return 1;
        }
        if (input_path) {
            FILE *f = fopen(input_path, "rb");
            if (!f) {
                printf("FAIL: cannot open --input %s\n", input_path);
                for (int j = 0; j <= i; j++) free(owned[j]);
                lancius_schedule_destroy(sched);
                lancius_graph_destroy(g);
                return 1;
            }
            long sz = file_size(input_path);
            if (sz == (long)(feed_elems[i] * sizeof(double))) {
                size_t nr = fread(owned[i], 1, feed_elems[i] * sizeof(double), f);
                (void)nr;
            } else if (sz == (long)(feed_elems[i] * sizeof(float))) {
                float *tmp = (float *)malloc(feed_elems[i] * sizeof(float));
                if (!tmp) {
                    fclose(f);
                    printf("FAIL: OOM\n");
                    for (int j = 0; j <= i; j++) free(owned[j]);
                    lancius_schedule_destroy(sched);
                    lancius_graph_destroy(g);
                    return 1;
                }
                size_t nr = fread(tmp, 1, feed_elems[i] * sizeof(float), f);
                (void)nr;
                for (size_t k = 0; k < feed_elems[i]; k++) owned[i][k] = (double)tmp[k];
                free(tmp);
            } else {
                printf("FAIL: --input size %ld != elems %zu x8/x4\n", sz, feed_elems[i]);
                fclose(f);
                for (int j = 0; j <= i; j++) free(owned[j]);
                lancius_schedule_destroy(sched);
                lancius_graph_destroy(g);
                return 1;
            }
            fclose(f);
        } else if (strcmp(fill, "zero") == 0) {
            memset(owned[i], 0, feed_elems[i] * sizeof(double));
        } else if (strcmp(fill, "one") == 0) {
            for (size_t k = 0; k < feed_elems[i]; k++) owned[i][k] = 1.0;
        } else {
            for (size_t k = 0; k < feed_elems[i]; k++)
                owned[i][k] = ((double)rand() / (double)RAND_MAX) - 0.5;
        }
        lancius_node_bind_external(feeds[i], owned[i]);
    }

    lancius_arena *scratch_keep = NULL;
    void *static_buf = NULL;
    if (strcmp(mode, "static") == 0) {
        size_t need = lancius_schedule_static_memory_required(sched);
        if (need == 0) need = 1024 * 1024;
        static_buf = malloc(need);
        if (!static_buf) {
            printf("FAIL: OOM static buffer %zu\n", need);
            for (int j = 0; j < nfeeds; j++) free(owned[j]);
            lancius_schedule_destroy(sched);
            lancius_graph_destroy(g);
            return 1;
        }
        lancius_schedule_execute_static(sched, static_buf);
    } else if (strcmp(mode, "wave") == 0) {
        scratch_keep = lancius_arena_create(64 * 1024 * 1024);
        if (!scratch_keep) {
            printf("FAIL: OOM arena\n");
            for (int j = 0; j < nfeeds; j++) free(owned[j]);
            lancius_schedule_destroy(sched);
            lancius_graph_destroy(g);
            return 1;
        }
        lancius_schedule_execute(sched, scratch_keep);
    } else {
        printf("FAIL: unknown --mode '%s' (wave|static)\n", mode);
        for (int j = 0; j < nfeeds; j++) free(owned[j]);
        lancius_schedule_destroy(sched);
        lancius_graph_destroy(g);
        return 1;
    }

    /* output: CE-logits convention else last computed node */
    lancius_node *out = NULL;
    for (uint32_t i = 0; i < g->node_count; i++) {
        if (g->nodes[i]->op == LANCIUS_OP_CROSS_ENTROPY && g->nodes[i]->input_count > 0) {
            out = (lancius_node *)g->nodes[i]->inputs[0];
            break;
        }
    }
    if (!out) {
        for (uint32_t i = g->node_count; i > 0; i--) {
            lancius_node *n = g->nodes[i - 1];
            if (n->op != LANCIUS_OP_INPUT && n->op != LANCIUS_OP_CONST &&
                n->op != LANCIUS_OP_NOP && n->runtime_data) {
                out = n;
                break;
            }
        }
    }
    int rc = 0;
    if (!out || !out->runtime_data) {
        printf("FAIL: no output buffer (graph did not execute?)\n");
        rc = 1;
    } else {
        size_t e = 0;
        (void)lancius_node_elements_checked(out, &e);
        printf("output: id=%u op=%s shape=[%zu,%zu,%zu,%zu] elems=%zu\n",
            out->id, op_name((int)out->op),
            out->shape[0], out->shape[1], out->shape[2], out->shape[3], e);
        size_t lim = e < (size_t)show ? e : (size_t)show;
        printf("values[0..%zu]:", lim);
        for (size_t k = 0; k < lim; k++) printf(" %.4f", out->runtime_data[k]);
        printf("\n");
        if (out->ndim == 2 && out->shape[1] >= 2) {
            size_t R = out->shape[0], C = out->shape[1];
            size_t rlim = R < 4 ? R : 4;
            int *taken = (int *)calloc(C, sizeof(int));
            if (!taken) {
                printf("FAIL: OOM\n");
                rc = 1;
            } else {
                for (size_t r = 0; r < rlim; r++) {
                    memset(taken, 0, C * sizeof(int));
                    int best = 0;
                    for (size_t c = 1; c < C; c++)
                        if (out->runtime_data[r * C + c] > out->runtime_data[r * C + (size_t)best]) best = (int)c;
                    printf("row %zu: argmax=%d (%.4f) | top%d:", r, best,
                        out->runtime_data[r * C + (size_t)best], topk);
                    for (int t = 0; t < topk && t < (int)C; t++) {
                        int bi = -1;
                        double bv = -1e300;
                        for (size_t c = 0; c < C; c++) {
                            if (taken[c]) continue;
                            double v = out->runtime_data[r * C + c];
                            if (v > bv) {
                                bv = v;
                                bi = (int)c;
                            }
                        }
                        if (bi < 0) break;
                        taken[(size_t)bi] = 1;
                        printf(" %d:%.3f", bi, bv);
                    }
                    printf("\n");
                }
                free(taken);
            }
        }
    }

    for (int j = 0; j < nfeeds; j++) free(owned[j]);
    if (scratch_keep) lancius_arena_destroy(scratch_keep);
    if (static_buf) free(static_buf);
    lancius_schedule_destroy(sched);
    lancius_graph_destroy(g);
    return rc;
}

/* ---------------- convert / export ---------------- */

static int cmd_convert(int argc, char **argv) {
    if (argc < 1) {
        printf("usage: lancius convert onnx2lancius <in.onnx> <out.lancius> | lancius2onnx <in.lancius> <out.onnx>\n");
        return 2;
    }
    if (strcmp(argv[0], "onnx2lancius") == 0) {
        if (argc != 3) {
            printf("usage: lancius convert onnx2lancius <in.onnx> <out.lancius>\n");
            return 2;
        }
        char cmd[1024];
        snprintf(cmd, sizeof(cmd), "python3 onnx_to_lancius.py %s %s", argv[1], argv[2]);
        return run_shell(cmd);
    }
    if (strcmp(argv[0], "lancius2onnx") == 0) {
        if (argc != 3) {
            printf("usage: lancius convert lancius2onnx <in.lancius> <out.onnx>\n");
            return 2;
        }
        char cmd[1024];
        snprintf(cmd, sizeof(cmd), "python3 export_lancius_onnx.py %s %s", argv[1], argv[2]);
        return run_shell(cmd);
    }
    printf("unknown convert '%s'\n", argv[0]);
    return 2;
}

static int cmd_export(int argc, char **argv) {
    if (argc < 3 || strcmp(argv[0], "pytorch") != 0) {
        printf("usage: lancius export pytorch <in.lancius> <out.py> [--onnx out.onnx]\n");
        return 2;
    }
    char cmd[2048];
    size_t off = (size_t)snprintf(cmd, sizeof(cmd), "python3 export_lancius_pytorch.py %s %s", argv[1], argv[2]);
    for (int i = 3; i < argc && off < sizeof(cmd) - 64; i++)
        off += (size_t)snprintf(cmd + off, sizeof(cmd) - off, " %s", argv[i]);
    return run_shell(cmd);
}

/* ---------------- TUI ---------------- */

static void tui_line(void) {
    printf("------------------------------------------------------------\n");
}

static int tui_prompt(const char *label, char *buf, size_t cap) {
    printf("%s: ", label);
    fflush(stdout);
    if (!fgets(buf, (int)cap, stdin)) return 0;
    buf[strcspn(buf, "\r\n")] = 0;
    return 1;
}

static int cmd_tui(void) {
    char buf[256], a[256], b[256], c[256];
    printf("\033[2J\033[H");
    printf("== lancius operator (TUI) %s ==\n", LANCIUS_VERSION_STRING);
    for (;;) {
        tui_line();
        printf(" 1 datasets pull   2 datasets status   3 train\n");
        printf(" 4 run model       5 info model        6 onnx->lancius\n");
        printf(" 7 lancius->onnx   8 lancius->pytorch  9 doctor\n");
        printf(" 10 generate demo  0 quit\n");
        tui_line();
        if (!tui_prompt("select", buf, sizeof(buf))) break;
        if (strcmp(buf, "0") == 0 || strcmp(buf, "q") == 0) break;
        else if (strcmp(buf, "1") == 0) {
            if (!tui_prompt("target (mnist|cifar10|vision|math|logic|all)", a, sizeof(a))) continue;
            char cmd[1024];
            snprintf(cmd, sizeof(cmd), "./lancius datasets pull %s", a);
            (void)run_shell(cmd);
        } else if (strcmp(buf, "2") == 0) {
            (void)run_shell("./lancius datasets status");
        } else if (strcmp(buf, "3") == 0) {
            if (!tui_prompt("target (mnist|cifar10|verifier)", a, sizeof(a))) continue;
            char cmd[1024];
            snprintf(cmd, sizeof(cmd), "./lancius train %s", a);
            (void)run_shell(cmd);
        } else if (strcmp(buf, "4") == 0) {
            if (!tui_prompt("model path", a, sizeof(a))) continue;
            if (!tui_prompt("mode (wave|static) [wave]", b, sizeof(b))) continue;
            if (!b[0]) strcpy(b, "wave");
            char cmd[2048];
            snprintf(cmd, sizeof(cmd), "./lancius run %s --mode %s", a, b);
            (void)run_shell(cmd);
        } else if (strcmp(buf, "5") == 0) {
            if (!tui_prompt("model path", a, sizeof(a))) continue;
            char cmd[1024];
            snprintf(cmd, sizeof(cmd), "./lancius info %s --nodes", a);
            (void)run_shell(cmd);
        } else if (strcmp(buf, "6") == 0) {
            if (!tui_prompt("in.onnx", a, sizeof(a))) continue;
            if (!tui_prompt("out.lancius", b, sizeof(b))) continue;
            char cmd[2048];
            snprintf(cmd, sizeof(cmd), "./lancius convert onnx2lancius %s %s", a, b);
            (void)run_shell(cmd);
        } else if (strcmp(buf, "7") == 0) {
            if (!tui_prompt("in.lancius", a, sizeof(a))) continue;
            if (!tui_prompt("out.onnx", b, sizeof(b))) continue;
            char cmd[2048];
            snprintf(cmd, sizeof(cmd), "./lancius convert lancius2onnx %s %s", a, b);
            (void)run_shell(cmd);
        } else if (strcmp(buf, "8") == 0) {
            if (!tui_prompt("in.lancius", a, sizeof(a))) continue;
            if (!tui_prompt("out.py", b, sizeof(b))) continue;
            if (!tui_prompt("also onnx path (empty=skip)", c, sizeof(c))) continue;
            char cmd[2048];
            if (c[0]) snprintf(cmd, sizeof(cmd), "./lancius export pytorch %s %s --onnx %s", a, b, c);
            else snprintf(cmd, sizeof(cmd), "./lancius export pytorch %s %s", a, b);
            (void)run_shell(cmd);
        } else if (strcmp(buf, "9") == 0) {
            (void)run_shell("./lancius doctor");
        } else if (strcmp(buf, "10") == 0) {
            (void)run_shell("./generate_text");
        } else {
            printf("unknown selection '%s'\n", buf);
        }
    }
    printf("bye.\n");
    return 0;
}

/* ---------------- main ---------------- */

int main(int argc, char **argv) {
    if (argc < 2) {
        print_usage();
        return 2;
    }
    if (strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "help") == 0) {
        print_usage();
        return 0;
    }
    if (strcmp(argv[1], "--version") == 0 || strcmp(argv[1], "version") == 0) {
        print_version();
        return 0;
    }
    if (strcmp(argv[1], "doctor") == 0) return cmd_doctor();
    if (strcmp(argv[1], "datasets") == 0) return cmd_datasets(argc - 2, argv + 2);
    if (strcmp(argv[1], "train") == 0) return cmd_train(argc - 2, argv + 2);
    if (strcmp(argv[1], "run") == 0) return cmd_run(argc - 2, argv + 2);
    if (strcmp(argv[1], "info") == 0) return cmd_info(argc - 2, argv + 2);
    if (strcmp(argv[1], "convert") == 0) return cmd_convert(argc - 2, argv + 2);
    if (strcmp(argv[1], "export") == 0) return cmd_export(argc - 2, argv + 2);
    if (strcmp(argv[1], "generate") == 0) {
        if (!file_exists("./generate_text")) {
            printf("FAIL: ./generate_text missing — run `make`.\n");
            return 1;
        }
        return run_shell("./generate_text");
    }
    if (strcmp(argv[1], "tui") == 0) return cmd_tui();
    printf("unknown verb '%s'\n", argv[1]);
    print_usage();
    return 2;
}
