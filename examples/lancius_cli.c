/* Lancius OS: unified operator — CLI + ANSI TUI.
 *
 * Verbs:
 *   help | doctor | status | models | demo | datasets | train | run | info |
 *   convert | export | generate | tui
 *
 * Design notes (binding):
 * - Generic framework operations only. No truth-algebra opcodes, no verifier
 *   doctrine in src/. Training wraps existing example binaries; inference
 *   uses the public IR/scheduler API directly.
 * - No ncurses dependency: ANSI escapes + stdin only (bare-metal friendly).
 * - Every verb fails loud (nonzero exit) on error; TUI loops instead.
 * - Ordinary-user contract: every verb has --help with examples, inputs are
 *   validated before work starts, offline use fails with guidance instead of
 *   a traceback, and `demo` proves the install in one command.
 */
#define _POSIX_C_SOURCE 200112L
#include <lancius.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <ctype.h>
#include <math.h>
#include <dirent.h>
#include <errno.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

static void print_version(void) {
    printf("lancius %s (github tag: %s)\n", LANCIUS_VERSION_STRING, LANCIUS_VERSION_PUBLIC);
}

static void print_usage(void) {
    printf("lancius %s — bare-metal ML inference runtime\n", LANCIUS_VERSION_STRING);
    printf("\n");
    printf("welcome! new here? run these in order:\n");
    printf("  1. lancius doctor              check everything is ready\n");
    printf("  2. lancius demo                prove the install in one command\n");
    printf("  3. lancius tui                 guided menus, nothing to memorize\n");
    printf("\n");
    printf("usage:\n");
    printf("  lancius quickstart            guided setup for new users\n");
    printf("  lancius help [verb]           show help (also: lancius <verb> --help)\n");
    printf("  lancius doctor              readiness check (binaries, python, data, network, disk)\n");
    printf("  lancius status              operational snapshot (version, cwd, models, data, disk, net)\n");
    printf("  lancius models [--check]    list .lancius files here (--check fully validates each)\n");
    printf("  lancius demo                one-command install proof (info + run on a local model)\n");
    printf("  lancius datasets list|pull <name|vision|math|logic|all>|status|distill [-- args]\n");
    printf("  lancius train mnist|cifar10|verifier [--dry]\n");
    printf("  lancius run <model.lancius> [--input f.bin] [--mode wave|static] [--fill random|zero|one] [--topk K] [--show N]\n");
    printf("  lancius eval <model.lancius> [--mode wave|static]   run vendored micromodel end to end (R2-6 gate)\n");
    printf("  lancius info <model.lancius> [--nodes]\n");
    printf("  lancius convert onnx2lancius <in.onnx> <out.lancius>\n");
    printf("  lancius convert lancius2onnx <in.lancius> <out.onnx>\n");
    printf("  lancius export pytorch <in.lancius> <out.py> [--onnx out.onnx]\n");
    printf("  lancius generate            transformer prefill/generation demo\n");
    printf("  lancius tui                 guided menus for everything above\n");
    printf("\n");
    printf("examples:\n");
    printf("  lancius quickstart\n");
    printf("  lancius info test_model.lancius --nodes\n");
    printf("  lancius run test_model.lancius --mode static --fill zero\n");
    printf("  lancius datasets pull vision && lancius train verifier\n");
}

static int cmd_quickstart(void);

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

/* Despot truth: system() with user paths was command injection
 * (filenames with ; $() `` inject). User-controlled args always go through
 * fork+execvp with no shell. Fixed commands (no user input) keep system(). */
static int run_argv(const char *prog, char *const argv[]) {
    pid_t pid;
    int st = 0;
    size_t i;
    if (!prog || !argv || !argv[0]) return 1;
    printf("+");
    for (i = 0; argv[i]; i++) printf(" %s", argv[i]);
    printf("\n");
    fflush(stdout);
    pid = fork();
    if (pid < 0) {
        printf("FAIL: fork failed (%s)\n", strerror(errno));
        return 1;
    }
    if (pid == 0) {
        execvp(prog, argv);
        _exit(127);
    }
    while (waitpid(pid, &st, 0) < 0 && errno == EINTR) {}
    if (!WIFEXITED(st) || WEXITSTATUS(st) != 0) {
        printf("FAIL: command exited with status %d\n",
            WIFEXITED(st) ? WEXITSTATUS(st) : -1);
        return 1;
    }
    return 0;
}

/* ---------------- small UX helpers ---------------- */

static void trim_inplace(char *s) {
    size_t n;
    size_t i;
    if (!s) return;
    n = strlen(s);
    while (n > 0 && isspace((unsigned char)s[n - 1])) s[--n] = 0;
    i = 0;
    while (s[i] && isspace((unsigned char)s[i])) i++;
    if (i > 0) memmove(s, s + i, strlen(s + i) + 1);
}

static int is_tty_stdin(void) {
    return isatty(STDIN_FILENO);
}

static int use_color(void) {
    const char *no;
    const char *term;
    if (!is_tty_stdin()) return 0;
    no = getenv("NO_COLOR");
    if (no && no[0]) return 0;
    term = getenv("TERM");
    if (term && strcmp(term, "dumb") == 0) return 0;
    return 1;
}

static const char *c_ok(void) { return use_color() ? "\033[32m" : ""; }
static const char *c_bad(void) { return use_color() ? "\033[31m" : ""; }
static const char *c_warn(void) { return use_color() ? "\033[33m" : ""; }
static const char *c_off(void) { return use_color() ? "\033[0m" : ""; }

/* Connectivity probe for ordinary users: 1 online, 0 offline, -1 unknown
 * (curl missing). Never hangs: --max-time 5. No dependencies otherwise. */
static int net_online(void) {
    int rc;
    if (system("command -v curl >/dev/null 2>&1") != 0) return -1;
    rc = system("curl -sI --max-time 5 https://example.com >/dev/null 2>&1");
    return rc == 0 ? 1 : 0;
}

static const char *net_word(int v) {
    if (v == 1) return "online";
    if (v == 0) return "offline";
    return "unknown (curl missing)";
}

static long disk_free_mb(const char *path) {
    struct statvfs sv;
    if (!path) path = ".";
    if (statvfs(path, &sv) != 0) return -1;
    if (sv.f_frsize == 0) return -1;
    {
        unsigned long long free_b = (unsigned long long)sv.f_bavail * (unsigned long long)sv.f_frsize;
        return (long)(free_b / (1024ULL * 1024ULL));
    }
}

/* List *.lancius in cwd. Prints name + bytes. Returns count, -1 on opendir fail. */
static int list_model_files(void) {
    DIR *d = opendir(".");
    struct dirent *e;
    int n = 0;
    if (!d) {
        printf("FAIL: cannot list working directory\n");
        return -1;
    }
    while ((e = readdir(d)) != NULL) {
        const char *nm = e->d_name;
        size_t L = strlen(nm);
        if (L < 8 || strcmp(nm + L - 8, ".lancius") != 0) continue;
        printf("  %-32s %ld bytes\n", nm, file_size(nm));
        n++;
    }
    closedir(d);
    if (n == 0) printf("  (no .lancius files in this directory)\n");
    return n;
}

/* First .lancius in cwd (prefer test_model.lancius). Returns 1 if found. */
static int pick_demo_model(char *out, size_t cap) {
    DIR *d;
    struct dirent *e;
    char first[512] = {0};
    if (!out || cap == 0) return 0;
    if (file_exists("test_model.lancius")) {
        snprintf(out, cap, "test_model.lancius");
        return 1;
    }
    d = opendir(".");
    if (!d) return 0;
    while ((e = readdir(d)) != NULL) {
        const char *nm = e->d_name;
        size_t L = strlen(nm);
        if (L < 8 || strcmp(nm + L - 8, ".lancius") != 0) continue;
        if (!first[0]) snprintf(first, sizeof(first), "%s", nm);
    }
    closedir(d);
    if (!first[0]) return 0;
    snprintf(out, cap, "%s", first);
    return 1;
}

static int is_help_arg(const char *a) {
    return a && (strcmp(a, "--help") == 0 || strcmp(a, "-h") == 0 || strcmp(a, "help") == 0);
}

/* Forward declarations: demo/TUI call verbs defined later in this TU. */
static int cmd_tui(void);
static int tui_read(char *buf, size_t cap);
static int cmd_datasets(int argc, char **argv);
static int cmd_info(int argc, char **argv);
static int cmd_run(int argc, char **argv);
static int cmd_eval(int argc, char **argv);
static int cmd_train(int argc, char **argv);
static int cmd_convert(int argc, char **argv);
static int cmd_export(int argc, char **argv);
static int cmd_doctor(void);
static int cmd_status(void);
static int cmd_models(int argc, char **argv);
static int cmd_demo(void);

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

static const char *dtype_name(int dt) {
    switch (dt) {
        case 0: return "FP64";
        case 1: return "INT8";
        case 2: return "FP32";
        case 3: return "INT32";
        default: return "?";
    }
}

static int is_backward_op(int op) {
    return op == 12 || op == 13 || op == 21 || op == 24 ||
           op == 25 || op == 26 || op == 39 || op == 41;
}

static int is_reserved_op(int op) {
    return op == 28 || op == 33 || op == 34;
}

/* ---------------- doctor ---------------- */

static int cmd_doctor(void) {
    int core_bad = 0;
    int data_missing = 0;
    int net;
    long dfree;
    printf("== lancius doctor ==\n");
    printf("version: %s\n", LANCIUS_VERSION_STRING);
    printf("-- binaries (run `make` if MISSING) --\n");
    {
        const char *bins[] = {"./train_mnist", "./train_cifar10", "./train_verifier_head",
            "./run_edge", "./generate_text", "./distill_prm800k", NULL};
        int i;
        for (i = 0; bins[i]; i++) {
            int ok = file_exists(bins[i]);
            printf("  %-24s %s%s%s\n", bins[i],
                ok ? c_ok() : c_bad(), ok ? "OK" : "MISSING (make)", c_off());
            if (!ok) core_bad = 1;
        }
    }
    printf("-- python interop (pip install onnx numpy for conversion) --\n");
    {
        const char *pys[] = {"manage_datasets.py", "onnx_to_lancius.py",
            "export_lancius_onnx.py", "export_lancius_pytorch.py", NULL};
        int i;
        for (i = 0; pys[i]; i++) {
            int ok = file_exists(pys[i]);
            printf("  %-24s %s%s%s\n", pys[i],
                ok ? c_ok() : c_bad(), ok ? "OK" : "MISSING", c_off());
            if (!ok) core_bad = 1;
        }
    }
    {
        int rc = system("python3 -c \"import onnx,numpy\" 2>/dev/null");
        printf("  %-24s %s%s%s\n", "python3+onnx+numpy", rc == 0 ? c_ok() : c_bad(),
            rc == 0 ? "OK" : "MISSING (pip install onnx numpy)", c_off());
        if (rc != 0) core_bad = 1;
        rc = system("python3 -c \"import torch\" 2>/dev/null");
        printf("  %-24s %s\n", "python3+torch",
            rc == 0 ? "OK (pytorch export available)" : "absent (pytorch export disabled, optional)");
    }
    printf("-- training data (optional for inference; needed for train) --\n");
    {
        const char *data[] = {"train-images-idx3-ubyte", "cifar-10-batches-bin",
            "data_text/prm800k_phase1_train.jsonl", NULL};
        int i;
        for (i = 0; data[i]; i++) {
            long sz = file_size(data[i]);
            if (sz >= 0) printf("  %-24s %sOK (%ld bytes)%s\n", data[i], c_ok(), sz, c_off());
            else {
                printf("  %-24s %sabsent (lancius datasets pull ...)%s\n", data[i], c_warn(), c_off());
                data_missing = 1;
            }
        }
    }
    printf("-- network + disk --\n");
    net = net_online();
    printf("  %-24s %s%s%s\n", "network", net == 1 ? c_ok() : (net == 0 ? c_warn() : ""),
        net_word(net), c_off());
    if (net == 0) printf("  hint: datasets pull needs network; cached data still works offline\n");
    dfree = disk_free_mb(".");
    if (dfree >= 0) printf("  %-24s %ld MB free\n", "disk(.)", dfree);
    else printf("  %-24s unknown\n", "disk(.)");
    if (core_bad) {
        printf("doctor: %sINCOMPLETE%s (see MISSING above)\n", c_bad(), c_off());
        return 1;
    }
    if (data_missing)
        printf("doctor: %sREADY%s for inference and quick demos; training data absent (optional, see pull hint)\n",
            c_ok(), c_off());
    else
        printf("doctor: %sREADY%s\n", c_ok(), c_off());
    return 0;
}

/* ---------------- status / models / demo ---------------- */

static int cmd_status(void) {
    char cwd[1024] = {0};
    int net;
    long dfree;
    printf("== lancius status ==\n");
    printf("version: %s\n", LANCIUS_VERSION_STRING);
    if (getcwd(cwd, sizeof(cwd)) != NULL) printf("cwd: %s\n", cwd);
    {
        const char *bins[] = {"./train_mnist", "./train_cifar10", "./train_verifier_head",
            "./run_edge", "./generate_text", "./distill_prm800k", NULL};
        int i, okc = 0, tot = 0;
        for (i = 0; bins[i]; i++) { tot++; if (file_exists(bins[i])) okc++; }
        printf("binaries: %d/%d present%s\n", okc, tot, okc == tot ? "" : " (run `make`)");
    }
    {
        int rc = system("python3 -c \"import onnx,numpy\" 2>/dev/null");
        printf("python onnx/numpy: %s\n", rc == 0 ? "OK" : "MISSING (pip install onnx numpy)");
    }
    {
        const char *mn = "train-images-idx3-ubyte";
        const char *cf = "cifar-10-batches-bin/data_batch_1.bin";
        const char *pm = "data_text/prm800k_phase1_train.jsonl";
        printf("data: mnist %s, cifar10 %s, prm800k %s\n",
            file_exists(mn) ? "present" : "absent",
            file_exists(cf) ? "present" : "absent",
            file_exists(pm) ? "present" : "absent");
    }
    printf("models in cwd:\n");
    (void)list_model_files();
    net = net_online();
    printf("network: %s\n", net_word(net));
    dfree = disk_free_mb(".");
    if (dfree >= 0) printf("disk free (.): %ld MB\n", dfree);
    else printf("disk free (.): unknown\n");
    printf("next: lancius demo (one-command proof) | lancius tui (guided menus)\n");
    return 0;
}

static int cmd_models(int argc, char **argv) {
    int check = 0;
    int i, n, bad = 0;
    DIR *d;
    struct dirent *e;
    for (i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--check") == 0) check = 1;
        else if (is_help_arg(argv[i])) {
            printf("usage: lancius models [--check]\n");
            printf("  lists .lancius files in the working directory.\n");
            printf("  --check  fully loads each file (validates CRC/integrity).\n");
            return 0;
        } else {
            printf("unknown arg '%s' (usage: lancius models [--check])\n", argv[i]);
            return 2;
        }
    }
    d = opendir(".");
    if (!d) {
        printf("FAIL: cannot list working directory\n");
        return 1;
    }
    n = 0;
    while ((e = readdir(d)) != NULL) {
        const char *nm = e->d_name;
        size_t L = strlen(nm);
        uint8_t hdr[48];
        FILE *f;
        uint32_t magic = 0, version = 0, ncount = 0, crc = 0;
        if (L < 8 || strcmp(nm + L - 8, ".lancius") != 0) continue;
        n++;
        f = fopen(nm, "rb");
        if (!f) {
            printf("  %-32s unreadable\n", nm);
            bad = 1;
            continue;
        }
        if (fread(hdr, 1, 48, f) != 48) {
            printf("  %-32s too small (%ld bytes)\n", nm, file_size(nm));
            fclose(f);
            bad = 1;
            continue;
        }
        fclose(f);
        memcpy(&magic, hdr + 0, 4);
        memcpy(&version, hdr + 4, 4);
        memcpy(&ncount, hdr + 12, 4);
        memcpy(&crc, hdr + 40, 4);
        if (magic != 0x32434E41u) {
            printf("  %-32s BAD MAGIC 0x%08X (want 0x32434E41)\n", nm, magic);
            bad = 1;
            continue;
        }
        printf("  %-32s %ld bytes v%u nodes=%u crc=%08x\n",
            nm, file_size(nm), version, ncount, crc);
        if (check) {
            lancius_graph *g = lancius_graph_load(nm);
            if (!g) {
                printf("    load: FAIL (rejected; see stderr)\n");
                bad = 1;
            } else {
                printf("    load: OK (%u nodes resident)\n", g->node_count);
                lancius_graph_destroy(g);
            }
        }
    }
    closedir(d);
    if (n == 0) {
        printf("(no .lancius files here; try lancius datasets pull vision, then train)\n");
        return 0;
    }
    if (check && bad) {
        printf("models --check: %d file(s) failed validation\n", bad);
        return 1;
    }
    return 0;
}

static int cmd_demo(void) {
    char model[512];
    int rc;
    printf("== lancius demo: one-command install proof ==\n");
    if (!pick_demo_model(model, sizeof(model))) {
        printf("FAIL: no .lancius files in this directory\n");
        printf("next: lancius datasets pull vision && lancius train verifier\n");
        return 1;
    }
    printf("-- step 1/2: info %s --\n", model);
    {
        char *av[1];
        av[0] = model;
        rc = cmd_info(1, av);
    }
    if (rc != 0) {
        printf("demo: FAIL (info step failed)\n");
        return 1;
    }
    printf("-- step 2/2: run %s (static, zero fill) --\n", model);
    {
        char *av[7];
        av[0] = model; av[1] = "--mode"; av[2] = "static"; av[3] = "--fill";
        av[4] = "zero"; av[5] = "--show"; av[6] = "8";
        rc = cmd_run(7, av);
    }
    if (rc != 0) {
        printf("demo: FAIL (run step failed)\n");
        return 1;
    }
    printf("demo: OK — install proven. next: lancius tui | lancius train verifier\n");
    return 0;
}

/* ---------------- quickstart ---------------- */

static int cmd_quickstart(void) {
    char buf[256];
    int step = 1;
    printf("== lancius quickstart: guided setup ==\n");
    printf("this walks you through the basics. press Enter to continue, 0 to skip.\n");
    printf("\n");

    /* Step 1: doctor */
    printf("[step %d/4] checking your install...\n", step++);
    if (!tui_read(buf, sizeof(buf))) return 0;
    if (buf[0] == '0') return 0;
    cmd_doctor();
    printf("\n");

    /* Step 2: demo */
    printf("[step %d/4] proving the install with a demo run...\n", step++);
    if (!tui_read(buf, sizeof(buf))) return 0;
    if (buf[0] == '0') return 0;
    cmd_demo();
    printf("\n");

    /* Step 3: train */
    printf("[step %d/4] training a small model (verifier, seconds)...\n", step++);
    printf("this trains a tiny model to verify training works.\n");
    if (!tui_read(buf, sizeof(buf))) return 0;
    if (buf[0] == '0') return 0;
    {
        char *av[1];
        av[0] = "verifier";
        cmd_train(1, av);
    }
    printf("\n");

    /* Step 4: TUI */
    printf("[step %d/4] exploring the TUI...\n", step++);
    printf("the TUI has guided menus for everything.\n");
    if (!tui_read(buf, sizeof(buf))) return 0;
    if (buf[0] == '0') return 0;
    cmd_tui();

    printf("\nquickstart complete! you now know the basics.\n");
    printf("next steps:\n");
    printf("  lancius datasets pull vision   # get training data\n");
    printf("  lancius train mnist           # train on MNIST (minutes)\n");
    printf("  lancius tui                   # explore everything\n");
    return 0;
}

/* ---------------- datasets ---------------- */

static void print_datasets_help(void) {
    printf("usage: lancius datasets list|pull <t>|status|distill [-- args]\n");
    printf("  list    show dataset names and groups (vision: C-trainable; math/logic: python-side)\n");
    printf("  pull t  download t (mnist|cifar10|vision|gsm8k|math|prm800k|svamp|minif2f|proofwriter|ruletaker|logic|all)\n");
    printf("          needs network; cached files are skipped, partial downloads resume cleanly\n");
    printf("  status  show which datasets are present locally\n");
    printf("  distill run the PRM800k selftest, or forward args to ./distill_prm800k\n");
    printf("examples:\n");
    printf("  lancius datasets pull vision\n");
    printf("  lancius datasets pull prm800k\n");
    printf("  lancius datasets status\n");
}

static int valid_pull_target(const char *t) {
    static const char *ok[] = {"mnist", "cifar10", "vision", "gsm8k", "math",
        "prm800k", "svamp", "minif2f", "proofwriter", "ruletaker",
        "logic", "all", NULL};
    int i;
    if (!t) return 0;
    for (i = 0; ok[i]; i++) if (strcmp(t, ok[i]) == 0) return 1;
    return 0;
}

static int cmd_datasets(int argc, char **argv) {
    if (argc < 1) {
        print_datasets_help();
        return 2;
    }
    if (is_help_arg(argv[0])) {
        print_datasets_help();
        return 0;
    }
    if (strcmp(argv[0], "list") == 0) {
        printf("vision (C-trainable): mnist cifar10\n");
        printf("  mnist:   70k 28x28 digits, minutes to train (good first train)\n");
        printf("  cifar10: 60k 32x32 color images, long train (30 epochs, hours)\n");
        printf("math/logic (python-side, distill to vectors): gsm8k math prm800k svamp minif2f proofwriter ruletaker\n");
        printf("groups: vision math logic all\n");
        return 0;
    }
    if (strcmp(argv[0], "status") == 0) {
        const char *paths[] = {"train-images-idx3-ubyte", "t10k-images-idx3-ubyte",
            "cifar-10-batches-bin/data_batch_1.bin", "data_text/gsm8k_train.jsonl",
            "data_text/prm800k_phase1_train.jsonl", "data_text/svamp.json",
            "data_text/miniF2F-v1", NULL};
        int i;
        for (i = 0; paths[i]; i++) {
            long sz = file_size(paths[i]);
            if (sz >= 0) printf("  PRESENT %s (%ld bytes)\n", paths[i], sz);
            else printf("  ABSENT  %s\n", paths[i]);
        }
        printf("hint: lancius datasets pull vision  (mnist+cifar10)\n");
        return 0;
    }
    if (strcmp(argv[0], "pull") == 0) {
        int net;
        if (argc < 2) {
            printf("usage: lancius datasets pull <name|vision|math|logic|all>\n");
            return 2;
        }
        if (!valid_pull_target(argv[1])) {
            printf("unknown dataset '%s'\n", argv[1]);
            printf("try: mnist cifar10 vision gsm8k math prm800k svamp minif2f proofwriter ruletaker logic all\n");
            return 2;
        }
        if (!file_exists("manage_datasets.py")) {
            printf("FAIL: manage_datasets.py missing in this directory\n");
            return 1;
        }
        net = net_online();
        if (net == 0) {
            printf("network: offline — pull needs network.\n");
            printf("if the data is already cached, training still works offline; else retry online.\n");
            printf("hint: lancius datasets status  (see what is cached)\n");
            return 1;
        }
        {
            char *xa[5];
            xa[0] = "python3"; xa[1] = "manage_datasets.py";
            xa[2] = "download"; xa[3] = argv[1]; xa[4] = NULL;
            return run_argv(xa[0], xa);
        }
    }
    if (strcmp(argv[0], "distill") == 0) {
        /* passthrough: lancius datasets distill -- --in ... --out ... */
        if (!file_exists("./distill_prm800k")) {
            printf("FAIL: ./distill_prm800k missing — run `make` first.\n");
            return 1;
        }
        {
            char *selftest[3];
            selftest[0] = "./distill_prm800k";
            selftest[1] = "--selftest";
            selftest[2] = NULL;
            if (run_argv(selftest[0], selftest) != 0) return 1;
        }
        if (argc > 1) {
            /* Despot truth: overlong args truncated then RAN (was: ignored
             * snprintf return). Fail instead; exec (no shell) kills injection. */
            char *xa[256];
            int i, ac = 0;
            if (argc - 1 > 250) {
                printf("FAIL: too many distill args (%d)\n", argc - 1);
                return 1;
            }
            xa[ac++] = "./distill_prm800k";
            for (i = 1; i < argc; i++) {
                if (strlen(argv[i]) > 1024) {
                    printf("FAIL: distill arg too long\n");
                    return 1;
                }
                xa[ac++] = argv[i];
            }
            xa[ac] = NULL;
            return run_argv(xa[0], xa);
        }
        printf("distill selftest OK. Passthrough args forwarded to ./distill_prm800k.\n");
        return 0;
    }
    printf("unknown datasets subcommand '%s'\n", argv[0]);
    print_datasets_help();
    return 2;
}

/* ---------------- train ---------------- */

static void print_train_help(void) {
    printf("usage: lancius train mnist|cifar10|verifier [--dry]\n");
    printf("  mnist     10 epochs, minutes, needs MNIST data (pull vision)\n");
    printf("  cifar10   30 epochs, hours, needs CIFAR-10 data (pull vision)\n");
    printf("  verifier  ~300 iters, seconds, no download needed (best first train)\n");
    printf("  --dry     show what would run (binary, data check, time hint) without training\n");
    printf("examples:\n");
    printf("  lancius train verifier --dry\n");
    printf("  lancius train verifier\n");
    printf("  lancius datasets pull vision && lancius train mnist\n");
}

static int train_data_ok(const char *target, char *hint, size_t hintcap) {
    if (strcmp(target, "mnist") == 0) {
        if (!file_exists("train-images-idx3-ubyte")) {
            snprintf(hint, hintcap, "lancius datasets pull mnist  (or vision)");
            return 0;
        }
        return 1;
    }
    if (strcmp(target, "cifar10") == 0) {
        if (!file_exists("cifar-10-batches-bin/data_batch_1.bin")) {
            snprintf(hint, hintcap, "lancius datasets pull cifar10  (or vision)");
            return 0;
        }
        return 1;
    }
    return 1;
}

static int cmd_train(int argc, char **argv) {
    const char *bin = NULL;
    const char *est = "";
    int dry = 0;
    int i;
    char hint[256] = {0};
    if (argc < 1) {
        print_train_help();
        return 2;
    }
    if (is_help_arg(argv[0])) {
        print_train_help();
        return 0;
    }
    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--dry") == 0) dry = 1;
        else if (is_help_arg(argv[i])) {
            print_train_help();
            return 0;
        } else {
            printf("unknown arg '%s' (usage: lancius train mnist|cifar10|verifier [--dry])\n", argv[i]);
            return 2;
        }
    }
    if (strcmp(argv[0], "mnist") == 0) {
        bin = "./train_mnist";
        est = "10 epochs, minutes";
    } else if (strcmp(argv[0], "cifar10") == 0) {
        bin = "./train_cifar10";
        est = "30 epochs, hours — long; try verifier first";
    } else if (strcmp(argv[0], "verifier") == 0) {
        bin = "./train_verifier_head";
        est = "~300 iters, seconds";
    } else {
        printf("unknown train target '%s' (mnist|cifar10|verifier)\n", argv[0]);
        return 2;
    }
    if (!file_exists(bin)) {
        printf("FAIL: %s missing — run `make` first.\n", bin);
        return 1;
    }
    if (!train_data_ok(argv[0], hint, sizeof(hint))) {
        int net = net_online();
        printf("training data missing for '%s'.\n", argv[0]);
        printf("fix: %s\n", hint);
        if (net == 0) printf("network: offline — reconnect to pull, or train verifier offline instead.\n");
        if (dry) return 1;
        printf("hint: lancius train verifier  (no download, seconds)\n");
        return 1;
    }
    if (dry) {
        printf("dry-run: would exec %s (%s)\n", bin, est);
        printf("data: OK, binary: OK\n");
        return 0;
    }
    printf("training %s (%s) ...\n", argv[0], est);
    return run_shell(bin);
}

/* ---------------- info ---------------- */

static void print_info_help(void) {
    printf("usage: lancius info <model.lancius> [--nodes]\n");
    printf("  prints magic, version, flags, node count, CRC, loader verdict.\n");
    printf("  --nodes  also list every node (id, op, shape, dtype, inputs).\n");
    printf("  training-only ops (_BWD) and reserved ops (EMBEDDING, KV_*) are flagged.\n");
    printf("example: lancius info test_model.lancius --nodes\n");
}

static int cmd_info(int argc, char **argv) {
    int show_nodes = 0;
    const char *model = NULL;
    int i;
    FILE *f;
    uint8_t hdr[48];
    size_t nr;
    uint32_t magic, version, flags, ncount;
    uint32_t checksum;
    lancius_graph *g;
    for (i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--nodes") == 0) show_nodes = 1;
        else if (is_help_arg(argv[i])) {
            print_info_help();
            return 0;
        } else if (argv[i][0] != '-' && !model) model = argv[i];
        else {
            printf("unknown arg '%s' (usage: lancius info <model.lancius> [--nodes])\n", argv[i]);
            return 2;
        }
    }
    if (!model) {
        print_info_help();
        return 2;
    }
    if (!file_exists(model)) {
        printf("FAIL: no such file %s (try lancius models)\n", model);
        return 1;
    }
    f = fopen(model, "rb");
    if (!f) {
        printf("FAIL: cannot open %s\n", model);
        return 1;
    }
    nr = fread(hdr, 1, 48, f);
    fclose(f);
    if (nr != 48) {
        printf("FAIL: %s too small (%zu bytes)\n", model, nr);
        return 1;
    }
    memcpy(&magic, hdr + 0, 4);
    memcpy(&version, hdr + 4, 4);
    memcpy(&flags, hdr + 8, 4);
    memcpy(&ncount, hdr + 12, 4);
    memcpy(&checksum, hdr + 40, 4);
    printf("model: %s (%ld bytes)\n", model, file_size(model));
    printf("  magic=0x%08X version=%u flags=%u nodes=%u crc=%08x\n",
        magic, version, flags, ncount, checksum);
    if (magic != 0x32434E41u) {
        printf("FAIL: bad magic (want 0x32434E41; is this a v1 or non-lancius file?)\n");
        return 1;
    }
    if (checksum == 0)
        printf("  note: checksum==0 legacy file; loader needs LANCIUS_ALLOW_LEGACY_UNVERIFIED=1\n");
    g = lancius_graph_load(model);
    if (!g) {
        printf("FAIL: loader rejected model (see stderr; CRC/integrity?)\n");
        return 1;
    }
    printf("  loader: OK (CRC/integrity verified), %u nodes resident\n", g->node_count);
    if (show_nodes) {
        uint32_t k;
        for (k = 0; k < g->node_count; k++) {
            lancius_node *n = g->nodes[k];
            size_t elems = 0;
            const char *flag = "";
            uint32_t j;
            (void)lancius_node_elements_checked(n, &elems);
            if (is_backward_op((int)n->op)) flag = " [training-only]";
            else if (is_reserved_op((int)n->op)) flag = " [reserved/unimplemented]";
            printf("  id=%u op=%s ndim=%u shape=[%zu,%zu,%zu,%zu] elems=%zu dtype=%s in=[",
                n->id, op_name((int)n->op), n->ndim,
                n->shape[0], n->shape[1], n->shape[2], n->shape[3],
                elems, dtype_name((int)n->dtype));
            for (j = 0; j < n->input_count; j++)
                printf("%s%u", j ? "," : "", n->inputs[j]->id);
            printf("]%s%s\n", n->runtime_data ? " W" : "", flag);
        }
    }
    lancius_graph_destroy(g);
    return 0;
}

/* ---------------- run (generic inference) ---------------- */

static void print_run_help(void) {
    printf("usage: lancius run <model.lancius> [--input f.bin] [--mode wave|static] [--fill random|zero|one] [--topk K] [--show N]\n");
    printf("  --input f.bin  raw f32/f64 vector for the single feed input (else --fill is used)\n");
    printf("  --mode  wave (default, arena) or static (flat buffer, edge-like)\n");
    printf("  --fill  random (default, seed 42), zero, or one (used when no --input)\n");
    printf("  --topk K  show top K classes per row (default 5)\n");
    printf("  --show N  show first N output values (default 16)\n");
    printf("examples:\n");
    printf("  lancius run test_model.lancius --mode static --fill zero\n");
    printf("  lancius run mymodel.lancius --fill one --topk 3 --show 8\n");
}

/* Forward declaration for demo reuse (same TU). */
int lancius_cli_run(int argc, char **argv);

static int cmd_run(int argc, char **argv) {
    const char *model = NULL;
    const char *input_path = NULL;
    const char *mode = "wave";
    const char *fill = "random";
    int topk = 5;
    int show = 16;
    int i;
    lancius_graph *g;
    lancius_schedule *sched;
    lancius_node *feeds[256];
    size_t feed_elems[256];
    int nfeeds = 0;
    double *owned[256] = {0};
    lancius_arena *scratch_keep = NULL;
    void *static_buf = NULL;
    lancius_node *out = NULL;
    uint32_t k;
    int rc = 0;
    for (i = 0; i < argc; i++) {
        if ((strcmp(argv[i], "--input") == 0) && i + 1 < argc) input_path = argv[++i];
        else if ((strcmp(argv[i], "--mode") == 0) && i + 1 < argc) mode = argv[++i];
        else if ((strcmp(argv[i], "--fill") == 0) && i + 1 < argc) fill = argv[++i];
        /* Despot truth: atoi overflow/UB on garbage (was: unchecked). */
        else if ((strcmp(argv[i], "--topk") == 0) && i + 1 < argc) {
            char *ep = NULL;
            long v;
            i++;
            v = strtol(argv[i], &ep, 10);
            if (!ep || *ep != 0 || v < 1 || v > 1000000) {
                printf("FAIL: bad --topk '%s' (want 1..1000000)\n", argv[i]);
                return 2;
            }
            topk = (int)v;
        }
        else if ((strcmp(argv[i], "--show") == 0) && i + 1 < argc) {
            char *ep = NULL;
            long v;
            i++;
            v = strtol(argv[i], &ep, 10);
            if (!ep || *ep != 0 || v < 1 || v > 1000000) {
                printf("FAIL: bad --show '%s' (want 1..1000000)\n", argv[i]);
                return 2;
            }
            show = (int)v;
        }
        else if (is_help_arg(argv[i])) {
            print_run_help();
            return 0;
        } else if (argv[i][0] != '-' && !model) model = argv[i];
        else {
            printf("unknown arg '%s'\n", argv[i]);
            print_run_help();
            return 2;
        }
    }
    if (!model) {
        print_run_help();
        return 2;
    }
    if (strcmp(mode, "wave") != 0 && strcmp(mode, "static") != 0) {
        printf("FAIL: unknown --mode '%s' (want wave|static)\n", mode);
        return 2;
    }
    if (strcmp(fill, "random") != 0 && strcmp(fill, "zero") != 0 && strcmp(fill, "one") != 0) {
        printf("FAIL: unknown --fill '%s' (want random|zero|one)\n", fill);
        return 2;
    }
    if (topk < 1) {
        printf("note: --topk %d clamped to 1\n", topk);
        topk = 1;
    }
    if (show < 1) {
        printf("note: --show %d clamped to 1\n", show);
        show = 1;
    }
    if (!file_exists(model)) {
        printf("FAIL: no such file %s (try lancius models)\n", model);
        return 1;
    }
    if (input_path && !file_exists(input_path)) {
        printf("FAIL: no such --input file %s\n", input_path);
        return 1;
    }
    g = lancius_graph_load(model);
    if (!g) {
        printf("FAIL: cannot load %s (see stderr; CRC/integrity? try lancius info)\n", model);
        return 1;
    }
    sched = lancius_ir_schedule(g);
    if (!sched) {
        printf("FAIL: cannot schedule %s (cycle or corrupt graph?)\n", model);
        lancius_graph_destroy(g);
        return 1;
    }

    /* feed inputs = INPUT nodes with no bound data (weights are pre-bound) */
    for (k = 0; k < g->node_count && nfeeds < 256; k++) {
        lancius_node *n = g->nodes[k];
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

    if (input_path && nfeeds != 1) {
        printf("FAIL: --input needs exactly 1 feed input, graph has %d\n", nfeeds);
        lancius_schedule_destroy(sched);
        lancius_graph_destroy(g);
        return 1;
    }
    srand(42);
    for (i = 0; i < nfeeds; i++) {
        owned[i] = (double *)calloc(feed_elems[i], sizeof(double));
        if (!owned[i]) {
            int j;
            printf("FAIL: OOM\n");
            for (j = 0; j < i; j++) free(owned[j]);
            lancius_schedule_destroy(sched);
            lancius_graph_destroy(g);
            return 1;
        }
        if (input_path) {
            FILE *f = fopen(input_path, "rb");
            long sz;
            if (!f) {
                int j;
                printf("FAIL: cannot open --input %s\n", input_path);
                for (j = 0; j <= i; j++) free(owned[j]);
                lancius_schedule_destroy(sched);
                lancius_graph_destroy(g);
                return 1;
            }
            sz = file_size(input_path);
            if (sz == (long)(feed_elems[i] * sizeof(double))) {
                /* Despot truth: short reads left calloc tails as silent zeros. */
                size_t nr = fread(owned[i], 1, feed_elems[i] * sizeof(double), f);
                if (nr != feed_elems[i] * sizeof(double)) {
                    int j;
                    printf("FAIL: short read on --input %s\n", input_path);
                    fclose(f);
                    for (j = 0; j <= i; j++) free(owned[j]);
                    lancius_schedule_destroy(sched);
                    lancius_graph_destroy(g);
                    return 1;
                }
            } else if (sz == (long)(feed_elems[i] * sizeof(float))) {
                float *tmp = (float *)malloc(feed_elems[i] * sizeof(float));
                if (!tmp) {
                    int j;
                    fclose(f);
                    printf("FAIL: OOM\n");
                    for (j = 0; j <= i; j++) free(owned[j]);
                    lancius_schedule_destroy(sched);
                    lancius_graph_destroy(g);
                    return 1;
                }
                {
                    size_t nr = fread(tmp, 1, feed_elems[i] * sizeof(float), f);
                    size_t kk;
                    if (nr != feed_elems[i] * sizeof(float)) {
                        int j;
                        printf("FAIL: short read on --input %s\n", input_path);
                        free(tmp);
                        fclose(f);
                        for (j = 0; j <= i; j++) free(owned[j]);
                        lancius_schedule_destroy(sched);
                        lancius_graph_destroy(g);
                        return 1;
                    }
                    for (kk = 0; kk < feed_elems[i]; kk++) owned[i][kk] = (double)tmp[kk];
                }
                free(tmp);
            } else {
                int j;
                printf("FAIL: --input size %ld != elems %zu x8 (f64) or x4 (f32)\n", sz, feed_elems[i]);
                printf("hint: feed %d wants %zu doubles (%zu bytes) or %zu floats (%zu bytes)\n",
                    i, feed_elems[i], feed_elems[i] * 8, feed_elems[i], feed_elems[i] * 4);
                fclose(f);
                for (j = 0; j <= i; j++) free(owned[j]);
                lancius_schedule_destroy(sched);
                lancius_graph_destroy(g);
                return 1;
            }
            fclose(f);
        } else if (strcmp(fill, "zero") == 0) {
            memset(owned[i], 0, feed_elems[i] * sizeof(double));
        } else if (strcmp(fill, "one") == 0) {
            size_t kk;
            for (kk = 0; kk < feed_elems[i]; kk++) owned[i][kk] = 1.0;
        } else {
            size_t kk;
            for (kk = 0; kk < feed_elems[i]; kk++)
                owned[i][kk] = ((double)rand() / (double)RAND_MAX) - 0.5;
        }
        lancius_node_bind_external(feeds[i], owned[i]);
    }

    if (strcmp(mode, "static") == 0) {
        size_t need = lancius_schedule_static_memory_required(sched);
        if (need == 0) need = 1024 * 1024;
        /* Despot V6 truth: 32B-aligned pool (was 16B malloc breaking AVX2). */
        need = (need + 31) & ~(size_t)31;
        static_buf = NULL;
        if (posix_memalign(&static_buf, 32, need + 32) != 0) static_buf = NULL;
        if (!static_buf) {
            int j;
            printf("FAIL: OOM static buffer %zu\n", need);
            for (j = 0; j < nfeeds; j++) free(owned[j]);
            lancius_schedule_destroy(sched);
            lancius_graph_destroy(g);
            return 1;
        }
        lancius_schedule_execute_static(sched, static_buf);
    } else {
        scratch_keep = lancius_arena_create(64 * 1024 * 1024);
        if (!scratch_keep) {
            int j;
            printf("FAIL: OOM arena\n");
            for (j = 0; j < nfeeds; j++) free(owned[j]);
            lancius_schedule_destroy(sched);
            lancius_graph_destroy(g);
            return 1;
        }
        lancius_schedule_execute(sched, scratch_keep);
    }

    /* output: CE-logits convention else last computed node */
    for (k = 0; k < g->node_count; k++) {
        if (g->nodes[k]->op == LANCIUS_OP_CROSS_ENTROPY && g->nodes[k]->input_count > 0) {
            out = (lancius_node *)g->nodes[k]->inputs[0];
            break;
        }
    }
    if (!out) {
        for (k = g->node_count; k > 0; k--) {
            lancius_node *n = g->nodes[k - 1];
            if (n->op != LANCIUS_OP_INPUT && n->op != LANCIUS_OP_CONST &&
                n->op != LANCIUS_OP_NOP && n->runtime_data) {
                out = n;
                break;
            }
        }
    }
    if (!out || !out->runtime_data) {
        int j;
        printf("FAIL: no output buffer (graph did not execute? try lancius info --nodes)\n");
        rc = 1;
        for (j = 0; j < nfeeds; j++) free(owned[j]);
        if (scratch_keep) lancius_arena_destroy(scratch_keep);
        if (static_buf) free(static_buf);
        lancius_schedule_destroy(sched);
        lancius_graph_destroy(g);
        return rc;
    } else {
        size_t e = 0;
        size_t lim, kk;
        (void)lancius_node_elements_checked(out, &e);
        printf("output: id=%u op=%s shape=[%zu,%zu,%zu,%zu] elems=%zu\n",
            out->id, op_name((int)out->op),
            out->shape[0], out->shape[1], out->shape[2], out->shape[3], e);
        lim = e < (size_t)show ? e : (size_t)show;
        printf("values[0..%zu]:", lim);
        for (kk = 0; kk < lim; kk++) printf(" %.4f", out->runtime_data[kk]);
        printf("\n");
        if (out->ndim == 2 && out->shape[1] >= 2) {
            size_t R = out->shape[0], C = out->shape[1];
            size_t rlim = R < 4 ? R : 4;
            size_t r;
            int *taken = (int *)calloc(C, sizeof(int));
            if (!taken) {
                printf("FAIL: OOM\n");
                rc = 1;
            } else {
                for (r = 0; r < rlim; r++) {
                    size_t c;
                    int t, best = 0;
                    memset(taken, 0, C * sizeof(int));
                    for (c = 1; c < C; c++)
                        if (out->runtime_data[r * C + c] > out->runtime_data[r * C + (size_t)best]) best = (int)c;
                    printf("row %zu: argmax=%d (%.4f) | top%d:", r, best,
                        out->runtime_data[r * C + (size_t)best], topk);
                    for (t = 0; t < topk && t < (int)C; t++) {
                        int bi = -1;
                        double bv = -1e300;
                        for (c = 0; c < C; c++) {
                            double v;
                            if (taken[c]) continue;
                            v = out->runtime_data[r * C + c];
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

    for (k = 0; (int)k < nfeeds; k++) free(owned[k]);
    if (scratch_keep) lancius_arena_destroy(scratch_keep);
    if (static_buf) free(static_buf);
    lancius_schedule_destroy(sched);
    lancius_graph_destroy(g);
    return rc;
}

/* R2-6 gate: eval runs a vendored micromodel end to end (run + report). */
static int cmd_eval(int argc, char **argv) {
    const char *model = (argc > 0 && argv[0][0] != '-') ? argv[0] : "test_model.lancius";
    const char *mode = "static";
    for (int i = 0; i < argc; i++) {
        if (strcmp(argv[i], "--mode") == 0 && i + 1 < argc) mode = argv[++i];
        else if (is_help_arg(argv[i])) {
            printf("usage: lancius eval <model.lancius> [--mode wave|static]\n");
            printf("  runs vendored micromodel end to end (R2-6 gate).\n");
            return 0;
        }
    }
    if (!file_exists(model)) {
        printf("FAIL: missing model %s\n", model);
        return 1;
    }
    {
        char *sub[5];
        char modebuf[16];
        snprintf(modebuf, sizeof(modebuf), "%s", mode);
        sub[0] = (char*)model;
        sub[1] = (char*)"--mode"; sub[2] = modebuf;
        sub[3] = (char*)"--fill"; sub[4] = (char*)"zero";
        int rc = cmd_run(5, sub);
        if (rc != 0) { printf("FAIL: eval run failed\n"); return rc; }
    }
    printf("eval: %s OK (mode=%s)\n", model, mode);
    return 0;
}

/* ---------------- convert / export ---------------- */

static void print_convert_help(void) {
    printf("usage: lancius convert onnx2lancius <in.onnx> <out.lancius> | lancius2onnx <in.lancius> <out.onnx>\n");
    printf("  onnx2lancius  strict LeNet-class converter (needs python onnx+numpy)\n");
    printf("  lancius2onnx  inference-subset exporter (needs python onnx+numpy)\n");
    printf("examples:\n");
    printf("  lancius convert onnx2lancius model.onnx model.lancius\n");
    printf("  lancius convert lancius2onnx model.lancius model.onnx\n");
}

static void print_export_help(void) {
    printf("usage: lancius export pytorch <in.lancius> <out.py> [--onnx out.onnx]\n");
    printf("  writes a PyTorch nn.Module mirroring the lancius graph (needs torch for --onnx trace check).\n");
    printf("example: lancius export pytorch model.lancius model.py --onnx model.onnx\n");
}

static int have_py(const char *mod) {
    char cmd[256];
    snprintf(cmd, sizeof(cmd), "python3 -c \"import %s\" 2>/dev/null", mod);
    return system(cmd) == 0;
}

static int cmd_convert(int argc, char **argv) {
    if (argc < 1) {
        print_convert_help();
        return 2;
    }
    if (is_help_arg(argv[0])) {
        print_convert_help();
        return 0;
    }
    if (strcmp(argv[0], "onnx2lancius") == 0) {
        if (argc != 3 || is_help_arg(argv[1])) {
            print_convert_help();
            return 2;
        }
        if (!file_exists(argv[1])) {
            printf("FAIL: no such input %s\n", argv[1]);
            return 1;
        }
        if (!file_exists("onnx_to_lancius.py")) {
            printf("FAIL: onnx_to_lancius.py missing in this directory\n");
            return 1;
        }
        if (!have_py("onnx") || !have_py("numpy")) {
            printf("FAIL: python onnx+numpy missing (pip install onnx numpy)\n");
            return 1;
        }
        {
            /* Despot truth: no shell (was: injection via filenames). */
            char *xa[5];
            xa[0] = "python3"; xa[1] = "onnx_to_lancius.py";
            xa[2] = argv[1]; xa[3] = argv[2]; xa[4] = NULL;
            return run_argv(xa[0], xa);
        }
    }
    if (strcmp(argv[0], "lancius2onnx") == 0) {
        if (argc != 3 || is_help_arg(argv[1])) {
            print_convert_help();
            return 2;
        }
        if (!file_exists(argv[1])) {
            printf("FAIL: no such input %s (try lancius models)\n", argv[1]);
            return 1;
        }
        if (!file_exists("export_lancius_onnx.py")) {
            printf("FAIL: export_lancius_onnx.py missing in this directory\n");
            return 1;
        }
        if (!have_py("onnx") || !have_py("numpy")) {
            printf("FAIL: python onnx+numpy missing (pip install onnx numpy)\n");
            return 1;
        }
        {
            char *xa[5];
            xa[0] = "python3"; xa[1] = "export_lancius_onnx.py";
            xa[2] = argv[1]; xa[3] = argv[2]; xa[4] = NULL;
            return run_argv(xa[0], xa);
        }
    }
    printf("unknown convert '%s'\n", argv[0]);
    print_convert_help();
    return 2;
}

static int cmd_export(int argc, char **argv) {
    int i;
    if (argc < 1 || is_help_arg(argv[0])) {
        print_export_help();
        return argc < 1 ? 2 : 0;
    }
    if (argc < 3 || strcmp(argv[0], "pytorch") != 0) {
        print_export_help();
        return 2;
    }
    if (is_help_arg(argv[1])) {
        print_export_help();
        return 0;
    }
    if (!file_exists(argv[1])) {
        printf("FAIL: no such input %s (try lancius models)\n", argv[1]);
        return 1;
    }
    if (!file_exists("export_lancius_pytorch.py")) {
        printf("FAIL: export_lancius_pytorch.py missing in this directory\n");
        return 1;
    }
    /* Despot truth: no shell + no silent truncation (was: both). */
    {
        char *xa[256];
        int ac = 0;
        if (argc > 250) {
            printf("FAIL: too many export args (%d)\n", argc);
            return 1;
        }
        xa[ac++] = "python3";
        xa[ac++] = "export_lancius_pytorch.py";
        for (i = 1; i < argc; i++) {
            if (strlen(argv[i]) > 1024) {
                printf("FAIL: export arg too long\n");
                return 1;
            }
            xa[ac++] = argv[i];
        }
        xa[ac] = NULL;
        return run_argv(xa[0], xa);
    }
}

/* ---------------- help ---------------- */

static void print_verb_help(const char *verb) {
    if (!verb || !verb[0]) {
        print_usage();
        return;
    }
    if (strcmp(verb, "quickstart") == 0) {
        printf("usage: lancius quickstart\n  guided setup for new users: doctor, demo, train, TUI.\n");
        return;
    }
    if (strcmp(verb, "doctor") == 0) {
        printf("usage: lancius doctor\n  readiness check; exits 1 when core pieces are MISSING.\n");
        return;
    }
    if (strcmp(verb, "status") == 0) {
        printf("usage: lancius status\n  operational snapshot; always exits 0.\n");
        return;
    }
    if (strcmp(verb, "models") == 0) {
        printf("usage: lancius models [--check]\n");
        return;
    }
    if (strcmp(verb, "demo") == 0) {
        printf("usage: lancius demo\n  one-command install proof on a local .lancius model.\n");
        return;
    }
    if (strcmp(verb, "datasets") == 0) {
        char *av[1];
        av[0] = "help";
        cmd_datasets(1, av);
        return;
    }
    if (strcmp(verb, "train") == 0) {
        print_train_help();
        return;
    }
    if (strcmp(verb, "run") == 0) {
        print_run_help();
        return;
    }
    if (strcmp(verb, "info") == 0) {
        print_info_help();
        return;
    }
    if (strcmp(verb, "convert") == 0) {
        print_convert_help();
        return;
    }
    if (strcmp(verb, "export") == 0) {
        print_export_help();
        return;
    }
    if (strcmp(verb, "generate") == 0) {
        printf("usage: lancius generate\n  transformer prefill/generation demo (needs ./generate_text).\n");
        return;
    }
    if (strcmp(verb, "tui") == 0) {
        printf("usage: lancius tui\n  guided menus; type a number, 'help', or '0' to quit.\n");
        return;
    }
    printf("unknown verb '%s'\n", verb);
    print_usage();
}

/* ---------------- TUI ---------------- */

static void tui_line(void) {
    printf("------------------------------------------------------------\n");
}

static int tui_read(char *buf, size_t cap) {
    int c;
    if (cap == 0) return 0;
    if (!fgets(buf, (int)cap, stdin)) return 0;
    /* Despot truth: overlong lines left tails for the next prompt, shifting
     * every later answer (was: undrained). */
    if (!strchr(buf, '\n') && !feof(stdin)) {
        while ((c = fgetc(stdin)) != '\n' && c != EOF) {}
    }
    buf[strcspn(buf, "\r\n")] = 0;
    trim_inplace(buf);
    return 1;
}

static int tui_prompt(const char *label, char *buf, size_t cap) {
    printf("%s: ", label);
    fflush(stdout);
    return tui_read(buf, cap);
}

static int tui_prompt_def(const char *label, const char *def, char *buf, size_t cap) {
    printf("%s [%s]: ", label, def ? def : "");
    fflush(stdout);
    if (!tui_read(buf, cap)) return 0;
    if (!buf[0] && def) snprintf(buf, cap, "%s", def);
    trim_inplace(buf);
    return 1;
}

static void tui_pause(void) {
    char b[16];
    if (!is_tty_stdin()) return;
    printf("press Enter to continue...");
    fflush(stdout);
    if (!fgets(b, sizeof(b), stdin)) return;
}

static int tui_quit_word(const char *s) {
    return s && (strcmp(s, "0") == 0 || strcmp(s, "q") == 0 ||
                 strcmp(s, "Q") == 0 || strcmp(s, "quit") == 0 ||
                 strcmp(s, "exit") == 0);
}

static int tui_back_word(const char *s) {
    return s && (s[0] == 0 || strcmp(s, "back") == 0 || strcmp(s, "b") == 0);
}

static int tui_help_word(const char *s) {
    return s && (strcmp(s, "help") == 0 || strcmp(s, "?") == 0 ||
                 strcmp(s, "h") == 0);
}

static void tui_header(void) {
    int n = 0;
    DIR *d = opendir(".");
    struct dirent *e;
    if (d) {
        while ((e = readdir(d)) != NULL) {
            size_t L = strlen(e->d_name);
            if (L >= 8 && strcmp(e->d_name + L - 8, ".lancius") == 0) n++;
        }
        closedir(d);
    }
    printf("\033[2J\033[H");
    printf("== lancius operator (TUI) %s ==\n", LANCIUS_VERSION_STRING);
    printf("models here: %d (.lancius) | data: mnist %s, cifar10 %s | type help, 0 to quit\n",
        n,
        file_exists("train-images-idx3-ubyte") ? "yes" : "no",
        file_exists("cifar-10-batches-bin/data_batch_1.bin") ? "yes" : "no");
}

static void tui_menu(void) {
    tui_line();
    printf(" 1 quickstart      guided setup for new users (doctor, demo, train, TUI)\n");
    printf(" 2 demo            prove the install in one command (seconds)\n");
    printf(" 3 doctor          readiness check (binaries, python, data, network)\n");
    printf(" 4 status/models   snapshot + list local models\n");
    printf(" 5 datasets        pull or check training data (needs network to pull)\n");
    printf(" 6 train           train mnist (minutes) | cifar10 (hours!) | verifier (seconds)\n");
    printf(" 7 run             run a model (outputs + top classes)\n");
    printf(" 8 info            inspect a model file\n");
    printf(" 9 convert/export  onnx <-> lancius, lancius -> pytorch\n");
    printf("10 generate        transformer demo text\n");
    printf(" 0 quit\n");
    tui_line();
}

static void tui_help(void) {
    printf("help — what each entry does:\n");
    printf(" 1 demo: picks test_model.lancius (or the first model here) and runs\n");
    printf("    info + static zero-fill run. Use it first after `make`.\n");
    printf(" 2 doctor: tells you what is missing and the exact fix command.\n");
    printf(" 3 status/models: shows version, files, data, disk, network, models.\n");
    printf(" 4 datasets: 'status' is offline-safe; 'pull' needs network.\n");
    printf(" 5 train: verifier needs nothing and takes seconds — start there.\n");
    printf("    mnist needs MNIST data (minutes); cifar10 needs CIFAR data (hours).\n");
    printf(" 6 run: needs a model path; defaults fill=random mode=wave work offline.\n");
    printf(" 7 info: needs a model path; --nodes lists every tensor.\n");
    printf(" 8 convert/export: needs python onnx+numpy (and torch for pytorch trace).\n");
    printf("tips: empty answer goes back, 0 quits from anywhere, answers are trimmed.\n");
}

static int tui_do_demo(void) {
    int rc = cmd_demo();
    printf("demo: %s%s%s (exit %d)\n", rc == 0 ? c_ok() : c_bad(),
        rc == 0 ? "OK" : "FAIL", c_off(), rc);
    return rc;
}

static int tui_do_doctor(void) {
    int rc = cmd_doctor();
    printf("doctor: %s%s%s (exit %d)\n", rc == 0 ? c_ok() : c_bad(),
        rc == 0 ? "READY" : "INCOMPLETE", c_off(), rc);
    return rc;
}

static int tui_do_status_models(void) {
    int rc;
    rc = cmd_status();
    printf("-- local models --\n");
    {
        int rcm = cmd_models(0, NULL);
        if (rc == 0) rc = rcm;
    }
    return rc;
}

static int tui_do_datasets(void) {
    char a[256], b[256];
    char *av[3];
    if (!tui_prompt("datasets: list|status|pull|distill|back [status]", a, sizeof(a))) return 0;
    if (!a[0]) snprintf(a, sizeof(a), "status");
    if (tui_back_word(a)) return 0;
    if (tui_help_word(a)) {
        print_datasets_help();
        return 0;
    }
    if (strcmp(a, "list") == 0 || strcmp(a, "status") == 0) {
        av[0] = a;
        return cmd_datasets(1, av);
    }
    if (strcmp(a, "pull") == 0) {
        int net;
        if (!tui_prompt("target (mnist|cifar10|vision|prm800k|all) [vision]", b, sizeof(b))) return 0;
        if (!b[0]) snprintf(b, sizeof(b), "vision");
        if (tui_back_word(b)) return 0;
        if (!valid_pull_target(b)) {
            printf("unknown dataset '%s' (try lancius datasets list)\n", b);
            return 2;
        }
        net = net_online();
        if (net == 0) {
            printf("network: offline — pull needs network. Cached data (status) still works.\n");
            return 1;
        }
        av[0] = "pull";
        av[1] = b;
        av[2] = NULL;
        return cmd_datasets(2, av);
    }
    if (strcmp(a, "distill") == 0) {
        av[0] = "distill";
        return cmd_datasets(1, av);
    }
    printf("unknown datasets command '%s' (list|status|pull|distill)\n", a);
    return 2;
}

static int tui_do_train(void) {
    char a[256];
    char *av[1];
    printf("train time guide: verifier seconds (no download) | mnist minutes | cifar10 HOURS\n");
    if (!tui_prompt("target (mnist|cifar10|verifier) [verifier]", a, sizeof(a))) return 0;
    if (!a[0]) snprintf(a, sizeof(a), "verifier");
    if (tui_back_word(a)) return 0;
    if (tui_help_word(a)) {
        print_train_help();
        return 0;
    }
    if (strcmp(a, "mnist") != 0 && strcmp(a, "cifar10") != 0 && strcmp(a, "verifier") != 0) {
        printf("unknown train target '%s'\n", a);
        return 2;
    }
    if (strcmp(a, "cifar10") == 0) {
        char yn[16];
        printf("warning: cifar10 trains 30 epochs (hours). verifier takes seconds.\n");
        if (!tui_prompt("still train cifar10? (yes|no) [no]", yn, sizeof(yn))) return 0;
        if (!(strcmp(yn, "yes") == 0 || strcmp(yn, "y") == 0)) {
            printf("train cancelled (try verifier).\n");
            return 0;
        }
    }
    av[0] = a;
    return cmd_train(1, av);
}

static int tui_do_run(void) {
    char m[512], mode[64], fill[64], topk[64], show[64];
    char *av[11];
    int ac = 0;
    if (!tui_prompt("model path [test_model.lancius]", m, sizeof(m))) return 0;
    if (!m[0]) snprintf(m, sizeof(m), "test_model.lancius");
    if (tui_back_word(m)) return 0;
    if (!file_exists(m)) {
        printf("FAIL: no such file %s (see menu 3 for local models)\n", m);
        return 1;
    }
    if (!tui_prompt_def("mode (wave|static)", "wave", mode, sizeof(mode))) return 0;
    if (tui_back_word(mode)) return 0;
    if (strcmp(mode, "wave") != 0 && strcmp(mode, "static") != 0) {
        printf("unknown mode '%s' (wave|static)\n", mode);
        return 2;
    }
    if (!tui_prompt_def("fill (random|zero|one)", "random", fill, sizeof(fill))) return 0;
    if (tui_back_word(fill)) return 0;
    if (strcmp(fill, "random") != 0 && strcmp(fill, "zero") != 0 && strcmp(fill, "one") != 0) {
        printf("unknown fill '%s' (random|zero|one)\n", fill);
        return 2;
    }
    if (!tui_prompt_def("topk", "5", topk, sizeof(topk))) return 0;
    if (tui_back_word(topk)) return 0;
    if (!tui_prompt_def("show values", "16", show, sizeof(show))) return 0;
    if (tui_back_word(show)) return 0;
    /* Despot truth: the show answer was prompted then dropped (stayed 16). */
    av[ac++] = m;
    av[ac++] = "--mode"; av[ac++] = mode;
    av[ac++] = "--fill"; av[ac++] = fill;
    av[ac++] = "--topk"; av[ac++] = topk;
    av[ac++] = "--show"; av[ac++] = show;
    return cmd_run(ac, av);
}

static int tui_do_info(void) {
    char m[512], yn[16];
    char *av[2];
    if (!tui_prompt("model path [test_model.lancius]", m, sizeof(m))) return 0;
    if (!m[0]) snprintf(m, sizeof(m), "test_model.lancius");
    if (tui_back_word(m)) return 0;
    if (!file_exists(m)) {
        printf("FAIL: no such file %s (see menu 3 for local models)\n", m);
        return 1;
    }
    if (!tui_prompt("list every node? (yes|no) [no]", yn, sizeof(yn))) return 0;
    if (strcmp(yn, "yes") == 0 || strcmp(yn, "y") == 0) {
        av[0] = m;
        av[1] = "--nodes";
        return cmd_info(2, av);
    }
    av[0] = m;
    return cmd_info(1, av);
}

static int tui_do_convert(void) {
    char kind[64], a[512], b[512];
    char *av[3];
    if (!tui_prompt("direction (onnx2lancius|lancius2onnx|pytorch) [onnx2lancius]", kind, sizeof(kind))) return 0;
    if (!kind[0]) snprintf(kind, sizeof(kind), "onnx2lancius");
    if (tui_back_word(kind)) return 0;
    if (strcmp(kind, "onnx2lancius") == 0) {
        if (!tui_prompt("in.onnx", a, sizeof(a))) return 0;
        if (tui_back_word(a)) return 0;
        if (!tui_prompt("out.lancius", b, sizeof(b))) return 0;
        if (tui_back_word(b)) return 0;
        av[0] = "onnx2lancius";
        av[1] = a;
        av[2] = b;
        return cmd_convert(3, av);
    }
    if (strcmp(kind, "lancius2onnx") == 0) {
        if (!tui_prompt("in.lancius", a, sizeof(a))) return 0;
        if (tui_back_word(a)) return 0;
        if (!tui_prompt("out.onnx", b, sizeof(b))) return 0;
        if (tui_back_word(b)) return 0;
        av[0] = "lancius2onnx";
        av[1] = a;
        av[2] = b;
        return cmd_convert(3, av);
    }
    if (strcmp(kind, "pytorch") == 0) {
        char c[512];
        char *avx[5];
        int acx = 0;
        if (!tui_prompt("in.lancius", a, sizeof(a))) return 0;
        if (tui_back_word(a)) return 0;
        if (!tui_prompt("out.py", b, sizeof(b))) return 0;
        if (tui_back_word(b)) return 0;
        if (!tui_prompt("also onnx path (empty=skip)", c, sizeof(c))) return 0;
        avx[acx++] = "pytorch";
        avx[acx++] = a;
        avx[acx++] = b;
        if (c[0]) {
            avx[acx++] = "--onnx";
            avx[acx++] = c;
        }
        return cmd_export(acx, avx);
    }
    printf("unknown direction '%s'\n", kind);
    return 2;
}

static int cmd_tui(void) {
    char buf[256];
    int last = 0;
    tui_header();
    printf("welcome — option 1 is a guided quickstart, 6 starts with verifier (seconds).\n");
    for (;;) {
        tui_menu();
        printf("last: %s%s%s | select (help, 0 quit): ",
            last == 0 ? c_ok() : c_bad(), last == 0 ? "OK" : "FAIL", c_off());
        fflush(stdout);
        if (!tui_read(buf, sizeof(buf))) break;
        if (tui_quit_word(buf)) break;
        else if (tui_help_word(buf)) {
            tui_help();
            tui_pause();
        } else if (strcmp(buf, "1") == 0) {
            last = cmd_quickstart();
            tui_pause();
        } else if (strcmp(buf, "2") == 0) {
            last = tui_do_demo();
            tui_pause();
        } else if (strcmp(buf, "3") == 0) {
            last = tui_do_doctor();
            tui_pause();
        } else if (strcmp(buf, "4") == 0) {
            last = tui_do_status_models();
            tui_pause();
        } else if (strcmp(buf, "5") == 0) {
            last = tui_do_datasets();
            tui_pause();
        } else if (strcmp(buf, "6") == 0) {
            last = tui_do_train();
            tui_pause();
        } else if (strcmp(buf, "7") == 0) {
            last = tui_do_run();
            tui_pause();
        } else if (strcmp(buf, "8") == 0) {
            last = tui_do_info();
            tui_pause();
        } else if (strcmp(buf, "9") == 0) {
            last = tui_do_convert();
            tui_pause();
        } else if (strcmp(buf, "10") == 0) {
            if (!file_exists("./generate_text")) {
                printf("FAIL: ./generate_text missing — run `make`.\n");
                last = 1;
            } else {
                last = run_shell("./generate_text");
            }
            tui_pause();
        } else {
            printf("unknown selection '%s' (try help)\n", buf);
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
        if (argc >= 3) print_verb_help(argv[2]);
        else print_usage();
        return 0;
    }
    if (strcmp(argv[1], "--version") == 0 || strcmp(argv[1], "version") == 0) {
        print_version();
        return 0;
    }
    if (strcmp(argv[1], "quickstart") == 0) {
        if (argc >= 3 && is_help_arg(argv[2])) {
            printf("usage: lancius quickstart\n  guided setup for new users: doctor, demo, train, TUI.\n");
            return 0;
        }
        return cmd_quickstart();
    }
    if (strcmp(argv[1], "doctor") == 0) {
        if (argc >= 3 && is_help_arg(argv[2])) {
            printf("usage: lancius doctor\n");
            return 0;
        }
        return cmd_doctor();
    }
    if (strcmp(argv[1], "status") == 0) {
        if (argc >= 3 && is_help_arg(argv[2])) {
            printf("usage: lancius status\n");
            return 0;
        }
        return cmd_status();
    }
    if (strcmp(argv[1], "models") == 0) return cmd_models(argc - 2, argv + 2);
    if (strcmp(argv[1], "demo") == 0) {
        if (argc >= 3 && is_help_arg(argv[2])) {
            printf("usage: lancius demo\n");
            return 0;
        }
        return cmd_demo();
    }
    if (strcmp(argv[1], "datasets") == 0) return cmd_datasets(argc - 2, argv + 2);
    if (strcmp(argv[1], "train") == 0) return cmd_train(argc - 2, argv + 2);
    if (strcmp(argv[1], "run") == 0) return cmd_run(argc - 2, argv + 2);
    if (strcmp(argv[1], "eval") == 0) return cmd_eval(argc - 2, argv + 2);
    if (strcmp(argv[1], "info") == 0) return cmd_info(argc - 2, argv + 2);
    if (strcmp(argv[1], "convert") == 0) return cmd_convert(argc - 2, argv + 2);
    if (strcmp(argv[1], "export") == 0) return cmd_export(argc - 2, argv + 2);
    if (strcmp(argv[1], "generate") == 0) {
        if (argc >= 3 && is_help_arg(argv[2])) {
            printf("usage: lancius generate\n");
            return 0;
        }
        if (!file_exists("./generate_text")) {
            printf("FAIL: ./generate_text missing — run `make`.\n");
            return 1;
        }
        return run_shell("./generate_text");
    }
    if (strcmp(argv[1], "tui") == 0) {
        if (argc >= 3 && is_help_arg(argv[2])) {
            printf("usage: lancius tui\n");
            return 0;
        }
        return cmd_tui();
    }
    printf("unknown verb '%s'\n", argv[1]);
    print_usage();
    return 2;
}
