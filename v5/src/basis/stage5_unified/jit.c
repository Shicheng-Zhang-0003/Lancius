#define _DEFAULT_SOURCE
#include "basis/stage5_unified/jit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include <unistd.h>
#include <stdarg.h>

// --- Dynamic String Buffer ---
typedef struct {
    char* data;
    size_t len;
    size_t cap;
} jit_buf;

static void jit_buf_init(jit_buf* b, size_t initial_cap) {
    b->cap = initial_cap;
    b->len = 0;
    b->data = (char*)malloc(b->cap);
    b->data[0] = '\0';
}

static void jit_buf_append(jit_buf* b, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    int needed = vsnprintf(NULL, 0, fmt, args);
    va_end(args);

    while (b->len + (size_t)needed + 1 > b->cap) {
        b->cap *= 2;
        b->data = (char*)realloc(b->data, b->cap);
    }

    va_start(args, fmt);
    vsnprintf(b->data + b->len, b->cap - b->len, fmt, args);
    va_end(args);
    b->len += (size_t)needed;
}

static void jit_buf_free(jit_buf* b) {
    free(b->data);
}

// --- Recursive Post-Order AST Emitter ---
// Returns the temp variable ID for this node
static int jit_emit_node(basis_symbol* sym, jit_buf* buf, int* temp_counter, char** var_names, size_t var_count) {
    if (!sym) return -1;

    if (sym->type == basis_symbol_type_constant) {
        int id = (*temp_counter)++;
        jit_buf_append(buf, "    double t%d = %.17g;\n", id, sym->basis_value);
        return id;
    }

    if (sym->type == basis_symbol_type_variable) {
        for (size_t i = 0; i < var_count; i++) {
            if (strcmp(sym->name, var_names[i]) == 0) {
                int id = (*temp_counter)++;
                jit_buf_append(buf, "    double t%d = vars[%zu];\n", id, i);
                return id;
            }
        }
        return -1;
    }

    if (sym->type == basis_symbol_type_operation) {
        int left_id = -1, right_id = -1;
        if (sym->argument_count > 0)
            left_id = jit_emit_node(sym->arguments[0], buf, temp_counter, var_names, var_count);
        if (sym->argument_count > 1)
            right_id = jit_emit_node(sym->arguments[1], buf, temp_counter, var_names, var_count);

        int id = (*temp_counter)++;

        if (strcmp(sym->operation, "+") == 0)
            jit_buf_append(buf, "    double t%d = t%d + t%d;\n", id, left_id, right_id);
        else if (strcmp(sym->operation, "-") == 0)
            jit_buf_append(buf, "    double t%d = t%d - t%d;\n", id, left_id, right_id);
        else if (strcmp(sym->operation, "*") == 0)
            jit_buf_append(buf, "    double t%d = t%d * t%d;\n", id, left_id, right_id);
        else if (strcmp(sym->operation, "**") == 0)
            jit_buf_append(buf, "    double t%d = pow(t%d, %.17g);\n", id, left_id, sym->basis_value);
        else if (strcmp(sym->operation, "exp") == 0)
            jit_buf_append(buf, "    double t%d = exp(t%d);\n", id, left_id);
        else if (strcmp(sym->operation, "log") == 0)
            jit_buf_append(buf, "    double t%d = log(t%d);\n", id, left_id);
        else if (strcmp(sym->operation, "sin") == 0)
            jit_buf_append(buf, "    double t%d = sin(t%d);\n", id, left_id);
        else if (strcmp(sym->operation, "cos") == 0)
            jit_buf_append(buf, "    double t%d = cos(t%d);\n", id, left_id);
        else if (strcmp(sym->operation, "tanh") == 0)
            jit_buf_append(buf, "    double t%d = tanh(t%d);\n", id, left_id);
        else
            jit_buf_append(buf, "    double t%d = 0.0; /* unknown op */\n", id);

        return id;
    }
    return -1;
}

// --- Public API ---
basis_jit_module* basis_jit_compile(basis_symbol* root, char** var_names, size_t var_count) {
    if (!root) return NULL;

    jit_buf buf;
    jit_buf_init(&buf, 4096);

    // Function header
    jit_buf_append(&buf, "#include <math.h>\n");
    jit_buf_append(&buf, "double basis_jit_func(double* vars) {\n");

    // Emit AST
    int temp_counter = 0;
    int root_id = jit_emit_node(root, &buf, &temp_counter, var_names, var_count);

    if (root_id < 0) {
        jit_buf_free(&buf);
        return NULL;
    }

    // Return the root temp variable
    jit_buf_append(&buf, "    return t%d;\n", root_id);
    jit_buf_append(&buf, "}\n");

    // Write to temp file
    char base_file[] = "/tmp/basis_jit_XXXXXX";
    int fd_c = mkstemp(base_file);
    if (fd_c == -1) {
        jit_buf_free(&buf);
        return NULL;
    }

    char c_file[256];
    char so_file[256];
    snprintf(c_file, sizeof(c_file), "%s.c", base_file);
    snprintf(so_file, sizeof(so_file), "%s.so", base_file);

    ssize_t written = write(fd_c, buf.data, buf.len);
    (void)written;
    close(fd_c);
    jit_buf_free(&buf);

    rename(base_file, c_file);

    // Compile with maximum optimization
    char cmd[1024];
    snprintf(cmd, sizeof(cmd), "gcc -shared -fPIC -O3 -march=native -o %s %s -lm 2>/dev/null", so_file, c_file);
    int status = system(cmd);
    unlink(c_file);

    if (status != 0) return NULL;

    // Load shared object
    void* handle = dlopen(so_file, RTLD_NOW);
    if (!handle) {
        unlink(so_file);
        return NULL;
    }

    basis_jit_func_t func = (basis_jit_func_t)dlsym(handle, "basis_jit_func");
    if (!func) {
        dlclose(handle);
        unlink(so_file);
        return NULL;
    }

    basis_jit_module* mod = (basis_jit_module*)malloc(sizeof(basis_jit_module));
    mod->handle = handle;
    mod->func = func;
    mod->so_path = strdup(so_file);
    return mod;
}

double basis_jit_execute(basis_jit_module* mod, double* var_values) {
    if (!mod || !mod->func) return 0.0;
    return mod->func(var_values);
}

void basis_jit_free(basis_jit_module* mod) {
    if (!mod) return;
    if (mod->handle) dlclose(mod->handle);
    if (mod->so_path) {
        unlink(mod->so_path);
        free(mod->so_path);
    }
    free(mod);
}
