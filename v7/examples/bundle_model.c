#include <basis.h>
#include <stdio.h>

// Helper to emit a BASIS tensor as a hardcoded C constant array
void emit_array(FILE* f, const char* name, basis_tensor* t) {
    fprintf(f, "const double %s[%zu] = {\n", name, t->row_count * t->column_count);
    for(size_t i=0; i<t->row_count; i++) {
        fprintf(f, "    ");
        for(size_t j=0; j<t->column_count; j++) {
            fprintf(f, "%.17g, ", t->data[i*t->column_count + j]->data);
        }
        fprintf(f, "\n");
    }
    fprintf(f, "};\n\n");
}

int main() {
    printf("=== Ghost-Weight Bundler ===\n");
    printf("Reading trained weights from disk...\n");

    basis_tensor* W1 = basis_tensor_load_binary("W1.basis");
    basis_tensor* b1 = basis_tensor_load_binary("b1.basis");
    basis_tensor* W2 = basis_tensor_load_binary("W2.basis");
    basis_tensor* b2 = basis_tensor_load_binary("b2.basis");

    if (!W1 || !b1 || !W2 || !b2) {
        printf("Error: Could not load .basis files. Run ./deploy_xor first!\n");
        return 1;
    }

    printf("Generating standalone_model.c...\n");
    FILE* f = fopen("standalone_model.c", "w");

    fprintf(f, "#include <stdio.h>\n\n");

    // 1. Embed the weights directly into the C source code!
    emit_array(f, "W1", W1);
    emit_array(f, "b1", b1);
    emit_array(f, "W2", W2);
    emit_array(f, "b2", b2);

    // 2. Emit the forward pass logic using the hardcoded arrays
    fprintf(f, "int main() {\n");
    fprintf(f, "    double X[8] = {0,0, 0,1, 1,0, 1,1};\n");
    fprintf(f, "    double H1[32] = {0};\n");
    fprintf(f, "    double Y[4] = {0};\n\n");

    // Layer 1 (MatMul + Broadcast Add + ReLU)
    fprintf(f, "    for(int i=0; i<4; i++) {\n");
    fprintf(f, "        for(int j=0; j<8; j++) {\n");
    fprintf(f, "            double sum = 0.0;\n");
    fprintf(f, "            for(int k=0; k<2; k++) sum += X[i*2+k] * W1[k*8+j];\n");
    fprintf(f, "            double val = sum + b1[j];\n");
    fprintf(f, "            H1[i*8+j] = val > 0.0 ? val : 0.0;\n");
    fprintf(f, "        }\n");
    fprintf(f, "    }\n\n");

    // Layer 2 (MatMul + Broadcast Add)
    fprintf(f, "    for(int i=0; i<4; i++) {\n");
    fprintf(f, "        double sum = 0.0;\n");
    fprintf(f, "        for(int k=0; k<8; k++) sum += H1[i*8+k] * W2[k];\n");
    fprintf(f, "        Y[i] = sum + b2[0];\n");
    fprintf(f, "    }\n\n");

    fprintf(f, "    printf(\"=== Standalone Ghost-Weight Inference ===\\n\");\n");
    fprintf(f, "    for(int i=0; i<4; i++) printf(\"Input: [%%.0f, %%.0f] | Output: %%.4f\\n\", X[i*2], X[i*2+1], Y[i]);\n");
    fprintf(f, "    return 0;\n");
    fprintf(f, "}\n");

    fclose(f);

    basis_tensor_free(W1); basis_tensor_free(b1);
    basis_tensor_free(W2); basis_tensor_free(b2);

    printf("✅ Bundling complete. 'standalone_model.c' is ready.\n");
    return 0;
}
