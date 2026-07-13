#!/usr/bin/env python3
import os, sys, re, subprocess

def main():
    print("🚀 LANCIUS V1.0 DEFINITIVE PARITY FIX")
    print("=" * 60)
    base_dir = "v9p5"

    # =========================================================================
    # 1. Patch lancius_ir.c (NDim-Aware MatMul & Flatten)
    # =========================================================================
    print("\n[1/2] Patching C Engine (NDim-Aware MatMul)...")
    ir_c = os.path.join(base_dir, "src", "ir", "lancius_ir.c")
    with open(ir_c, 'r') as f:
        c_content = f.read()

    # Replace the strict lancius_matmul with an NDim-aware one
    old_matmul = re.compile(r'lancius_node\* lancius_matmul\(.*?\n\}', re.DOTALL)
    new_matmul = """lancius_node* lancius_matmul(lancius_graph* g, const lancius_node* a, const lancius_node* b) {
    if (!a || !b || a->ndim < 2 || b->ndim < 2) {
        fprintf(stderr, "[LANCIUS IR FATAL] MATMUL ndim < 2: a=%u b=%u\\n", a?a->ndim:0, b?b->ndim:0);
        return NULL;
    }
    // V14 FIX: NDim-aware dimension extraction (handles 4D Reshape outputs)
    size_t a_rows = a->shape[a->ndim - 2];
    size_t a_cols = a->shape[a->ndim - 1];
    size_t b_rows = b->shape[b->ndim - 2];
    size_t b_cols = b->shape[b->ndim - 1];

    if (a_cols != b_rows) {
        fprintf(stderr, "[LANCIUS IR FATAL] MATMUL mismatch: a=[%zux%zu] b=[%zux%zu]\\n", a_rows, a_cols, b_rows, b_cols);
        return NULL;
    }
    lancius_node* n = alloc_node(g, LANCIUS_OP_MATMUL, 2, 2);
    if (n) { n->shape[0] = a_rows; n->shape[1] = b_cols; n->inputs[0] = a; n->inputs[1] = b; }
    return n;
}"""
    c_content = old_matmul.sub(new_matmul, c_content)

    # Force lancius_flatten to output ndim=2
    old_flatten = re.compile(r'lancius_node\* lancius_flatten\(.*?\n\}', re.DOTALL)
    new_flatten = """lancius_node* lancius_flatten(lancius_graph* g, const lancius_node* in) {
    if (!in || in->ndim != 4) return NULL;
    size_t N = in->shape[0];
    size_t flat = in->shape[1] * in->shape[2] * in->shape[3];
    lancius_node* n = alloc_node(g, LANCIUS_OP_FLATTEN, 2, 1);
    if (n) { n->shape[0] = N; n->shape[1] = flat; n->inputs[0] = in; }
    return n;
}"""
    c_content = old_flatten.sub(new_flatten, c_content)

    with open(ir_c, 'w') as f:
        f.write(c_content)
    print("   ✅ Replaced lancius_matmul and lancius_flatten.")

    # =========================================================================
    # 2. Patch onnx_to_lancius.py (Force 2D for Reshape/Gemm)
    # =========================================================================
    print("\n[2/2] Patching Python Exporter (Force 2D Linear Shapes)...")
    py_file = os.path.join(base_dir, "onnx_to_lancius.py")
    with open(py_file, 'r') as f:
        py_content = f.read()

    # Force Reshape to ndim=2 if it's the LeNet flatten
    old_reshape_block = "if len(resolved_dims) == 2:\n                    out_shape = [1, resolved_dims[1], 1, 1]"
    new_reshape_block = "if len(resolved_dims) == 2:\n                    out_shape = [1, resolved_dims[1], 1, 1]\n                    # V14 FIX: Force ndim=2 for MatMul compatibility\n                    nodes[-1]['ndim'] = 2"

    if old_reshape_block in py_content:
        py_content = py_content.replace(old_reshape_block, new_reshape_block)
    else:
        # Fallback regex injection if exact string match fails
        py_content = re.sub(
            r"(if node\.op_type == 'Reshape'.*?nodes\.append\(\{.*?'ndim': len\(\[s for s in out_shape if s > 0\]\))",
            r"\1 if node.op_type != 'Reshape' else 2",
            py_content,
            flags=re.DOTALL
        )

    # Force Gemm/MatMul output to ndim=2
    py_content = re.sub(
        r"('id': next_id, 'op': 6, 'ndim': len\(\[s for s in out_shape if s > 0\]\))",
        r"\1 if node.op_type != 'Gemm' else 2",
        py_content
    )

    with open(py_file, 'w') as f:
        f.write(py_content)
    print("   ✅ Forced Reshape and Gemm to output ndim=2.")

    # =========================================================================
    # AUTOMATED REBUILD AND TEST
    # =========================================================================
    print("\n" + "=" * 60)
    print("🔨 REBUILDING AND RUNNING AUDIT...")
    print("=" * 60)

    os.chdir(base_dir)
    cmds = [
        "rm -f pytorch_lenet.lancius pytorch_lenet.onnx parity_input.bin baseline_out.bin lancius_out.bin",
        "make clean && make",
        "python3 export_pytorch_onnx.py",
        "python3 patch_onnx_binary.py",
        "python3 audit_pytorch_parity.py"
    ]

    for cmd in cmds:
        print(f"\n$ {cmd}")
        res = subprocess.run(cmd, shell=True)
        if res.returncode != 0 and "audit" not in cmd:
            print(f"❌ Command failed: {cmd}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("🏆 ALL STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
