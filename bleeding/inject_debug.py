#!/usr/bin/env python3
import os
import sys
import subprocess

def main():
    print("🔍 LANCIUS V1.0 DEEP DEBUG INJECTOR")
    print("=" * 60)

    base_dir = "v9p5"

    # 1. Patch onnx_to_lancius.py
    py_file = os.path.join(base_dir, "onnx_to_lancius.py")
    with open(py_file, 'r') as f:
        py_content = f.read()

    debug_py = """        print(f"  [DEBUG PY] Node: {node.op_type} | Inputs: {list(node.input)} | Out: {node.output[0]} | Shape: {out_shape}")\n"""

    if "DEBUG PY" not in py_content:
        py_content = py_content.replace("        meta = [0, 0, 0, 0]", debug_py + "        meta = [0, 0, 0, 0]")
        with open(py_file, 'w') as f:
            f.write(py_content)
        print("✅ Injected debug prints into onnx_to_lancius.py")

    # 2. Patch lancius_serialize.c
    c_file = os.path.join(base_dir, "src", "core", "lancius_serialize.c")
    with open(c_file, 'r') as f:
        c_content = f.read()

    debug_c = """        if (n) {
            printf("  [DEBUG C] Loaded Node %u (Op %u) | Shape: [%zux%zux%zux%zu] | Inputs: ", id, op, n->shape[0], n->shape[1], n->shape[2], n->shape[3]);
            for(uint32_t k=0; k<input_count; k++) printf("%u ", in_ids[k]);
            printf("\\n");
        }"""

    if "DEBUG C" not in c_content:
        old_block = 'else { fprintf(stderr, "[LANCIUS SERIAL WARN] Unknown op %u during load, skipping\\n", op); }\nif (n) {'
        new_block = 'else { fprintf(stderr, "[LANCIUS SERIAL WARN] Unknown op %u during load, skipping\\n", op); }\n' + debug_c + '\nif (n) {'
        if old_block in c_content:
            c_content = c_content.replace(old_block, new_block)
            with open(c_file, 'w') as f:
                f.write(c_content)
            print("✅ Injected debug prints into lancius_serialize.c")
        else:
            print("⚠️ Could not find injection point in lancius_serialize.c")

    # 3. Patch parity_runner.c
    pr_file = os.path.join(base_dir, "examples", "parity_runner.c")
    with open(pr_file, 'r') as f:
        pr_content = f.read()

    debug_pr = """    printf("  [DEBUG C] Graph Topology Dump:\\n");
    for(uint32_t i=0; i<g->node_count; i++) {
        lancius_node* n = g->nodes[i];
        printf("    Node %u: Op=%d Shape=[%zux%zux%zux%zu] Inputs=[", n->id, n->op, n->shape[0], n->shape[1], n->shape[2], n->shape[3]);
        for(uint32_t k=0; k<n->input_count; k++) printf(" %u", n->inputs[k] ? n->inputs[k]->id : 9999);
        printf(" ]\\n");
    }\n"""

    if "DEBUG C] Graph Topology" not in pr_content:
        pr_content = pr_content.replace("if(!in_node || !out_node) {", debug_pr + "\nif(!in_node || !out_node) {")
        with open(pr_file, 'w') as f:
            f.write(pr_content)
        print("✅ Injected debug prints into parity_runner.c")

    print("\n" + "=" * 60)
    print("🔨 REBUILDING AND RUNNING WITH DEBUG OUTPUT...")
    print("=" * 60)

    os.chdir(base_dir)
    cmds = [
        "rm -f pytorch_lenet.lancius",
        "make clean && make",
        "python3 audit_pytorch_parity.py"
    ]

    for cmd in cmds:
        print(f"\n$ {cmd}")
        res = subprocess.run(cmd, shell=True)
        if res.returncode != 0 and "audit_pytorch_parity" not in cmd:
            print(f"❌ Command failed: {cmd}")
            sys.exit(1)

if __name__ == "__main__":
    main()
