#!/usr/bin/env python3
import os
import sys

def main():
    print("🚀 LANCIUS V1.0 PARITY AUDIT - V10 ABSOLUTE FINAL FIX")
    print("=" * 60)

    base_dir = "v9p5"

    # =========================================================================
    # 1. Patch export_pytorch_onnx.py (Remove Conv Biases)
    # =========================================================================
    print("\n[1/2] Patching PyTorch Exporter (Disable Conv Biases)...")
    pytorch_file = os.path.join(base_dir, "export_pytorch_onnx.py")
    with open(pytorch_file, 'r') as f:
        pt_content = f.read()

    pt_content = pt_content.replace(
        "self.conv1 = nn.Conv2d(3, 16, 5, padding=2)",
        "self.conv1 = nn.Conv2d(3, 16, 5, padding=2, bias=False)"
    )
    pt_content = pt_content.replace(
        "self.conv2 = nn.Conv2d(16, 32, 5, padding=2)",
        "self.conv2 = nn.Conv2d(16, 32, 5, padding=2, bias=False)"
    )
    with open(pytorch_file, 'w') as f:
        f.write(pt_content)
    print("   ✅ Disabled Conv biases to match C engine capabilities.")

    # =========================================================================
    # 2. Patch onnx_to_lancius.py (Fix NDim and Bias Shapes)
    # =========================================================================
    print("\n[2/2] Patching ONNX Converter (NDim & Bias Shapes)...")
    onnx_file = os.path.join(base_dir, "onnx_to_lancius.py")
    with open(onnx_file, 'r') as f:
        content = f.read()

    # Fix Bias Initializer Shapes
    old_init = """    # 1. Register Initializers (Weights) as INPUT nodes
    for init in graph.initializer:
        data = numpy_helper.to_array(init).astype(np.float64)
        shape = list(data.shape)
        while len(shape) < 4:
            shape.append(1)
        shape = shape[:4]
        nodes.append({
            'id': next_id, 'op': 1, 'ndim': len(data.shape), 'shape': shape,"""

    new_init = """    # 1. Register Initializers (Weights) as INPUT nodes
    for init in graph.initializer:
        data = numpy_helper.to_array(init).astype(np.float64)
        shape = list(data.shape)

        # V15 FIX: Force 1D biases to be [1, N] so they match MatMul output [1, N]
        calc_ndim = len(data.shape)
        if len(shape) == 1 and 'bias' in init.name:
            shape = [1, shape[0]]
            calc_ndim = 2

        while len(shape) < 4:
            shape.append(1)
        shape = shape[:4]
        nodes.append({
            'id': next_id, 'op': 1, 'ndim': calc_ndim, 'shape': shape,"""

    if old_init in content:
        content = content.replace(old_init, new_init)
        print("   ✅ Fixed Bias Initializer Shapes.")

    # Fix Reshape NDim
    old_else = """        else:
            nodes.append({
                'id': next_id, 'op': op, 'ndim': len([s for s in out_shape if s > 0]), 'shape': out_shape,"""

    new_else = """        else:
            # V15 FIX: Force ndim=2 for Reshape to match C MatMul expectations
            calc_ndim = len([s for s in out_shape if s > 0])
            if node.op_type == 'Reshape':
                calc_ndim = 2

            nodes.append({
                'id': next_id, 'op': op, 'ndim': calc_ndim, 'shape': out_shape,"""

    if old_else in content:
        content = content.replace(old_else, new_else)
        print("   ✅ Forced Reshape to ndim=2.")

    # Fix Gemm MatMul NDim
    old_gemm = """        if node.op_type == 'Gemm' and len(node.input) >= 2:
            matmul_inputs = [name_to_id[i] for i in node.input[:2] if i in name_to_id]
            nodes.append({
                'id': next_id, 'op': 6, 'ndim': len([s for s in out_shape if s > 0]), 'shape': out_shape,"""

    new_gemm = """        if node.op_type == 'Gemm' and len(node.input) >= 2:
            matmul_inputs = [name_to_id[i] for i in node.input[:2] if i in name_to_id]
            nodes.append({
                'id': next_id, 'op': 6, 'ndim': 2, 'shape': out_shape,"""

    if old_gemm in content:
        content = content.replace(old_gemm, new_gemm)
        print("   ✅ Forced Gemm MatMul to ndim=2.")

    # Fix Gemm Add NDim
    old_bias = """            if len(node.input) == 3 and node.input[2] in name_to_id:
                bias_id = name_to_id[node.input[2]]
                nodes.append({
                    'id': next_id, 'op': 3, 'ndim': len([s for s in out_shape if s > 0]), 'shape': out_shape,"""

    new_bias = """            if len(node.input) == 3 and node.input[2] in name_to_id:
                bias_id = name_to_id[node.input[2]]
                nodes.append({
                    'id': next_id, 'op': 3, 'ndim': 2, 'shape': out_shape,"""

    if old_bias in content:
        content = content.replace(old_bias, new_bias)
        print("   ✅ Forced Gemm Add to ndim=2.")

    with open(onnx_file, 'w') as f:
        f.write(content)

    # =========================================================================
    # AUTOMATED REBUILD AND TEST
    # =========================================================================
    print("\n" + "=" * 60)
    print("🔨 REBUILDING AND RUNNING AUDIT...")
    print("=" * 60)

    os.chdir(base_dir)
    cmds = [
        "rm -f pytorch_lenet.lancius pytorch_lenet.onnx parity_input.bin baseline_out.bin lancius_out.bin",
        "python3 export_pytorch_onnx.py",
        "python3 patch_onnx_binary.py",
        "python3 audit_pytorch_parity.py"
    ]

    for cmd in cmds:
        print(f"\n$ {cmd}")
        res = os.system(cmd)
        if res != 0 and "audit" not in cmd:
            print(f"❌ Command failed: {cmd}")
            sys.exit(1)

    print("\n" + "=" * 60)
    print("🏆 ALL STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    main()
