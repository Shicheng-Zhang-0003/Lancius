#!/usr/bin/env python3
"""Export a Lancius v2 binary model (.lancius) to ONNX.

Inference-subset only, fail-loud by design:
  supported Lancius ops -> ONNX:
    INPUT/CONST (weights -> initializer, weightless -> graph input)
    ADD, SUB, MUL, MATMUL, RELU, TANH,
    SOFTMAX (axis=1), SUM (ReduceSum all, keepdims),
    SUM_AXIS0/1, TRANSPOSE ([1,0]), PERMUTE, FLATTEN, RESHAPE,
    CONV2D (+meta kh,kw,stride,pad), MAXPOOL2D, MSE (Sub/Mul/ReduceMean)
  everything else (training BWD, CE, ATTENTION/GQA/NORM/ROPE/SWIGLU/GELU,
  KV_CACHE, EMBEDDING, BROADCAST-exotic, BATCHED, FUSED) raises.

Numerics: Lancius is FP64-first. ONNX output uses FLOAT (float32) for
max runtime compat; values are downcast with a stderr warning when the
cast is lossy. INT8 weights are dequantized (w*scale) on export.

Usage:
  python3 export_lancius_onnx.py in.lancius out.onnx
"""
import os
import struct
import sys
import zlib
import numpy as np

MAGIC_V2 = 0x32434E41
VERSION_V2 = 2
HEADER_FMT = '<8IQ2I'
NODE_FMT = '<IIB4QId4I4I3BdQ'
NODE_SIZE = struct.calcsize(NODE_FMT)
assert NODE_SIZE == 104, NODE_SIZE

DTYPE_ELEM = {0: 8, 1: 1, 2: 4, 3: 4}

OP_NAMES = {
    0: 'NOP', 1: 'INPUT', 2: 'CONST', 3: 'ADD', 4: 'SUB', 5: 'MUL',
    6: 'MATMUL', 7: 'RELU', 8: 'SOFTMAX', 9: 'SUM', 10: 'BROADCAST',
    11: 'TRANSPOSE', 12: 'RELU_BWD', 13: 'SOFTMAX_BWD', 14: 'SUM_AXIS0',
    15: 'SUM_AXIS1', 16: 'CONV2D', 17: 'MAXPOOL2D', 18: 'FLATTEN',
    19: 'CONV2D_RELU_FUSED', 20: 'CROSS_ENTROPY', 21: 'CROSS_ENTROPY_BWD',
    22: 'PERMUTE', 23: 'MATMUL_BATCHED', 24: 'CONV2D_BWD', 25: 'CONV2D_BWD_W',
    26: 'MAXPOOL2D_BWD', 27: 'RESHAPE', 28: 'EMBEDDING', 29: 'LAYERNORM',
    30: 'GELU', 31: 'ROPE', 32: 'ATTENTION', 33: 'KV_CACHE_READ',
    34: 'KV_CACHE_WRITE', 35: 'RMSNORM', 36: 'SWIGLU', 37: 'GQA',
    38: 'TANH', 39: 'TANH_BWD', 40: 'MSE', 41: 'MSE_BWD',
}


def parse_lancius(path, allow_legacy=False):
    # Despot truth: whole-file read was uncapped (OOM on GB files).
    try:
        if os.path.getsize(path) > 2 * 1024 * 1024 * 1024:
            raise ValueError(f"{path}: file exceeds 2GB cap; refusing parse.")
    except OSError as e:
        raise ValueError(f"{path}: cannot stat: {e}.") from e
    with open(path, 'rb') as f:
        blob = f.read()
    if len(blob) < 48:
        raise ValueError(f"{path}: file too small ({len(blob)} bytes)")
    hdr = struct.unpack(HEADER_FMT, blob[:48])
    magic, version, flags, node_count, tensor_count, attr_count, hsize, r0 = hdr[:8]
    woff, checksum, r1 = hdr[8], hdr[9], hdr[10]
    if magic != MAGIC_V2:
        raise ValueError(f"{path}: bad magic 0x{magic:08X} (want 0x{MAGIC_V2:08X})")
    if version != VERSION_V2:
        raise ValueError(f"{path}: bad version {version}")
    if hsize != 48:
        raise ValueError(f"{path}: bad header_size {hsize}")
    # Despot truth: flags/counters/reserved were ignored while C rejects them.
    if not (flags & 0x1):
        raise ValueError(f"{path}: missing little-endian flag (flags=0x{flags:08X})")
    if flags & 0x4:
        raise ValueError(f"{path}: EXTERNAL_WEIGHTS reserved flag set; refusing.")
    if attr_count != 0 or woff != 0 or r0 != 0 or r1 != 0:
        raise ValueError(f"{path}: reserved header fields must be 0 (attr={attr_count} woff={woff} r0={r0} r1={r1})")
    if node_count > 1000000:
        raise ValueError(f"{path}: node_count {node_count} exceeds 1M cap.")
    body = blob[48:]
    if checksum != 0:
        calc = zlib.crc32(body) & 0xFFFFFFFF
        if calc != checksum:
            raise ValueError(f"{path}: CRC32 mismatch header={checksum:08x} calc={calc:08x}")
    elif not allow_legacy:
        # Despot truth: zero-4-bytes bypassed integrity in this path (was: WARN
        # + continue, while C rejects by default).
        raise ValueError(f"{path}: checksum==0 legacy unverified file; refusing (pass allow_legacy=True to override).")
    nodes = []
    off = 0
    for _ in range(node_count):
        if off + NODE_SIZE > len(body):
            raise ValueError(f"{path}: truncated node header at offset {off}")
        f = struct.unpack(NODE_FMT, body[off:off + NODE_SIZE])
        off += NODE_SIZE
        nid, op, ndim = f[0], f[1], f[2]
        shape = [f[3], f[4], f[5], f[6]]
        in_count = f[7]
        attr = f[8]
        meta = [f[9], f[10], f[11], f[12]]
        axes = [f[13], f[14], f[15], f[16]]
        _fl0, dtype, has_w = f[17], f[18], f[19]
        scale = f[20]
        w_elems = f[21]
        if ndim < 1 or ndim > 4:
            raise ValueError(f"node {nid}: bad ndim {ndim}")
        if in_count > 16:
            raise ValueError(f"node {nid}: bad input_count {in_count}")
        inputs = []
        for _i in range(in_count):
            if off + 4 > len(body):
                raise ValueError(f"node {nid}: truncated inputs")
            (iid,) = struct.unpack('<I', body[off:off + 4])
            off += 4
            inputs.append(iid)
        weights = None
        if has_w and w_elems > 0:
            if dtype not in DTYPE_ELEM:
                raise ValueError(f"node {nid}: bad dtype {dtype}")
            nbytes = w_elems * DTYPE_ELEM[dtype]
            if off + nbytes > len(body):
                raise ValueError(f"node {nid}: truncated weights")
            raw = body[off:off + nbytes]
            off += nbytes
            if dtype == 0:
                weights = np.frombuffer(raw, dtype=np.float64).copy()
            elif dtype == 1:
                weights = np.frombuffer(raw, dtype=np.int8).copy()
            elif dtype == 2:
                weights = np.frombuffer(raw, dtype=np.float32).copy()
            else:
                weights = np.frombuffer(raw, dtype=np.int32).copy()
        nodes.append({
            'id': nid, 'op': op, 'ndim': ndim, 'shape': shape,
            'inputs': inputs, 'attr': attr, 'meta': meta, 'axes': axes,
            'dtype': dtype, 'has_w': has_w, 'scale': scale,
            'weights': weights,
        })
    return {'flags': flags, 'node_count': node_count, 'checksum': checksum,
            'nodes': nodes}


def lancius_shape(n):
    """Unpad 4-wide shape to true rank."""
    return [int(s) for s in n['shape'][:n['ndim']]]


def convert(parsed, out_path):
    import onnx
    from onnx import helper, TensorProto
    nodes = parsed['nodes']
    by_id = {n['id']: n for n in nodes}
    if len(by_id) != len(nodes):
        raise ValueError("duplicate node id in file")

    initializers = []
    graph_nodes = []
    graph_inputs = []
    tensor_elemtype = {}  # onnx name -> TensorProto type (all FLOAT here)

    def tname(lid):
        return f"t{lid}"

    def add_initializer(name, arr_f32):
        from onnx import numpy_helper
        initializers.append(numpy_helper.from_array(
            np.ascontiguousarray(arr_f32, dtype=np.float32), name=name))

    def add_initializer_i64(name, arr_i64):
        from onnx import numpy_helper
        initializers.append(numpy_helper.from_array(
            np.ascontiguousarray(arr_i64, dtype=np.int64), name=name))

    def weight_as_f32(n):
        w = n['weights']
        if w is None:
            raise ValueError(f"node {n['id']}: expected weights")
        if n['dtype'] == 0:
            a = w.astype(np.float32)
            if not np.allclose(a.astype(np.float64), w, rtol=1e-6, atol=1e-6):
                print(f"WARN: node {n['id']}: FP64->FP32 downcast lossy", file=sys.stderr)
            return a
        if n['dtype'] == 1:
            return (w.astype(np.float32) * float(n['scale']))
        if n['dtype'] == 2:
            return w.astype(np.float32)
        raise ValueError(f"node {n['id']}: INT32 weights not exportable")

    # Register tensors
    for n in nodes:
        op = n['op']
        name = tname(n['id'])
        if op == 1 and n['has_w'] and n['weights'] is not None:
            arr = weight_as_f32(n).reshape(lancius_shape(n))
            add_initializer(name, arr)
            tensor_elemtype[name] = TensorProto.FLOAT
        elif op == 1 and not n['has_w']:
            shp = lancius_shape(n)
            graph_inputs.append(helper.make_tensor_value_info(name, TensorProto.FLOAT, shp))
            tensor_elemtype[name] = TensorProto.FLOAT
        elif op == 2:
            shp = lancius_shape(n)
            arr = np.full(shp if shp else [1], float(n['attr']), dtype=np.float32)
            add_initializer(name, arr)
            tensor_elemtype[name] = TensorProto.FLOAT
        elif op == 0:
            pass  # NOP: no tensor
        else:
            tensor_elemtype[name] = TensorProto.FLOAT

    def inp(lid):
        if lid not in by_id:
            raise ValueError(f"forward/missing reference to id {lid}")
        return tname(lid)

    for n in nodes:
        op, nid = n['op'], n['id']
        nm = OP_NAMES.get(op, f"OP{op}")
        out = tname(nid)
        ins = [inp(i) for i in n['inputs']]
        shp = lancius_shape(n)

        if op in (0, 1, 2):
            continue
        elif op == 3:
            graph_nodes.append(helper.make_node('Add', ins, [out], name=f"Add_{nid}"))
        elif op == 4:
            graph_nodes.append(helper.make_node('Sub', ins, [out], name=f"Sub_{nid}"))
        elif op == 5:
            graph_nodes.append(helper.make_node('Mul', ins, [out], name=f"Mul_{nid}"))
        elif op == 6:
            if len(ins) != 2:
                raise ValueError(f"MATMUL {nid}: need 2 inputs")
            graph_nodes.append(helper.make_node('MatMul', ins, [out], name=f"MatMul_{nid}"))
        elif op == 7:
            graph_nodes.append(helper.make_node('Relu', ins, [out], name=f"Relu_{nid}"))
        elif op == 38:
            graph_nodes.append(helper.make_node('Tanh', ins, [out], name=f"Tanh_{nid}"))
        elif op == 8:
            graph_nodes.append(helper.make_node('Softmax', ins, [out], axis=1, name=f"Softmax_{nid}"))
        elif op == 9:
            graph_nodes.append(helper.make_node('ReduceSum', ins, [out], keepdims=1, name=f"Sum_{nid}"))
        elif op == 14:
            ax_name = f"{out}__axes"
            add_initializer_i64(ax_name, np.array([0], dtype=np.int64))
            graph_nodes.append(helper.make_node('ReduceSum', [ins[0], ax_name], [out], keepdims=1, name=f"Sum0_{nid}"))
        elif op == 15:
            ax_name = f"{out}__axes"
            add_initializer_i64(ax_name, np.array([1], dtype=np.int64))
            graph_nodes.append(helper.make_node('ReduceSum', [ins[0], ax_name], [out], keepdims=1, name=f"Sum1_{nid}"))
        elif op == 11:
            graph_nodes.append(helper.make_node('Transpose', ins, [out], perm=[1, 0], name=f"Tr_{nid}"))
        elif op == 22:
            perm = [int(a) for a in n['axes'][:n['ndim']]]
            if sorted(perm) != list(range(n['ndim'])):
                raise ValueError(f"PERMUTE {nid}: bad perm {perm}")
            graph_nodes.append(helper.make_node('Transpose', ins, [out], perm=perm, name=f"Perm_{nid}"))
        elif op == 18:
            graph_nodes.append(helper.make_node('Flatten', ins, [out], axis=1, name=f"Flat_{nid}"))
        elif op == 27:
            shape_name = f"{out}__shape"
            add_initializer_i64(shape_name, np.array(shp, dtype=np.int64))
            graph_nodes.append(helper.make_node('Reshape', [ins[0], shape_name], [out], name=f"Reshape_{nid}"))
        elif op == 10:
            # Explicit broadcast -> Expand to target shape
            shape_name = f"{out}__shape"
            add_initializer_i64(shape_name, np.array(shp, dtype=np.int64))
            graph_nodes.append(helper.make_node('Expand', [ins[0], shape_name], [out], name=f"Expand_{nid}"))
        elif op == 16:
            kh, kw, s, p = (int(x) for x in n['meta'])
            if (kh <= 0 or kw <= 0) and len(n['inputs']) >= 2 and n['inputs'][1] in by_id:
                # Legacy files may carry zeroed kh/kw (redundant: the C runtime
                # and IR builder derive the kernel from the weight shape).
                # Recover from the weight tensor [Cout,Cin,Kh,Kw].
                wgt = by_id[n['inputs'][1]]
                ws = lancius_shape(wgt)
                if len(ws) == 4 and ws[2] > 0 and ws[3] > 0:
                    kh, kw = int(ws[2]), int(ws[3])
            if kh <= 0 or kw <= 0 or s <= 0:
                raise ValueError(f"CONV2D {nid}: bad meta {n['meta']}")
            graph_nodes.append(helper.make_node(
                'Conv', ins, [out], kernel_shape=[kh, kw],
                strides=[s, s], pads=[p, p, p, p], name=f"Conv_{nid}"))
        elif op == 17:
            kh, kw, s, _p = (int(x) for x in n['meta'])
            if kh != kw or kh <= 0 or s <= 0:
                raise ValueError(f"MAXPOOL {nid}: need square kernel, got {n['meta']}")
            graph_nodes.append(helper.make_node(
                'MaxPool', ins, [out], kernel_shape=[kh, kh],
                strides=[s, s], name=f"Pool_{nid}"))
        elif op == 40:
            if len(ins) != 2:
                raise ValueError(f"MSE {nid}: need 2 inputs")
            sub_o, sq_o = f"{out}__sub", f"{out}__sq"
            graph_nodes.append(helper.make_node('Sub', ins, [sub_o], name=f"MSEsub_{nid}"))
            graph_nodes.append(helper.make_node('Mul', [sub_o, sub_o], [sq_o], name=f"MSEsq_{nid}"))
            graph_nodes.append(helper.make_node('ReduceMean', [sq_o], [out], keepdims=1, name=f"MSE_{nid}"))
        else:
            raise ValueError(
                f"lancius op {nm} (id {nid}) has no ONNX mapping in this exporter. "
                f"Supported: ADD/SUB/MUL/MATMUL/RELU/TANH/SOFTMAX/SUM(+AXIS)/"
                f"TRANSPOSE/PERMUTE/FLATTEN/RESHAPE/BROADCAST/CONV2D/MAXPOOL/MSE.")

    # Output: last computed (non INPUT/CONST/NOP) tensor; prefer CE-logits convention
    out_node = None
    for n in nodes:
        if n['op'] == 20 and n['inputs']:
            out_node = by_id.get(n['inputs'][0])
            break
    if out_node is None:
        for n in reversed(nodes):
            if n['op'] not in (0, 1, 2):
                out_node = n
                break
    if out_node is None:
        raise ValueError("no computable output node found")
    out_name = tname(out_node['id'])
    out_shape = lancius_shape(out_node)
    graph_out = [helper.make_tensor_value_info(out_name, TensorProto.FLOAT, out_shape)]

    graph = helper.make_graph(graph_nodes, 'lancius_export', graph_inputs,
                              graph_out, initializer=initializers)
    model = helper.make_model(graph, producer_name='lancius_export',
                              opset_imports=[helper.make_opsetid('', 17)])
    model.ir_version = 8
    import onnx as _onnx
    _onnx.checker.check_model(model)
    _onnx.save(model, out_path)
    print(f"OK: {len(nodes)} lancius nodes -> {len(graph_nodes)} ONNX nodes -> {out_path} (output {out_name} {out_shape})")


def main():
    args = [a for a in sys.argv[1:] if a != '--allow-legacy']
    allow_legacy = len(args) != len(sys.argv[1:])
    if len(args) != 2:
        print(f"usage: {sys.argv[0]} in.lancius out.onnx [--allow-legacy]", file=sys.stderr)
        return 2
    try:
        parsed = parse_lancius(args[0], allow_legacy=allow_legacy)
        convert(parsed, args[1])
    except Exception as e:
        print(f"FAIL: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
