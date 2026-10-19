#!/usr/bin/env python3
"""
Lancius v2 model constructor — builds .lancius model files from scratch.

Architecture: input_dim -> hidden1 -> hidden2 -> n_classes (tanh/tanh)
All weights FP64, initialized with He initialization.
Serializes to the frozen v2 binary format (48-byte header + node records + body CRC32).

Usage:
    python3 fabric/models/construct_model.py <out.lancius> <input_dim> <h1> <h2> <n_classes> [seed]
"""
import struct, sys, os, zlib, random

MAGIC_V2 = 0x32434E41
VERSION_V2 = 2
FLAGS = 0x00000003  # little_endian | static_graph
HEADER_SIZE = 48
DTYPE_FP64 = 0

# Opcodes (from lancius_model_format.h)
OP_INPUT = 1
OP_MATMUL = 6
OP_ADD = 3
OP_TANH = 38

def he_init(fan_in, fan_out, rng):
    """He initialization: uniform in [-sqrt(6/fan_in), sqrt(6/fan_in)]."""
    limit = (6.0 / fan_in) ** 0.5
    return [rng.uniform(-limit, limit) for _ in range(fan_in * fan_out)]

def crc32(data):
    return zlib.crc32(data) & 0xFFFFFFFF

def build_model(path, input_dim, h1, h2, n_classes, seed=20261007):
    rng = random.Random(seed)

    # ---- Weight matrices (FP64) ----
    W1 = he_init(input_dim, h1, rng)      # input_dim x h1
    B1 = [0.0] * h1                       # bias h1
    W2 = he_init(h1, h2, rng)             # h1 x h2
    B2 = [0.0] * h2                       # bias h2
    W3 = he_init(h2, n_classes, rng)      # h2 x n_classes
    B3 = [0.0] * n_classes                # bias n_classes

    # ---- Node definitions ----
    # id, op, shape, ndim, inputs, dtype, weight_data (or None)
    nodes = []

    def node(nid, op, shape, ndim, inputs, dtype, weights, meta, axes, scale, attr):
        return {
            "id": nid, "op": op, "shape": shape, "ndim": ndim,
            "inputs": inputs, "dtype": dtype, "weights": weights,
            "input_count": len(inputs),
            "meta": meta, "axes": axes, "scale": scale, "attr": attr
        }

    nodes.append(node(
        0, OP_INPUT, [1, input_dim, 0, 0], 2, [], DTYPE_FP64, None,
        [0, 0, 0, 0], [0, 0, 0, 0], 0.0, 0.0
    ))

    nodes.append(node(
        1, OP_INPUT, [input_dim, h1, 0, 0], 2, [], DTYPE_FP64, W1,
        [0, 0, 0, 0], [0, 0, 0, 0], 0.0, 0.0
    ))

    # Node 2: INPUT [1, h1] — B1
    nodes.append(node(
        2, OP_INPUT, [1, h1, 0, 0], 2, [], DTYPE_FP64, B1,
        [0, 0, 0, 0], [0, 0, 0, 0], 0.0, 0.0
    ))

    # Node 3: MATMUL [1, input_dim] x [input_dim, h1] -> [1, h1]
    nodes.append(node(
        3, OP_MATMUL, [1, h1, 0, 0], 2, [0, 1], DTYPE_FP64, None,
        [0, 0, 0, 0], [0, 0, 0, 0], 0.0, 0.0
    ))

    # Node 4: ADD [1, h1] + [1, h1] -> [1, h1]
    nodes.append(node(
        4, OP_ADD, [1, h1, 0, 0], 2, [3, 2], DTYPE_FP64, None,
        [0, 0, 0, 0], [0, 0, 0, 0], 0.0, 0.0
    ))

    # Node 5: TANH [1, h1] -> [1, h1]
    nodes.append(node(
        5, OP_TANH, [1, h1, 0, 0], 2, [4], DTYPE_FP64, None,
        [0, 0, 0, 0], [0, 0, 0, 0], 0.0, 0.0
    ))

    # Node 6: INPUT [h1, h2] — W2
    nodes.append(node(
        6, OP_INPUT, [h1, h2, 0, 0], 2, [], DTYPE_FP64, W2,
        [0, 0, 0, 0], [0, 0, 0, 0], 0.0, 0.0
    ))

    # Node 7: INPUT [1, h2] — B2
    nodes.append(node(
        7, OP_INPUT, [1, h2, 0, 0], 2, [], DTYPE_FP64, B2,
        [0, 0, 0, 0], [0, 0, 0, 0], 0.0, 0.0
    ))

    # Node 8: MATMUL [1, h1] x [h1, h2] -> [1, h2]
    nodes.append(node(
        8, OP_MATMUL, [1, h2, 0, 0], 2, [5, 6], DTYPE_FP64, None,
        [0, 0, 0, 0], [0, 0, 0, 0], 0.0, 0.0
    ))

    # Node 9: ADD [1, h2] + [1, h2] -> [1, h2]
    nodes.append(node(
        9, OP_ADD, [1, h2, 0, 0], 2, [8, 7], DTYPE_FP64, None,
        [0, 0, 0, 0], [0, 0, 0, 0], 0.0, 0.0
    ))

    # Node 10: TANH [1, h2] -> [1, h2]
    nodes.append(node(
        10, OP_TANH, [1, h2, 0, 0], 2, [9], DTYPE_FP64, None,
        [0, 0, 0, 0], [0, 0, 0, 0], 0.0, 0.0
    ))

    # Node 11: INPUT [h2, n_classes] — W3
    nodes.append(node(
        11, OP_INPUT, [h2, n_classes, 0, 0], 2, [], DTYPE_FP64, W3,
        [0, 0, 0, 0], [0, 0, 0, 0], 0.0, 0.0
    ))

    # Node 12: INPUT [1, n_classes] — B3
    nodes.append(node(
        12, OP_INPUT, [1, n_classes, 0, 0], 2, [], DTYPE_FP64, B3,
        [0, 0, 0, 0], [0, 0, 0, 0], 0.0, 0.0
    ))

    # Node 13: MATMUL [1, h2] x [h2, n_classes] -> [1, n_classes]
    nodes.append(node(
        13, OP_MATMUL, [1, n_classes, 0, 0], 2, [10, 11], DTYPE_FP64, None,
        [0, 0, 0, 0], [0, 0, 0, 0], 0.0, 0.0
    ))

    # Node 14: ADD [1, n_classes] + [1, n_classes] -> [1, n_classes] (output logits)
    nodes.append(node(
        14, OP_ADD, [1, n_classes, 0, 0], 2, [13, 12], DTYPE_FP64, None,
        [0, 0, 0, 0], [0, 0, 0, 0], 0.0, 0.0
    ))

    node_count = len(nodes)

    # ---- Serialize ----
    body = bytearray()

    for n in nodes:
        # v2_node packed struct (104 bytes):
        # uint32_t id, uint32_t op, uint8_t ndim, uint64_t shape[4],
        # uint32_t input_count, double attr, uint32_t meta[4],
        # uint32_t axes[4],
        # uint8_t flags, uint8_t dtype, uint8_t has_weights,
        # double scale, uint64_t weight_elems
        rec = struct.pack("<II", n["id"], n["op"])  # id, op
        rec += struct.pack("<B", n["ndim"])         # ndim
        for s in range(4):
            rec += struct.pack("<Q", n["shape"][s]) # shape[4] as uint64
        rec += struct.pack("<I", n["input_count"])  # input_count
        rec += struct.pack("<d", n["attr"])         # attr (double)
        for m in range(4):
            rec += struct.pack("<I", n["meta"][m])  # meta[4]
        for a in range(4):
            rec += struct.pack("<I", n["axes"][a])  # axes[4]
        rec += struct.pack("<B", 0)                 # flags
        rec += struct.pack("<B", n["dtype"])        # dtype
        has_w = 1 if n["weights"] is not None else 0
        rec += struct.pack("<B", has_w)             # has_weights
        rec += struct.pack("<d", n["scale"])        # scale
        weight_elems = len(n["weights"]) if n["weights"] else 0
        rec += struct.pack("<Q", weight_elems)      # weight_elems

        assert len(rec) == 104, f"node record size {len(rec)} != 104"

        body.extend(rec)

        # input IDs
        for in_id in n["inputs"]:
            body.extend(struct.pack("<I", in_id))

        # weight data (FP64, little-endian)
        if n["weights"] is not None:
            for w in n["weights"]:
                body.extend(struct.pack("<d", w))

    # ---- Header ----
    header = struct.pack("<II", MAGIC_V2, VERSION_V2)  # magic, version
    header += struct.pack("<I", FLAGS)                  # flags
    header += struct.pack("<II", node_count, node_count)  # node_count, tensor_count
    header += struct.pack("<II", 0, HEADER_SIZE)        # attribute_count, header_size
    header += struct.pack("<I", 0)                      # reserved0
    header += struct.pack("<Q", 0)                      # weight_block_offset
    header += struct.pack("<II", 0, 0)                  # checksum_crc32 placeholder, reserved1

    assert len(header) == HEADER_SIZE

    # ---- CRC32 over body ----
    body_crc = crc32(bytes(body))
    if body_crc == 0:
        body_crc = 1  # never emit 0 (means legacy/unverified)

    # Patch CRC into header
    header = header[:40] + struct.pack("<I", body_crc) + header[44:]

    # ---- Write ----
    out = header + bytes(body)
    with open(path, "wb") as f:
        f.write(out)

    print(f"constructed model: {path}")
    print(f"  arch: {input_dim} -> {h1} -> {h2} -> {n_classes}")
    print(f"  nodes: {node_count}")
    print(f"  size: {len(out)} bytes")
    print(f"  crc32: {body_crc:08x}")
    print(f"  seed: {seed}")

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("usage: construct_model.py <out.lancius> <input_dim> <h1> <h2> <n_classes> [seed]")
        sys.exit(1)
    out = sys.argv[1]
    input_dim = int(sys.argv[2])
    h1 = int(sys.argv[3])
    h2 = int(sys.argv[4])
    n_classes = int(sys.argv[5])
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 20261007
    build_model(out, input_dim, h1, h2, n_classes, seed)
