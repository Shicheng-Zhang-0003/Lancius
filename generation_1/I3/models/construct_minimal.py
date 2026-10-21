#!/usr/bin/env python3
"""Minimal v2 model constructor for debugging."""
import struct, zlib, sys

MAGIC_V2 = 0x32434E41
VERSION_V2 = 2
FLAGS = 0x00000003
HEADER_SIZE = 48
DTYPE_FP64 = 0
OP_INPUT = 1
OP_MATMUL = 6
OP_ADD = 3
OP_RELU = 7

def crc32(data):
    return zlib.crc32(data) & 0xFFFFFFFF

def build_minimal(path):
    nodes = [
        # id, op, shape, ndim, inputs, dtype, weights, meta, axes, scale, attr
        (0, OP_INPUT, [10, 20, 0, 0], 2, [], DTYPE_FP64, None, [0,0,0,0], [0,0,0,0], 0.0, 0.0),
        (1, OP_INPUT, [20, 5, 0, 0], 2, [], DTYPE_FP64, [0.1]*100, [0,0,0,0], [0,0,0,0], 0.0, 0.0),
        (2, OP_MATMUL, [10, 5, 0, 0], 2, [0, 1], DTYPE_FP64, None, [0,0,0,0], [0,0,0,0], 0.0, 0.0),
        (3, OP_RELU, [10, 5, 0, 0], 2, [2], DTYPE_FP64, None, [0,0,0,0], [0,0,0,0], 0.0, 0.0),
    ]
    node_count = len(nodes)
    
    body = bytearray()
    for (nid, op, shape, ndim, inputs, dtype, weights, meta, axes, scale, attr) in nodes:
        rec = struct.pack("<II", nid, op)
        rec += struct.pack("<B", ndim)
        for s in range(4):
            rec += struct.pack("<Q", shape[s])
        rec += struct.pack("<I", len(inputs))
        rec += struct.pack("<d", attr)
        for m in range(4):
            rec += struct.pack("<I", meta[m])
        for a in range(4):
            rec += struct.pack("<I", axes[a])
        rec += struct.pack("<BBB", 0, dtype, 1 if weights is not None else 0)
        rec += struct.pack("<d", scale)
        we = len(weights) if weights else 0
        rec += struct.pack("<Q", we)
        assert len(rec) == 104
        body.extend(rec)
        for in_id in inputs:
            body.extend(struct.pack("<I", in_id))
        if weights:
            for w in weights:
                body.extend(struct.pack("<d", w))
    
    header = struct.pack("<II", MAGIC_V2, VERSION_V2)
    header += struct.pack("<I", FLAGS)
    header += struct.pack("<II", node_count, node_count)
    header += struct.pack("<II", 0, HEADER_SIZE)
    header += struct.pack("<I", 0)
    header += struct.pack("<Q", 0)
    header += struct.pack("<II", 0, 0)
    assert len(header) == HEADER_SIZE
    
    body_crc = crc32(bytes(body))
    if body_crc == 0:
        body_crc = 1
    header = header[:40] + struct.pack("<I", body_crc) + header[44:]
    
    with open(path, "wb") as f:
        f.write(header + bytes(body))
    
    print(f"Wrote {path}: {len(header)+len(body)} bytes, crc=0x{body_crc:08x}")

if __name__ == "__main__":
    build_minimal(sys.argv[1] if len(sys.argv) > 1 else "/tmp/minimal.lancius")
