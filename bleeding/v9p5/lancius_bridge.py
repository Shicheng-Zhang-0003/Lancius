#!/usr/bin/env python3
import struct
import sys
import os
import numpy as np

# Magic Numbers
LANCIUS_MAGIC = 0x21434E41  # "LANC!" in little-endian
GGUF_MAGIC = 0x46554747     # "GGUF"
GGUF_VERSION = 3

# GGUF Tensor Types
GGML_TYPE_F32 = 0
GGML_TYPE_Q8_0 = 7

def read_lancius_tensors(path):
    """Parses the custom .lancius format and extracts weight tensors."""
    tensors = []
    with open(path, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        if magic != LANCIUS_MAGIC:
            print(f"❌ Invalid LANCIUS Magic Number! Expected 0x{LANCIUS_MAGIC:08X}, got 0x{magic:08X}")
            sys.exit(1)

        node_count = struct.unpack('<I', f.read(4))[0]
        print(f"📦 Found {node_count} nodes in {path}")

        for i in range(node_count):
            node_id = struct.unpack('<I', f.read(4))[0]
            op = struct.unpack('<I', f.read(4))[0]
            ndim = struct.unpack('<B', f.read(1))[0]
            shape = struct.unpack('<4Q', f.read(32))
            input_count = struct.unpack('<I', f.read(4))[0]

            # Skip inputs, attr_val, meta, axes
            f.read(input_count * 4)
            f.read(8)
            f.read(16)
            f.read(16)

            has_weights = struct.unpack('<B', f.read(1))[0]
            if has_weights:
                dtype_flag = struct.unpack('<B', f.read(1))[0]
                scale = struct.unpack('<d', f.read(8))[0]

                elems = 1
                for d in range(ndim): elems *= shape[d]

                # Name the tensor based on node ID and shape for GGUF visibility
                name = f"lancius_tensor_{node_id}"

                if dtype_flag == 1: # INT8
                    data = np.frombuffer(f.read(elems), dtype=np.int8)
                    tensors.append({'name': name, 'shape': shape[:ndim], 'data': data, 'type': GGML_TYPE_Q8_0, 'scale': scale})
                else: # FP64 (Convert to F32 for GGUF standard)
                    data = np.frombuffer(f.read(elems * 8), dtype=np.float64).astype(np.float32)
                    tensors.append({'name': name, 'shape': shape[:ndim], 'data': data, 'type': GGML_TYPE_F32, 'scale': 1.0})
    return tensors

def write_gguf(path, tensors):
    """Writes a standard GGUF file compatible with llama.cpp."""
    print(f"🔄 Translating {len(tensors)} tensors to GGUF Format...")
    with open(path, 'wb') as f:
        # 1. Header
        f.write(struct.pack('<I', GGUF_MAGIC))
        f.write(struct.pack('<I', GGUF_VERSION))
        f.write(struct.pack('<Q', len(tensors))) # n_tensors
        f.write(struct.pack('<Q', 0))            # n_kv (metadata)

        # 2. Tensor Info
        offset = 0
        for t in tensors:
            # Name (string)
            name_bytes = t['name'].encode('utf-8')
            f.write(struct.pack('<Q', len(name_bytes)))
            f.write(name_bytes)

            # Dimensions
            f.write(struct.pack('<I', len(t['shape'])))
            for dim in t['shape']:
                f.write(struct.pack('<Q', dim))

            # Type
            f.write(struct.pack('<I', t['type']))

            # Offset (relative to the start of the tensor data block)
            f.write(struct.pack('<Q', offset))

            # Calculate size for next offset
            if t['type'] == GGML_TYPE_F32:
                offset += len(t['data']) * 4
            elif t['type'] == GGML_TYPE_Q8_0:
                offset += len(t['data']) * 1

        # 3. Alignment Padding (GGUF requires 32-byte alignment for tensor data)
        current_offset = f.tell()
        padding = (32 - (current_offset % 32)) % 32
        f.write(b'\x00' * padding)

        # 4. Tensor Data
        for t in tensors:
            f.write(t['data'].tobytes())

    print(f"✅ GGUF Export Complete: {path}")
    print(f"   📦 Tensors: {len(tensors)}")
    print(f"   🤖 Compatible with: llama.cpp, Ollama, LM Studio")

if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == 'export-gguf':
        in_path = sys.argv[2]
        out_path = sys.argv[3]
        if not os.path.exists(in_path):
            print(f"❌ Input file {in_path} not found.")
            sys.exit(1)
        tensors = read_lancius_tensors(in_path)
        write_gguf(out_path, tensors)
    else:
        print("Lancius Ecosystem Bridge (GGUF Exporter)")
        print("Usage: python3 lancius_bridge.py export-gguf <input.lancius> <output.gguf>")
