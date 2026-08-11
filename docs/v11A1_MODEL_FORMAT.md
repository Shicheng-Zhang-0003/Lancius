# Lancius Model Format (v11A3 Freeze)

## Active format before Task 6b

Until Task 6b is applied, the active model format is **v1**.

Active components:

- C writer: `lancius_graph_save()`
- C loader: `lancius_graph_load()`
- Python converter: `onnx_to_lancius.py`
- Magic: `0x21434E41`

## Known v1 limitations

The v1 format is usable but not stable:

- uses native `size_t` fields
- depends on native enum size
- has no explicit version header
- has no checksum
- is not portable across architectures
- is not guaranteed to remain loadable in future versions

## Task 6b goal

Task 6b implemented the C v2 format:

- explicit magic: `0x32434E41`
- explicit version: `2`
- fixed-width little-endian fields
- no native `size_t`
- no native enum-size dependence
- explicit header flags
- checksum field
- stronger loader validation

## Compatibility policy (v11A3 freeze)

The v2 model format is **frozen** as of v11A3.
Binary compatibility is guaranteed for v2 models written by v11A3 and later.

### CRC32 integrity

The `checksum_crc32` header field covers the model body (bytes 48..EOF).
- `checksum_crc32 == 0`: legacy model (pre-CRC), accepted without verification.
- `checksum_crc32 != 0`: verified on load; mismatch rejects the model.

### Reserved flags

`LANCIUS_MODEL_FLAG_EXTERNAL_WEIGHTS` (bit 2) is reserved and must not be set.
The loader rejects models with this flag.

### v1 fallback policy

v1 loading is **deprecated** legacy best-effort. It may be removed in v12.
v2 is the only supported save format.

Models written by v11A1 before Task 6b should be treated as development artifacts.

## Task 6b status

C model saving now uses v2 by default.

C model loading supports:

- v2 preferred
- v1 legacy fallback

The Python ONNX converter writes v2. The C loader accepts both v2 (preferred) and v1 (legacy fallback).
