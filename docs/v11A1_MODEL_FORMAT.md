# Lancius v11A1 Model Format Decision

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

## Compatibility policy for v11A1

Binary model compatibility is **not guaranteed** during v11A1.

Models written by v11A1 before Task 6b should be treated as development artifacts.

## Task 6b status

C model saving now uses v2 by default.

C model loading supports:

- v2 preferred
- v1 legacy fallback

The Python ONNX converter still writes v1 for now. The C loader accepts v1, so existing parity workflows remain functional.
