# BASIS V8: The Production Distillation

V7 was the crucible. It proved the O(1) Arena, the Immutable SSA IR, and the VJP Autodiff.
V8 is the product.

## The V8 Mandate
1. **No Dual APIs:** The V5 scalar tape is dead. The V7 IR is the single source of truth.
2. **No String Hacks:** AOT compilation via `system("gcc")` is banned. V8 uses binary serialization and direct memory mapping.
3. **Notepad First:** Infrastructure exists only to serve the model. V8 ships with an MNIST trainer.
4. **Hardware Awareness:** The Wave Scheduler will map to pthreads and respect cache lines.

## Building
```bash
make clean && make