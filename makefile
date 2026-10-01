CC = gcc
CFLAGS = -Wall -Wextra -g -Werror -O3 -mavx2 -mfma -fopenmp -std=c11 -I./include -fPIC -MMD -MP
LDFLAGS = -lm
SRCS = src/core/lancius_arena.c \
       src/core/lancius_serialize.c \
       src/core/lancius_serialize_v2.c \
       src/core/lancius_stable_api.c \
       src/core/lancius_error.c \
       src/ir/lancius_ir.c \
       src/runtime/lancius_scheduler.c \
       src/runtime/lancius_memory_planner.c \
       src/math/lancius_autodiff.c \
       src/math/lancius_kernels.c \
       src/runtime/lancius_threadpool.c \
       src/runtime/lancius_transformer.c \
       src/runtime/lancius_vision_ops.c \
       src/compiler/lancius_bytecode.c \
       src/compiler/lancius_optimizer.c \
       src/core/lancius_checked.c \
src/core/lancius_validate.c \
src/compiler/lancius_quantize.c \
src/train/lancius_train.c \
src/runtime/lancius_sandbox.c

OBJS = $(SRCS:.c=.o)
-include $(OBJS:.o=.d)
all: liblancius.a lancius audit_internals stress_test test_torture train_mnist train_cifar10 fuzz_lancius test_path_bg run_edge test_grad_check audit_ffi audit_memory_pool test_diamond_memory soak_fuzz parity_runner run_trained_batch audit_threadpool_parity audit_nan_injection audit_flash_attention audit_modern_llm audit_known_answer audit_regression_13c audit_transformer_known_answer audit_fp32_path audit_fault_injection audit_despot_probe train_verifier_head distill_prm800k audit_train_lib audit_sandbox train_micromodel eval_verifier
lancius: examples/lancius_cli.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS) -fopenmp -lpthread
liblancius.a: $(OBJS)
	ar rcs $@ $(OBJS)
train_mnist: examples/train_mnist.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS) -fopenmp -lpthread
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@
clean:
	rm -f $(OBJS) liblancius.a
	rm -f $(OBJS:.o=.d) src/*/*.d *.d
	rm -f src/runtime/lancius_memory_planner.o src/core/lancius_stable_api.o
	rm -f train_mnist train_cifar10 fuzz_lancius test_path_bg run_edge test_grad_check
	rm -f test_torture stress_test audit_internals
	rm -f audit_memory_pool audit_flash_attention audit_modern_llm audit_ffi
	rm -f audit_threadpool_parity audit_nan_injection test_diamond_memory soak_fuzz
	rm -f parity_runner run_trained_batch
	rm -f audit_regression_13c regression_roundtrip.lancius regression_bad_*.lancius regression_trunc_*.lancius regression_huge_*.lancius
	rm -f audit_known_answer audit_transformer_known_answer audit_fp32_path audit_fault_injection audit_flash_attention
	rm -f audit_despot_probe train_verifier_head distill_prm800k lancius
	rm -f audit_train_lib audit_sandbox train_micromodel eval_verifier
.PHONY: all clean check check-long check-sanitizers
train_cifar10: examples/train_cifar10.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS) -fopenmp -lpthread
fuzz_lancius: examples/fuzz_lancius.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS) -fopenmp
test_path_bg: examples/test_path_bg.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS)
run_edge: examples/run_edge.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS)
test_grad_check: examples/test_grad_check.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS)
test_torture: examples/test_torture.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS)

stress_test: examples/stress_test.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS)

audit_internals: examples/audit_internals.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS)

# --- v11A1: memory planner is now part of liblancius.a ---

audit_memory_pool: examples/audit_memory_pool.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS)

audit_flash_attention: examples/audit_flash_attention.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS) -fopenmp -lpthread

audit_modern_llm: examples/audit_modern_llm.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS) -fopenmp -lpthread

# --- v11A1: stable API is now part of liblancius.a ---

audit_ffi: examples/audit_ffi.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS) -fopenmp -lpthread

# --- P1 RED TEAM: Sanitizer Gauntlet ---
test_asan: CFLAGS := -Wall -Wextra -g -O0 -fsanitize=address -fno-omit-frame-pointer -fopenmp -std=c11 -I./include -fPIC
test_asan: LDFLAGS := -fsanitize=address -fopenmp -lm -lpthread
test_asan: stress_test test_torture fuzz_lancius
	@echo "✅ ASan Build Complete. Run ./stress_test && ./test_torture"

test_ubsan: CFLAGS := -Wall -Wextra -g -O0 -fsanitize=undefined -fopenmp -std=c11 -I./include -fPIC
test_ubsan: LDFLAGS := -fsanitize=undefined -fopenmp -lm -lpthread
test_ubsan: stress_test test_torture fuzz_lancius
	@echo "✅ UBSan Build Complete. Run ./stress_test && ./test_torture"

# --- V1.0 Red Team: Threadpool & NaN Audits ---
audit_threadpool_parity: examples/audit_threadpool_parity.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS) -fopenmp -lpthread

audit_nan_injection: examples/audit_nan_injection.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS) -fopenmp -lpthread

# --- v10S Adversarial Alpha: Diamond Graph Test ---
test_diamond_memory: examples/test_diamond_memory.c liblancius.a
	$(CC) $(CFLAGS) -fsanitize=address -fno-omit-frame-pointer -g -o $@ $< liblancius.a $(LDFLAGS) -fsanitize=address

# --- v10S Ecosystem Mandate: System Installation ---
PREFIX ?= /usr/local

install: liblancius.a
	@echo "📦 Installing Lancius headers to $(PREFIX)/include/lancius..."
	@mkdir -p $(PREFIX)/include/lancius
	@cp -r include/lancius/*.h $(PREFIX)/include/lancius/
	@cp include/lancius.h $(PREFIX)/include/ 2>/dev/null || true
	@echo "📦 Installing Lancius static library to $(PREFIX)/lib..."
	@mkdir -p $(PREFIX)/lib
	@cp liblancius.a $(PREFIX)/lib/
	@echo "✅ Lancius v12R1 installed successfully."

uninstall:
	@echo "🗑️  Removing Lancius from $(PREFIX)..."
	@rm -rf $(PREFIX)/include/lancius
	@rm -f $(PREFIX)/include/lancius.h
	@rm -f $(PREFIX)/lib/liblancius.a
	@echo "✅ Lancius uninstalled."

# --- v10S Adversarial Soak Gauntlet ---
soak_fuzz: examples/soak_fuzz.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS) -fopenmp

# --- v11A1 repair: missing parity/trained-batch targets ---
parity_runner: examples/parity_runner.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS) -fopenmp -lpthread

run_trained_batch: examples/run_trained_batch.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS) -fopenmp -lpthread

# --- v11A1 Task 13a: validation gates ---
check: all
	./stress_test
	./test_torture
	./test_path_bg
	./test_grad_check
	./audit_internals
	./audit_ffi
	./audit_threadpool_parity
	./audit_nan_injection
	./audit_memory_pool
	./test_diamond_memory
	./audit_flash_attention
	./audit_modern_llm
	./audit_known_answer
	./audit_regression_13c
	./audit_transformer_known_answer
	./audit_fp32_path
	./audit_fault_injection
	./audit_despot_probe
	./train_verifier_head
	./distill_prm800k --selftest
	./audit_train_lib
	./audit_sandbox
	./train_micromodel
	./eval_verifier
	python3 audit_text_pipeline.py
	python3 audit_abi.py
	python3 audit_binding_smoke.py
	./lancius info test_model.lancius
	./lancius run test_model.lancius --mode static --fill zero
	./lancius eval test_model.lancius --mode static
	@echo "v12R2 check complete."

check-long: check
	./soak_fuzz
	./fuzz_lancius 12345
	@echo "v12R2 long check complete."

# --- v11A1 Task 13b: known-answer audit ---
audit_known_answer: examples/audit_known_answer.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS) -fopenmp -lpthread

# --- v11A1 Task 13c: regression audit ---
audit_regression_13c: examples/audit_regression_13c.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS) -fopenmp -lpthread

# --- v11A1 Task 13c: sanitizer gate ---
check-sanitizers:
	$(MAKE) -B test_asan
	./stress_test
	./test_torture
	./fuzz_lancius 12345
	$(MAKE) -B test_ubsan
	./stress_test
	./test_torture
	./fuzz_lancius 12345
	@echo "v12R2 sanitizer gate complete."
	@echo "Restoring normal build (removing sanitizer instrumentation)..."
	$(MAKE) -B all
	@echo "Normal build restored. Safe to run 'make check' now."

audit_transformer_known_answer: examples/audit_transformer_known_answer.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS) -fopenmp -lpthread

audit_fp32_path: examples/audit_fp32_path.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS) -fopenmp -lpthread

# --- v12R1: despot probe pins the P0 fixes (broadcast, planner, CE truth) ---
audit_despot_probe: examples/audit_despot_probe.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS) -fopenmp -lpthread

# --- v12R2: verifier-head training example (models-side, tanh + MSE) ---
train_verifier_head: examples/train_verifier_head.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS) -fopenmp -lpthread

# --- v12R2: PRM800k step distiller in C (retires distill_prm800k.py) ---
distill_prm800k: examples/distill_prm800k.c
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# --- v12R2 training + sandbox + micromodel gates ---
audit_train_lib: examples/audit_train_lib.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS) -fopenmp -lpthread

audit_sandbox: examples/audit_sandbox.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS) -fopenmp -lpthread

train_micromodel: examples/train_micromodel.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS) -fopenmp -lpthread

eval_verifier: examples/eval_verifier.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS) -fopenmp -lpthread

audit_fault_injection: examples/audit_fault_injection.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS) -fopenmp -lpthread
