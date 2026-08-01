CC = gcc
CFLAGS = -Wall -Wextra -g -Werror -O3 -mavx2 -mfma -fopenmp -std=c11 -I./include -fPIC
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
       src/compiler/lancius_quantize.c

OBJS = $(SRCS:.c=.o)
all: liblancius.a audit_internals stress_test test_torture generate_text run_llm train_mnist train_cifar10 fuzz_lancius test_path_bg run_edge test_grad_check audit_ffi audit_memory_pool test_diamond_memory soak_fuzz parity_runner run_trained_batch audit_threadpool_parity audit_nan_injection audit_flash_attention audit_modern_llm audit_known_answer audit_regression_13c audit_transformer_known_answer audit_fp32_path
liblancius.a: $(OBJS)
	ar rcs $@ $(OBJS)
train_mnist: examples/train_mnist.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS) -fopenmp -lpthread
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@
clean:
	rm -f $(OBJS) liblancius.a
	rm -f src/runtime/lancius_memory_planner.o src/core/lancius_stable_api.o
	rm -f train_mnist train_cifar10 fuzz_lancius test_path_bg run_edge test_grad_check
	rm -f run_llm generate_text test_torture stress_test audit_internals
	rm -f audit_memory_pool audit_flash_attention audit_modern_llm audit_ffi
	rm -f audit_threadpool_parity audit_nan_injection test_diamond_memory soak_fuzz
	rm -f parity_runner run_trained_batch
	rm -f audit_regression_13c regression_roundtrip.lancius regression_bad_*.lancius regression_trunc_*.lancius regression_huge_*.lancius
.PHONY: all clean check check-long check-sanitizers
train_cifar10: examples/train_cifar10.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS) -fopenmp
fuzz_lancius: examples/fuzz_lancius.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS) -fopenmp
test_path_bg: examples/test_path_bg.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS)
run_edge: examples/run_edge.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS)
test_grad_check: examples/test_grad_check.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS)
run_llm: examples/run_llm.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS)
generate_text: examples/generate_text.c liblancius.a
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
	@echo "✅ Lancius v11A3 installed successfully."

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
	@echo "v11A3 check complete."

check-long: check
	./soak_fuzz
	./fuzz_lancius 12345
	@echo "v11A3 long check complete."

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
	@echo "v11A3 sanitizer gate complete."

audit_transformer_known_answer: examples/audit_transformer_known_answer.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS) -fopenmp -lpthread

audit_fp32_path: examples/audit_fp32_path.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS) -fopenmp -lpthread
