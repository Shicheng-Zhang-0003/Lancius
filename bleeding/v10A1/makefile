CC = gcc
CFLAGS = -Wall -Wextra -g -Werror -O3 -march=native -mavx2 -mfma -fopenmp -std=c11 -I./include -fPIC
LDFLAGS = -lm

SRCS = src/core/lancius_arena.c \
       src/core/lancius_serialize.c \
       src/ir/lancius_ir.c \
       src/runtime/lancius_scheduler.c \
       src/math/lancius_autodiff.c \
       src/math/lancius_kernels.c \
       src/runtime/lancius_threadpool.c \
       src/runtime/lancius_vision_ops.c \
       src/compiler/lancius_bytecode.c \
       src/compiler/lancius_optimizer.c \
       src/compiler/lancius_quantize.c

OBJS = $(SRCS:.c=.o)

all: liblancius.a audit_internals stress_test test_torture generate_text run_llm train_mnist train_cifar10 fuzz_lancius test_path_bg run_edge test_grad_check

liblancius.a: $(OBJS)
	ar rcs $@ $(OBJS)

train_mnist: examples/train_mnist.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS) -fopenmp -lpthread

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) liblancius.a train_mnist

.PHONY: all clean

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

# --- v9A2 Memory Determinism Mandate ---
src/runtime/lancius_memory_planner.o: src/runtime/lancius_memory_planner.c include/lancius/lancius_memory_planner.h
	$(CC) $(CFLAGS) -c $< -o $@

audit_memory_pool: examples/audit_memory_pool.c src/runtime/lancius_memory_planner.o liblancius.a
	$(CC) $(CFLAGS) -o $@ $< src/runtime/lancius_memory_planner.o liblancius.a $(LDFLAGS)

audit_flash_attention: examples/audit_flash_attention.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS) -fopenmp -lpthread

audit_modern_llm: examples/audit_modern_llm.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS) -fopenmp -lpthread

src/core/lancius_stable_api.o: src/core/lancius_stable_api.c include/lancius/lancius_stable_api.h
	$(CC) $(CFLAGS) -c $< -o $@

audit_ffi: examples/audit_ffi.c src/core/lancius_stable_api.o liblancius.a
	$(CC) $(CFLAGS) -o $@ $< src/core/lancius_stable_api.o liblancius.a $(LDFLAGS) -fopenmp -lpthread

# --- P1 RED TEAM: Sanitizer Gauntlet ---
test_asan: CFLAGS += -fsanitize=address -fno-omit-frame-pointer -g
test_asan: LDFLAGS += -fsanitize=address
test_asan: stress_test test_torture fuzz_lancius
	@echo "✅ ASan Build Complete. Run ./stress_test && ./test_torture"

test_ubsan: CFLAGS += -fsanitize=undefined -g
test_ubsan: LDFLAGS += -fsanitize=undefined
test_ubsan: stress_test test_torture fuzz_lancius
	@echo "✅ UBSan Build Complete. Run ./stress_test && ./test_torture"

# --- V1.0 Red Team: Threadpool & NaN Audits ---
audit_threadpool_parity: examples/audit_threadpool_parity.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS) -fopenmp -lpthread

audit_nan_injection: examples/audit_nan_injection.c liblancius.a
	$(CC) $(CFLAGS) -o $@ $< liblancius.a $(LDFLAGS) -fopenmp -lpthread
