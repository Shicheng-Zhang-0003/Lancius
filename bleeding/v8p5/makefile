CC = gcc
CFLAGS = -Wall -Wextra -g -Werror -O3 -march=native -fopenmp -std=c11 -I./include -fPIC
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

all: liblancius.a run_llm train_mnist train_cifar10 fuzz_lancius test_path_bg run_edge test_grad_check

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
