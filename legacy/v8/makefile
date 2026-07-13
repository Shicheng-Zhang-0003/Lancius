CC = gcc
CFLAGS = -Wall -Wextra -g -Werror -O3 -march=native -fopenmp -std=c11 -I./include -fPIC
LDFLAGS = -lm

SRCS = src/core/v8_arena.c \
       src/core/v8_serialize.c \
       src/ir/v8_ir.c \
       src/runtime/v8_scheduler.c \
       src/math/v8_autodiff.c \
       src/math/v8_kernels.c \
       src/runtime/v8_threadpool.c \
       src/runtime/v8_vision_ops.c \
       src/compiler/v8_bytecode.c \
       src/compiler/v8_optimizer.c

OBJS = $(SRCS:.c=.o)

all: libbasis8.a train_mnist_v8 train_cifar10_v8 fuzz_v8 test_path_bg run_edge test_grad_check

libbasis8.a: $(OBJS)
	ar rcs $@ $(OBJS)

train_mnist_v8: examples/train_mnist_v8.c libbasis8.a
	$(CC) $(CFLAGS) -o $@ $< libbasis8.a $(LDFLAGS) -fopenmp -lpthread

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) libbasis8.a train_mnist_v8

.PHONY: all clean

train_cifar10_v8: examples/train_cifar10_v8.c libbasis8.a
	$(CC) $(CFLAGS) -o $@ $< libbasis8.a $(LDFLAGS) -fopenmp

fuzz_v8: examples/fuzz_v8.c libbasis8.a
	$(CC) $(CFLAGS) -o $@ $< libbasis8.a $(LDFLAGS) -fopenmp

test_path_bg: examples/test_path_bg.c libbasis8.a
	$(CC) $(CFLAGS) -o $@ $< libbasis8.a $(LDFLAGS)

run_edge: examples/run_edge.c libbasis8.a
	$(CC) $(CFLAGS) -o $@ $< libbasis8.a $(LDFLAGS)

test_grad_check: examples/test_grad_check.c libbasis8.a
	$(CC) $(CFLAGS) -o $@ $< libbasis8.a $(LDFLAGS)
