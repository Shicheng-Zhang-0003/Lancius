CC = gcc
CFLAGS = -Wall -Wextra -g -Werror -O3 -march=native -std=c11 -I./include -fPIC
LDFLAGS = -lm

SRCS = src/core/v8_arena.c \
       src/ir/v8_ir.c \
       src/runtime/v8_scheduler.c \
       src/math/v8_autodiff.c \
       src/runtime/v8_threadpool.c \
       src/compiler/v8_bytecode.c

OBJS = $(SRCS:.c=.o)

all: libbasis8.a train_mnist_v8

libbasis8.a: $(OBJS)
	ar rcs $@ $(OBJS)

train_mnist_v8: examples/train_mnist_v8.c libbasis8.a
	$(CC) $(CFLAGS) -o $@ $< libbasis8.a $(LDFLAGS) -lpthread

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm -f $(OBJS) libbasis8.a train_mnist_v8

.PHONY: all clean
