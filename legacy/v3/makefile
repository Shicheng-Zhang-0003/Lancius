CC = gcc
CFLAGS = -Wall -Wextra -O3 -march=native -std=c11 -I./include
LDFLAGS = -lm

SRCS = src/basis/stage1_atomic/scalar.c \
       src/basis/stage2_linear/tensor.c \
       src/basis/stage3_sequence/sequence.c \
       src/basis/stage4_learning/learning.c \
       src/basis/stage4_learning/optim.c \
       src/basis/stage5_unified/symbolic.c \
       src/basis/stage5_unified/compiler.c \
       src/basis/stage5_unified/geometry.c

OBJS = $(SRCS:.c=.o)

TESTS = test_scalar test_tensor test_attention test_symbolic test_compiler test_frontier test_grand_unified

all: $(TESTS)

test_scalar: tests/test_scalar.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS) $(LDFLAGS)

test_tensor: tests/test_tensor.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS) $(LDFLAGS)

test_attention: tests/test_attention.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS) $(LDFLAGS)

test_symbolic: tests/test_symbolic.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS) $(LDFLAGS)

test_compiler: tests/test_compiler.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS) $(LDFLAGS)

test_frontier: tests/test_frontier.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS) $(LDFLAGS)

test_grand_unified: tests/test_grand_unified.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS) $(LDFLAGS)

clean:
	rm -f $(OBJS) $(TESTS)

.PHONY: all clean
