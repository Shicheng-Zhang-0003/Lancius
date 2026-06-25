
CC = gcc
CFLAGS = -Wall -Wextra -O3 -march=native -std=c11 -I./include
# Add -DBASIS_USE_CBLAS and -lopenblas to CFLAGS/LDFLAGS to enable hardware-accelerated SIMD MatMul
LDFLAGS = -lm -ldl

SRCS = src/basis/core/error.c \
       src/basis/stage1_atomic/scalar.c \
       src/basis/stage2_linear/tensor.c \
       src/basis/stage3_sequence/sequence.c \
       src/basis/stage4_learning/learning.c \
       src/basis/stage4_learning/optim.c \
       src/basis/stage5_unified/symbolic.c \
       src/basis/stage5_unified/compiler.c \
       src/basis/stage5_unified/geometry.c \
       src/basis/stage5_unified/jit.c

OBJS = $(SRCS:.c=.o)

all: test_v5_suite

test_v5_suite: tests/test_v5_suite.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS) $(LDFLAGS)

clean:
	rm -f $(OBJS) test_v5_suite

.PHONY: all clean
