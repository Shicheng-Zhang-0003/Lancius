CC = gcc
CFLAGS = -Wall -Wextra -O3 -march=native -std=c11 -I./include -fPIC
LDFLAGS = -lm -ldl

SRCS = src/basis/core/error.c \
       src/basis/core/memory.c \
       src/basis/core/ir.c \
       src/basis/stage1_atomic/scalar.c \
       src/basis/stage2_linear/tensor.c \
       src/basis/stage3_sequence/sequence.c \
       src/basis/stage4_learning/learning.c \
       src/basis/stage4_learning/optim.c \
       src/basis/stage5_unified/symbolic.c \
       src/basis/stage5_unified/compiler.c \
       src/basis/stage5_unified/geometry.c \
       src/basis/stage5_unified/jit.c \
       src/basis/stage6_ir/graph.c \
       src/basis/stage6_ir/lowering.c \
       src/basis/l1_math/tensor_v7.c \
       src/basis/l4_compiler/pass_manager.c \
       src/basis/l4_compiler/lowering.c \
       src/basis/l5_runtime/scheduler.c \
       src/basis/l6_learning/autodiff.c \
       src/basis/l7_intelligence/swarm.c

OBJS = $(SRCS:.c=.o)

all: test_v5_suite libbasis.so libbasis.a

libbasis.so: $(OBJS)
	$(CC) -shared -o $@ $(OBJS) $(LDFLAGS)

libbasis.a: $(OBJS)
	ar rcs $@ $(OBJS)

test_v5_suite: tests/test_v5_suite.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS) $(LDFLAGS)

test_v7_memory: tests/test_v7_memory.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS) $(LDFLAGS)

test_v7_ir: tests/test_v7_ir.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS) $(LDFLAGS)

test_v7_tensor: tests/test_v7_tensor.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS) $(LDFLAGS)

test_v7_passes: tests/test_v7_passes.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS) $(LDFLAGS)

test_v7_lowering: tests/test_v7_lowering.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS) $(LDFLAGS)

test_v7_scheduler: tests/test_v7_scheduler.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS) $(LDFLAGS)


test_v7_transformer_primitives: tests/test_v7_transformer_primitives.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS) $(LDFLAGS)


test_v7_transformer: tests/test_v7_transformer.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS) $(LDFLAGS)


test_v7_autodiff: tests/test_v7_autodiff.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS) $(LDFLAGS)


test_v7_swarm: tests/test_v7_swarm.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS) $(LDFLAGS)

PREFIX ?= /usr/local
install: all
	install -d $(PREFIX)/lib
	install -d $(PREFIX)/include/basis
	install -m 755 libbasis.so $(PREFIX)/lib/
	install -m 644 libbasis.a $(PREFIX)/lib/
	cp -r include/basis/* $(PREFIX)/include/basis/
	install -m 644 include/basis.h $(PREFIX)/include/

uninstall:
	rm -f $(PREFIX)/lib/libbasis.so $(PREFIX)/lib/libbasis.a
	rm -rf $(PREFIX)/include/basis
	rm -f $(PREFIX)/include/basis.h

clean:
	rm -f $(OBJS) test_v5_suite test_v7_* libbasis.so libbasis.a

.PHONY: all clean install uninstall
