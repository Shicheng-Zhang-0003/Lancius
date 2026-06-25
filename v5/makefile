CC = gcc
CFLAGS = -Wall -Wextra -O3 -march=native -std=c11 -I./include -fPIC
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

all: test_v5_suite libbasis.so libbasis.a

# Shared Library (Dynamic Linking)
libbasis.so: $(OBJS)
	$(CC) -shared -o $@ $(OBJS) $(LDFLAGS)

# Static Library (Static Linking)
libbasis.a: $(OBJS)
	ar rcs $@ $(OBJS)

# Test Suite
test_v5_suite: tests/test_v5_suite.c $(OBJS)
	$(CC) $(CFLAGS) -o $@ $< $(OBJS) $(LDFLAGS)

# System-wide Installation (Standard Linux FHS)
PREFIX ?= /usr/local
install: all
	install -d $(PREFIX)/lib
	install -d $(PREFIX)/include/basis
	install -m 755 libbasis.so $(PREFIX)/lib/
	install -m 644 libbasis.a $(PREFIX)/lib/
	cp -r include/basis/* $(PREFIX)/include/basis/
	install -m 644 include/basis.h $(PREFIX)/include/
	@echo "✅ BASIS installed to $(PREFIX)"
	@echo "   You can now compile with: gcc main.c -lbasis -lm -ldl"

uninstall:
	rm -f $(PREFIX)/lib/libbasis.so $(PREFIX)/lib/libbasis.a
	rm -rf $(PREFIX)/include/basis
	rm -f $(PREFIX)/include/basis.h
	@echo "❌ BASIS uninstalled"

clean:
	rm -f $(OBJS) test_v5_suite libbasis.so libbasis.a

.PHONY: all clean install uninstall
