# Lancius build system.
#
# ONE build tree: every compiled thing lands under build/, mirrored by what
# it is. The repository root is for sources, corpora and docs only — a clean
# `ls` at the root shows what the project IS, not what you last compiled.
#
#   build/lib/          liblancius.a              (the library; consumers link this path)
#   build/bin/          lancius CLI + every example, trainer and audit binary
#   build/obj/          object files + .d depfiles, mirroring src/
#   build/san/          check-sanitizers workspace (instrumented lib + audits)
#   temp/               gate scratch + probe outputs (transient)
#
# `make help` prints the map. `make env` prints an eval-able PATH addition.
# CMake builds land in build/ as well, so both build systems agree on one
# output convention.

CC = gcc
CFLAGS = -Wall -Wextra -g -Werror -O3 -mavx2 -mfma -fopenmp -std=c11 -I./include -fPIC
LDFLAGS = -lm

BUILD  = build
BIN    = $(BUILD)/bin
LIBD   = $(BUILD)/lib
OBJD   = $(BUILD)/obj
LIB    = $(LIBD)/liblancius.a

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

OBJS = $(SRCS:%.c=$(OBJD)/%.o)

# ---- link rules, table-driven so 40 targets stay one rule each ----------
# $(1) = binary name, $(2) = its single .c source, $(3) = extra libs.
# LINK          : bare                    (none needed)
# LINK_PTHREAD  : + -fopenmp -lpthread    (normal OpenMP programs)
# LINK_OMP      : + -fopenmp only         (program opens its own pool)
define LINK
$$(BIN)/$(1): $(2) $(LIB) | $$(BIN)
	$$(CC) $$(CFLAGS) -o $$@ $$< $$(LIB) $$(LDFLAGS) $(3)
endef
$(eval $(call LINK,lancius,examples/lancius_cli.c,-fopenmp -lpthread))
$(eval $(call LINK,train_mnist,examples/train_mnist.c,-fopenmp -lpthread))
$(eval $(call LINK,train_verifier,examples/train_verifier.c,-fopenmp -lpthread))
$(eval $(call LINK,train_cifar10,examples/train_cifar10.c,-fopenmp -lpthread))
$(eval $(call LINK,train_micromodel,examples/train_micromodel.c,-fopenmp -lpthread))
$(eval $(call LINK,train_i3,examples/train_i3.c,-fopenmp -lpthread))
$(eval $(call LINK,eval_verifier,examples/eval_verifier.c,-fopenmp -lpthread))
$(eval $(call LINK,train_verifier_head,examples/train_verifier_head.c,-fopenmp -lpthread))
$(eval $(call LINK,fuzz_lancius,examples/fuzz_lancius.c,-fopenmp))
$(eval $(call LINK,soak_fuzz,examples/soak_fuzz.c,-fopenmp))
$(eval $(call LINK,test_path_bg,examples/test_path_bg.c,))
$(eval $(call LINK,run_edge,examples/run_edge.c,))
$(eval $(call LINK,test_grad_check,examples/test_grad_check.c,))
$(eval $(call LINK,test_torture,examples/test_torture.c,))
$(eval $(call LINK,stress_test,examples/stress_test.c,))
$(eval $(call LINK,audit_internals,examples/audit_internals.c,))
$(eval $(call LINK,audit_memory_pool,examples/audit_memory_pool.c,))
$(eval $(call LINK,audit_ffi,examples/audit_ffi.c,-fopenmp -lpthread))
$(eval $(call LINK,audit_flash_attention,examples/audit_flash_attention.c,-fopenmp -lpthread))
$(eval $(call LINK,audit_modern_llm,examples/audit_modern_llm.c,-fopenmp -lpthread))
$(eval $(call LINK,audit_threadpool_parity,examples/audit_threadpool_parity.c,-fopenmp -lpthread))
$(eval $(call LINK,audit_nan_injection,examples/audit_nan_injection.c,-fopenmp -lpthread))
$(eval $(call LINK,parity_runner,examples/parity_runner.c,-fopenmp -lpthread))
$(eval $(call LINK,run_trained_batch,examples/run_trained_batch.c,-fopenmp -lpthread))
$(eval $(call LINK,audit_known_answer,examples/audit_known_answer.c,-fopenmp -lpthread))
$(eval $(call LINK,audit_regression_13c,examples/audit_regression_13c.c,-fopenmp -lpthread))
$(eval $(call LINK,audit_transformer_known_answer,examples/audit_transformer_known_answer.c,-fopenmp -lpthread))
$(eval $(call LINK,audit_fp32_path,examples/audit_fp32_path.c,-fopenmp -lpthread))
$(eval $(call LINK,audit_fault_injection,examples/audit_fault_injection.c,-fopenmp -lpthread))
$(eval $(call LINK,audit_despot_probe,examples/audit_despot_probe.c,-fopenmp -lpthread))
$(eval $(call LINK,audit_train_lib,examples/audit_train_lib.c,-fopenmp -lpthread))
$(eval $(call LINK,trainlib_dump,tools/audit/trainlib_dump.c,-fopenmp -lpthread))
$(eval $(call LINK,audit_sandbox,examples/audit_sandbox.c,-fopenmp -lpthread))
$(eval $(call LINK,audit_sum_axis_nd,examples/audit_sum_axis_nd.c,-fopenmp -lpthread))
$(eval $(call LINK,audit_v7_hardening,examples/audit_v7_hardening.c,-fopenmp -lpthread))
$(eval $(call LINK,audit_train_bwd,examples/audit_train_bwd.c,-fopenmp -lpthread))
$(eval $(call LINK,audit_train_converge,examples/audit_train_converge.c,-fopenmp -lpthread))
$(eval $(call LINK,bench_kernels,examples/bench_kernels.c,-fopenmp -lpthread))

# distill_prm800k is the one standalone gate binary: no liblancius.a (whole
# point of V9 gate truth: it must not link so it cannot inherit a defect).
$(BIN)/distill_prm800k: examples/distill_prm800k.c | $(BIN)
	$(CC) $(CFLAGS) -o $@ $< $(LDFLAGS)

# test_diamond_memory is its own sanitizer build: ASan by construction, not
# reached through the combined gate flags.
$(BIN)/test_diamond_memory: examples/test_diamond_memory.c $(LIB) | $(BIN)
	$(CC) -Wall -Wextra -fsanitize=address -fno-omit-frame-pointer -g -I./include -fopenmp -o $@ $< $(LIB) $(LDFLAGS) -fsanitize=address

# ldfd_stream_train (experimental, v12R3): the LDFD -> Lancius stream plug.
# It compiles every LDFD source directly into the binary, so one executable
# owns BOTH the Lancius trainer and the live data path — no libsnapshot.so,
# no shared state between the two frameworks. The curl_multi loop in fetch.c
# reaches snap_pipeline_rearm (context.c) and pipeline construction reaches
# the JSON parser vtables, so the FULL LDFD source set links here — same
# dependency shape as the module itself (libcurl + cJSON + zlib).
# Deliberately NOT part of `all` (needs libcurl dev headers, like every LDFD
# consumer): build it explicitly via `make check-ldfd-stream`. A missing
# dependency turns that build red, never a silent pass.
LDFD_PLUG_SRC = 3463-LDFD/src/fetch.c 3463-LDFD/src/buffer.c \
                3463-LDFD/src/parser_csv.c 3463-LDFD/src/parser_json.c \
                3463-LDFD/src/context.c 3463-LDFD/src/config.c \
                3463-LDFD/src/assoc.c 3463-LDFD/src/decompress.c \
                3463-LDFD/src/output.c
$(BIN)/ldfd_stream_train: examples/ldfd_stream_train.c $(LDFD_PLUG_SRC) $(LIB) | $(BIN)
	$(CC) $(CFLAGS) -D_POSIX_C_SOURCE=200809L -I3463-LDFD/include -o $@ $< \
	    $(LDFD_PLUG_SRC) $(LIB) $(LDFLAGS) -lcurl -lcjson -lz -fopenmp -lpthread

$(LIB): $(OBJS) | $(LIBD)
	ar rcs $@ $(OBJS)

$(BIN) $(LIBD):
	@mkdir -p $@

$(OBJD)/%.o: %.c
	@mkdir -p $(@D)
	$(CC) $(CFLAGS) -MMD -MP -c $< -o $@

# Default goal must precede any -include: depfiles define %-targets that
# would otherwise hijack the default goal.
.DEFAULT_GOAL := all

# ---- the old root-level target names stay typed the same way; their
# products simply moved into build/bin. Scripts and docs that reference the
# products point at build/ now. ------------------------------------------
BINS = $(addprefix $(BIN)/,lancius train_mnist train_cifar10 fuzz_lancius \
        test_path_bg run_edge test_grad_check test_torture stress_test \
        audit_internals audit_memory_pool audit_ffi audit_flash_attention \
        audit_modern_llm audit_threadpool_parity audit_nan_injection \
        test_diamond_memory soak_fuzz parity_runner run_trained_batch \
        audit_known_answer audit_regression_13c \
        audit_transformer_known_answer audit_fp32_path \
        audit_fault_injection audit_despot_probe train_verifier_head \
        distill_prm800k audit_train_lib trainlib_dump audit_sandbox \
        train_micromodel eval_verifier audit_sum_axis_nd audit_train_bwd \
        audit_train_converge audit_v7_hardening train_verifier)

all: $(LIB) $(BINS)
	@echo "build ready."
	@echo "  library : $(LIB)"
	@echo "  binaries: $(BIN)/  ($$(ls $(BIN) | wc -l) binaries — 'make help' for the map)"

clean:
	rm -rf $(BUILD)
	# legacy root-level products from before the build/ tree existed; a
	# clean tree should not half-remember them.
	rm -f $(notdir $(BINS))
	rm -f test_model.lancius regression_roundtrip.lancius regression_bad_*.lancius \
	      regression_trunc_*.lancius regression_huge_*.lancius temp/mutation/_t
	rm -f *.o *.d bench_kernels liblancius.a
	rm -rf build-debug stress-logs temp/phys temp/audit/physics
	rm -f temp/audit/physics_probe_out.txt
	@echo "build tree, legacy root binaries and gate scratch removed."

.PHONY: all clean help env check check-long check-sanitizers ldfd-test \
        ldfd-build check-ldfd-stream check-oracle check-mutation \
        check-ubstrict check-physics install uninstall

# ---- terminal friendliness: know where things are without searching -----
help:
	@echo "Lancius v12R2 — outputs all live under build/ ($(BUILD)/bin, $(BUILD)/lib, $(BUILD)/obj)"
	@echo
	@echo "  make               build everything (library + all binaries)"
	@echo '  make env           print eval-able:  eval $$(make env)'
	@echo "  make help          this map"
	@echo
	@echo "run:"
	@echo "  lancius                        the CLI: build/bin/lancius run|info|eval <model>"
	@echo "  train_verifier                 unified verifier trainer (CE, persistent graph)"
	@echo "  train_mnist train_cifar10 train_micromodel train_verifier_head"
	@echo "  eval_verifier                  score a .lancius verifier"
	@echo "  distill_prm800k                standalone PRM distiller (links nothing)"
	@echo
	@echo "language gates:"
	@echo "  make check                     full suite + LDFD bridge + CLI self-test"
	@echo "  make check-long                + soak_fuzz, fuzz_lancius"
	@echo "  make check-sanitizers          instrumented lib + all audits under ASan/UBSan/LSan"
	@echo "  make check-ubstrict            strict UBSan sweep (abort on first finding)"
	@echo "  make check-oracle              external NumPy/torch oracle"
	@echo "  make check-physics             independent IEEE/libm/CRC reference"
	@echo "  make check-mutation            injects defects, demands red"
	@echo
	@echo "measurement (never a gate):"
	@echo "  make bench_kernels             build/bin/bench_kernels"
	@echo
	@echo "data feeding framework (own deps: libcurl, zlib):"
	@echo "  make ldfd-build                build 3463-LDFD"
	@echo "  make ldfd-test                 run its suites"
	@echo "  make check-ldfd-stream         experimental LDFD -> Lancius plug:"
	@echo "                                 train one tiny model from a live LDFD"
	@echo "                                 stream AND from the predownloaded same"
	@echo "                                 corpus, then gate parity + round-trip"
	@echo
	@echo "install:"
	@echo "  make install [PREFIX=/usr/local]   headers + lib + CLI"
	@echo "  make uninstall"
	@echo
	@echo "consumers link: $(LIB)"
	@echo "examples compile: gcc -I./include your.c $(LIB) -fopenmp -lpthread -lm"

env:
	@echo 'export PATH="$(CURDIR)/$(BIN):$${PATH}"'

install: $(LIB) $(BIN)/lancius
	@mkdir -p $(PREFIX)/include/lancius $(PREFIX)/lib $(PREFIX)/bin
	@cp -r include/lancius/*.h $(PREFIX)/include/lancius/
	@cp include/lancius.h $(PREFIX)/include/ 2>/dev/null || true
	@cp $(LIB) $(PREFIX)/lib/
	@cp $(BIN)/lancius $(PREFIX)/bin/
	@echo "installed: $(PREFIX)/include/lancius, $(PREFIX)/lib/liblancius.a, $(PREFIX)/bin/lancius"

uninstall:
	@rm -rf $(PREFIX)/include/lancius
	@rm -f $(PREFIX)/include/lancius.h
	@rm -f $(PREFIX)/lib/liblancius.a
	@rm -f $(PREFIX)/bin/lancius
	@echo "Lancius uninstalled from $(PREFIX)."

# Dependency tracking (must trail all explicit targets so included
# depfiles never become the default goal).
-include $(OBJS:.o=.d)

# ---- v10S Adversarial Alpha: sanitizer gauntlet entry points -------------
test_asan: CFLAGS := -Wall -Wextra -g -O0 -fsanitize=address -fno-omit-frame-pointer -fopenmp -std=c11 -I./include -fPIC
test_asan: LDFLAGS := -fsanitize=address -fopenmp -lm -lpthread
test_asan: stress_test test_torture fuzz_lancius
	@echo "✅ ASan Build Complete — binaries in $(BUILD)/bin/"

test_ubsan: CFLAGS := -Wall -Wextra -g -O0 -fsanitize=undefined -fopenmp -std=c11 -I./include -fPIC
test_ubsan: LDFLAGS := -fsanitize=undefined -fopenmp -lm -lpthread
test_ubsan: stress_test test_torture fuzz_lancius
	@echo "✅ UBSan Build Complete — binaries in $(BUILD)/bin/"

# stress_test/test_torture/fuzz_lancius keep their bare names callable from
# the targets above; they are aliases onto the build/bin products.
stress_test fuzz_lancius test_torture soak_fuzz: %: $(BIN)/%
	@true

# ------------------------------------------------------------------ 3463-LDFD
#
# The Live Data Feeding Framework is a separate component with its own
# makefile and its own dependencies (libcurl, zlib). It is wired in as a
# target rather than folded into `all` so a machine without libcurl headers
# can still build and gate the ML runtime. Everything here degrades to an
# honest SKIP instead of a silent pass.

LDFD_DIR = 3463-LDFD

ldfd-test:
	@echo "LDFD acquisition path: `python3 tools/data/manage_datasets.py status | sed -n 's/.*LDFD (3463) *: //p'`"
	@$(MAKE) -C $(LDFD_DIR) test
	@# exit 77 = skipped (library not built); anything else is a failure
	@python3 tools/audit/audit_ldfd_bridge.py; st=$$?; \
	 if [ $$st -eq 77 ]; then echo "bridge audit skipped (no libsnapshot.so)"; \
	 elif [ $$st -ne 0 ]; then echo "bridge audit FAILED"; exit $$st; fi

ldfd-build:
	$(MAKE) -C $(LDFD_DIR) all

# v12R3 experimental plug: live LDFD stream and a predownloaded corpus both
# train the same tiny model; servers/parsers/trainer in one binary.
check-ldfd-stream: $(BIN)/ldfd_stream_train
	$(BIN)/ldfd_stream_train

check: all
	$(BIN)/stress_test
	$(BIN)/test_torture
	$(BIN)/test_path_bg
	$(BIN)/test_grad_check
	$(BIN)/audit_internals
	$(BIN)/audit_ffi
	$(BIN)/audit_threadpool_parity
	$(BIN)/audit_nan_injection
	$(BIN)/audit_memory_pool
	$(BIN)/test_diamond_memory
	$(BIN)/audit_flash_attention
	$(BIN)/audit_modern_llm
	$(BIN)/audit_known_answer
	$(BIN)/audit_regression_13c
	$(BIN)/audit_transformer_known_answer
	$(BIN)/audit_fp32_path
	$(BIN)/audit_fault_injection
	$(BIN)/audit_despot_probe
	$(BIN)/train_verifier_head
	$(BIN)/distill_prm800k --selftest
	$(BIN)/audit_train_lib
	$(BIN)/audit_sum_axis_nd
	$(BIN)/audit_train_bwd
	$(BIN)/audit_train_converge
	$(BIN)/audit_sandbox
	$(BIN)/audit_v7_hardening
	$(BIN)/train_micromodel
	$(BIN)/eval_verifier
	python3 tools/audit/audit_text_pipeline.py
	python3 tools/audit/audit_abi.py
	python3 tools/audit/audit_binding_smoke.py
	@echo "--- LDFD bridge (3463) ---"
	@$(MAKE) --no-print-directory ldfd-test
	$(BIN)/lancius info test_model.lancius
	$(BIN)/lancius run test_model.lancius --mode static --fill zero
	$(BIN)/lancius eval test_model.lancius --mode static
	@echo "v12R2 check complete."

check-long: check
	$(BIN)/soak_fuzz
	$(BIN)/fuzz_lancius 12345
	@echo "v12R2 long check complete."

# --- v11A1 Task 13b/c (known-answer + regression audits) live in the LINK
# table above; the following is the V7 truth block on sanitizers. ---
SAN_FLAGS = -fsanitize=address,undefined -fno-omit-frame-pointer
SAN_CFLAGS = -Wall -Wextra -g -O1 $(SAN_FLAGS) -fopenmp -std=c11 -I./include -fPIC
SAN_LDFLAGS = $(SAN_FLAGS) -fopenmp -lm -lpthread
SAN_DIR = $(BUILD)/san
SAN_AUDITS = stress_test test_torture fuzz_lancius test_path_bg test_grad_check \
             audit_internals audit_ffi audit_threadpool_parity audit_nan_injection \
             audit_memory_pool test_diamond_memory audit_flash_attention audit_modern_llm \
             audit_known_answer audit_regression_13c \
             audit_transformer_known_answer audit_fp32_path \
             audit_fault_injection audit_despot_probe audit_train_lib \
             audit_sum_axis_nd audit_train_bwd audit_train_converge \
             audit_sandbox audit_v7_hardening train_micromodel eval_verifier \
             train_verifier_head
# Despot V9: distill_prm800k is standalone (no liblancius.a) so it is not part
# of the instrumented-library gate; it runs under --selftest in make check.

check-sanitizers:
	@mkdir -p $(SAN_DIR)/obj $(SAN_DIR)/logs
	@echo "--- instrumenting the library (this is the part the old gate skipped) ---"
	@for f in src/*/*.c; do \
	    $(CC) $(SAN_CFLAGS) -c $$f -o $(SAN_DIR)/obj/`basename $$f .c`.o || exit 1; \
	 done
	@ar rcs $(SAN_DIR)/liblancius_san.a $(SAN_DIR)/obj/*.o
	@echo "--- building every audit against the instrumented library ---"
	@for a in $(SAN_AUDITS); do \
	    $(CC) $(SAN_CFLAGS) -o $(SAN_DIR)/$$a examples/$$a.c $(SAN_DIR)/liblancius_san.a $(SAN_LDFLAGS) \
	      || { echo "BUILD FAILED: $$a"; exit 1; }; \
	 done
	@echo "--- running every audit under ASan+UBSan+LSan ---"
	@fails=0; for a in $(SAN_AUDITS); do \
	    ASAN_OPTIONS=detect_leaks=1 UBSAN_OPTIONS=print_stacktrace=1 \
	      ./$(SAN_DIR)/$$a > $(SAN_DIR)/logs/$$a.run 2>&1; rc=$$?; \
	    if grep -qE "runtime error:|ERROR: (AddressSanitizer|LeakSanitizer)" $(SAN_DIR)/logs/$$a.run; then \
	      echo "SANITIZER FINDING in $$a:"; \
	      grep -E "runtime error:|ERROR: (AddressSanitizer|LeakSanitizer)" $(SAN_DIR)/logs/$$a.run | head -4; \
	      fails=$$((fails+1)); \
	    fi; \
	    if [ $$rc -ne 0 ]; then echo "EXIT NONZERO: $$a (rc=$$rc)"; fails=$$((fails+1)); fi; \
	 done; \
	 if [ $$fails -ne 0 ]; then echo "sanitizer gate FAILED ($$fails findings) in $(SAN_DIR)/logs"; exit 1; fi
	@echo "Sanitizer gate: all $(words $(SAN_AUDITS)) audits clean under ASan+UBSan+LSan."
	@echo "Restoring normal build (removing sanitizer instrumentation)..."
	$(MAKE) -B all
	@echo "Normal build restored. Safe to run 'make check' now."

# --- V7: strict UBSan pass (aborts on the first UB, incl. signed overflow
# and float-cast-overflow, which the combined gate does not fail on) ---
check-ubstrict:
	@./tools/audit/ubstrict_sweep.sh

# --- V7: external-oracle gate. Recomputes every kernel and graph-level op in
# NumPy/PyTorch/closed form and compares. Fails nonzero on any divergence. ---
check-oracle:
	@./tools/audit/oracle_gate.sh

# --- V7: mutation gate. Injects real defects and requires the gate to go red.
# A gate that cannot fail is decoration; this proves it can. ---
check-mutation:
	@./tools/audit/mutation_test.sh

# --- V12R2: physics gate. Pins IEEE 754, libm, GELU/attention/RoPE constants,
# FMA-contraction behaviour, kernel/scheduler determinism, the pool thread
# budget, and the v2 model CRC against an independent reference (ISO decimal
# arithmetic + zlib + the source papers) that never executes Lancius. ---
check-physics:
	@./tools/audit/physics_gate.sh

PREFIX ?= /usr/local
