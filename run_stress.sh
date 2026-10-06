#!/usr/bin/env bash
#
# Lancius stress / soak / fuzz runner.
#
# The binary stress tests have hardcoded iteration counts (stress_test:
# 100k arena cycles + 10k-node graph; soak_fuzz: 2k KV steps + 10k malformed
# binaries; fuzz_lancius: 500 graphs). The only knob any of them expose is
# fuzz_lancius's seed, so that is what this script sweeps: a long run is many
# seeds, not one long run.
#
# It also repeats the deterministic suites, because the bugs found during the
# 3463 integration only appeared on the *second* poll -- a single run cannot
# see state that leaks between invocations.
#
# Usage:
#   ./run_stress.sh                     # tier: full (default)
#   ./run_stress.sh --tier smoke        # ~1 min
#   ./run_stress.sh --tier soak         # soak + long fuzz sweep
#   ./run_stress.sh --seeds 200         # fuzz sweep size
#   ./run_stress.sh --repeat 3          # repeat deterministic suites
#   ./run_stress.sh --list              # show the plan and exit
#   ./run_stress.sh --dry-run           # show the plan, run nothing
#
# Exit codes:
#   0  everything requested passed
#   1  at least one real failure
#   2  bad usage
#   3  a required binary is missing (nothing was run)
#
# Results land in stress-logs/<UTC timestamp>/, one log per case.

set -u -o pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_DIR" || exit 3

# ----------------------------------------------------------------- defaults
TIER="full"
SEEDS=50
REPEAT=2
SEED_START=1000
JOBS=1
TIMEOUT_S=1800
KEEP_GOING=1
LIST_ONLY=0
DRY_RUN=0
# Track explicit flags so a tier preset never overrides what the user asked for.
SEEDS_SET=0
REPEAT_SET=0
# Debug knobs.
DEBUG=0                 # --debug: sanitizers + core dumps + gdb + verbose
SANITIZE="none"          # none | address | undefined | both
OPT_LEVEL=0              # -O0 by default; see the warning emitted at runtime
VERBOSE=0
BUILD_DIR=""             # non-empty => run these binaries instead of ./
CORE_DIR=""              # where core files land (debug mode only)
HAVE_GDB=0
CORES_ONLY=0            # --cores alone: core dumps, no rebuild, no sanitizer
LOG_ROOT="$REPO_DIR/stress-logs"

# Binaries that report their own pass/fail via exit code.
UNIT_TESTS=(stress_test test_torture soak_fuzz)

usage() {
    cat <<'USAGE'
Lancius stress / soak / fuzz runner.

The binary stress tests have hardcoded iteration counts (stress_test:
100k arena cycles + 10k-node graph; soak_fuzz: 2k KV steps + 10k malformed
binaries; fuzz_lancius: 500 graphs). The only knob any of them expose is
fuzz_lancius's seed, so that is what this script sweeps: a long run is many
seeds, not one long run.

It also repeats the deterministic suites, because the bugs found during the
3463 integration only appeared on the *second* poll -- a single run cannot
see state that leaks between invocations.

Usage:
  ./run_stress.sh                     # tier: full (default)
  ./run_stress.sh --tier smoke        # ~1 min
  ./run_stress.sh --tier soak         # soak + long fuzz sweep
  ./run_stress.sh --seeds 200         # fuzz sweep size
  ./run_stress.sh --repeat 3          # repeat deterministic suites
  ./run_stress.sh --jobs 4            # run cases in parallel
  ./run_stress.sh --list              # show the plan and exit
  ./run_stress.sh --dry-run           # show the plan, run nothing

Debug run (sanitizers + core dumps + gdb recipes):
  ./run_stress.sh --debug             # = --sanitize=address --cores -v
  ./run_stress.sh --sanitize both     # address + undefined
  ./run_stress.sh --cores             # core dumps only, no rebuild
  ./run_stress.sh -v                  # verbose (prints each fuzz tally)

  --debug rebuilds the stress binaries into build-debug/ with -g -O0 and
  the chosen sanitizers. It never touches the release binaries or
  liblancius.a in the repo root, so a later `make check` still tests the
  real build. UBSan is built with -fno-sanitize-recover=all and run with
  halt_on_error=1, because plain UBSan only prints and continues -- which
  would make an undefined-behaviour finding a silent pass.

  On failure the script prints the sanitizer summary and drops a
  <log>.gdb.cmds file next to the log for reproducing under gdb.

  The debug build defaults to -O0 on purpose. Measured with gcc+ASan here,
  a heap-buffer-overflow and a leak are both caught at -O0 and both go
  UNDETECTED at -O1, because the optimiser elides or vectorises the exact
  operations the sanitizer instruments. A green -O1 run looks clean and is
  not. Use --opt-level=N only when chasing a specific finding.

Exit codes: 0 all passed | 1 a real failure | 2 bad usage | 3 nothing ran
USAGE
}

die_usage() { echo "run_stress.sh: $*" >&2; usage >&2; exit 2; }

# Accept both "--opt value" and "--opt=value". Normalise into a fresh list
# first: appending in place would leave the un-split originals ahead of the
# split ones and the parser would reject them first.
if [ $# -gt 0 ]; then
    NORM=()
    for a in "$@"; do
        case "$a" in
            --*=*) NORM+=("${a%%=*}"); NORM+=("${a#*=}") ;;
            *)     NORM+=("$a") ;;
        esac
    done
    set -- "${NORM[@]}"
    unset NORM
fi

while [ $# -gt 0 ]; do
    case "$1" in
        --tier)      [ $# -ge 2 ] || die_usage "--tier needs a value"
                     TIER="$2"; shift 2 ;;
        --seeds)     [ $# -ge 2 ] || die_usage "--seeds needs a value"
                     SEEDS="$2"; SEEDS_SET=1; shift 2 ;;
        --repeat)    [ $# -ge 2 ] || die_usage "--repeat needs a value"
                     REPEAT="$2"; REPEAT_SET=1; shift 2 ;;
        --seed-start)[ $# -ge 2 ] || die_usage "--seed-start needs a value"
                     SEED_START="$2"; shift 2 ;;
        --jobs)      [ $# -ge 2 ] || die_usage "--jobs needs a value"
                     JOBS="$2"; shift 2 ;;
        --timeout)   [ $# -ge 2 ] || die_usage "--timeout needs a value"
                     TIMEOUT_S="$2"; shift 2 ;;
        --logs)      [ $# -ge 2 ] || die_usage "--logs needs a value"
                     LOG_ROOT="$2"; shift 2 ;;
        --keep-going) KEEP_GOING=1; shift ;;
        --fail-fast)  KEEP_GOING=0; shift ;;
        --debug)      DEBUG=1; VERBOSE=1; shift ;;
        --sanitize)   [ $# -ge 2 ] || die_usage "--sanitize needs a value"
                     case "$2" in
                         none|address|undefined|both) SANITIZE="$2" ;;
                         *) die_usage "--sanitize must be none, address, undefined or both" ;;
                     esac
                     [ "$SANITIZE" = none ] || VERBOSE=1
                     shift 2 ;;
        --opt-level) [ $# -ge 2 ] || die_usage "--opt-level needs a value"
                     OPT_LEVEL="$2"; shift 2 ;;
        --verbose|-v) VERBOSE=1; shift ;;
        --cores)      CORES_ONLY=1; VERBOSE=1; shift ;;
        --list)      LIST_ONLY=1; shift ;;
        --dry-run)   DRY_RUN=1; shift ;;
        -h|--help)   usage; exit 0 ;;
        *)           die_usage "unknown argument: $1" ;;
    esac
done

# --debug implies ASan unless a sanitizer was chosen explicitly.
# --cores on its own is NOT a debug build: it asks for core dumps against
# whatever binaries already exist, with no rebuild and no sanitizer. Setting
# DEBUG there used to drag ASan in with it, contradicting its own --help.
if [ "$DEBUG" -eq 1 ] && [ "$SANITIZE" = none ]; then
    SANITIZE=address
fi

# Warn loudly if the optimisation level is raised for a detection build.
# Measured on this toolchain (gcc, ASan):
#     -O0  heap-overflow detected, leaks reported
#     -O1  NEITHER detected  (optimiser elides/vectorises the instrumented ops)
#     -O2  NEITHER detected
# A green -O1 ASan run therefore proves much less than it looks like it does.
if [ "$SANITIZE" != none ] && [ "$OPT_LEVEL" -gt 0 ]; then
    echo "WARNING: --opt-level=$OPT_LEVEL weakens sanitizer detection." >&2
    echo "         At -O1 and above GCC elides or vectorises the very" >&2
    echo "         operations ASan instruments: measured on this toolchain," >&2
    echo "         a heap-buffer-overflow and a leak both go UNDETECTED at" >&2
    echo "         -O1, and both are caught at -O0. Use -O0 unless you are" >&2
    echo "         debugging a specific finding." >&2
fi

case "$TIER" in
    smoke) [ "$SEEDS_SET"   -eq 1 ] || SEEDS=5
           [ "$REPEAT_SET"  -eq 1 ] || REPEAT=1 ;;
    full)  : ;;
    soak)  [ "$SEEDS_SET"   -eq 1 ] || SEEDS=$(( SEEDS < 200 ? 200 : SEEDS )) ;;
    *)     die_usage "--tier must be smoke, full or soak" ;;
esac
case "$SEEDS$REPEAT$JOBS$SEED_START$TIMEOUT_S$OPT_LEVEL" in
    *[!0-9]*) die_usage "numeric options must be integers" ;;
esac
[ "$SEEDS" -ge 1 ] || die_usage "--seeds must be >= 1"
[ "$REPEAT" -ge 1 ] || die_usage "--repeat must be >= 1"
[ "$JOBS"  -ge 1 ] || die_usage "--jobs must be >= 1"

# ------------------------------------------------------------------ build
#
# In debug mode the stress binaries are rebuilt from source into their own
# directory with sanitizer + frame-pointer flags.
#
# This is deliberately NOT `make test_asan`: that target rebuilds the very
# same stress_test / liblancius.a / src/*.o names with different flags, so
# running it would silently replace your release build and any later
# `make check` would then be testing an -O0 ASan binary. A separate build
# dir keeps the two from ever colliding.
#
# The ISA flags (-mavx2 -mfma) are kept identical to the release build so the
# debug run exercises the same code paths, not a different variant.
STRESS_BINS=(stress_test test_torture soak_fuzz fuzz_lancius)

build_debug_binaries() {
    local dir="$1"
    [ -n "$dir" ] || return 0

    local san_flags="" opt_flags="-g -O$OPT_LEVEL -fno-omit-frame-pointer"
    case "$SANITIZE" in
        address)   san_flags="-fsanitize=address -fno-common" ;;
        undefined) san_flags="-fsanitize=undefined -fno-sanitize-recover=all" ;;
        both)      san_flags="-fsanitize=address,undefined -fno-common -fno-sanitize-recover=all" ;;
        none)      san_flags="" ;;
    esac

    local base="-Wall -Wextra -mavx2 -mfma -fopenmp -std=c11 -I./include -fPIC"

    # The library sources are read from the makefile so this stays in sync.
    local srcs
    srcs="$(sed -n '/^SRCS = /,/^$/p' makefile | tr -d '\\' \
            | sed 's/^SRCS = //' | tr -s ' ' '\n' | grep '\.c$' || true)"
    if [ -z "$srcs" ]; then
        echo "run_stress.sh: could not read SRCS from the makefile" >&2
        return 1
    fi

    mkdir -p "$dir/obj" || return 1
    echo "building debug binaries -> $dir (sanitize=$SANITIZE)"

    local objs="" s o
    for s in $srcs; do
        o="$dir/obj/$(echo "$s" | tr '/' '_' | sed 's/\.c$/.o/')"
        # shellcheck disable=SC2086
        ${CC:-cc} $base $opt_flags $san_flags -c "$s" -o "$o" || return 1
        objs="$objs $o"
    done
    # shellcheck disable=SC2086
    ar rcs "$dir/liblancius.a" $objs || return 1

    local b
    for b in "${STRESS_BINS[@]}"; do
        local ex="examples/$b.c"
        [ -f "$ex" ] || { echo "run_stress.sh: missing source $ex" >&2; return 1; }
        # shellcheck disable=SC2086
        ${CC:-cc} $base $opt_flags $san_flags -o "$dir/$b" "$ex" "$dir/liblancius.a" \
            $san_flags -lm -fopenmp -lpthread || return 1
    done
    return 0
}

# ------------------------------------------------------------------ plan
# Build the case list up front so --list/--dry-run can show it and so a
# missing binary is detected before anything runs.
CASES=()

for t in "${UNIT_TESTS[@]}"; do
    for r in $(seq 1 "$REPEAT"); do
        CASES+=("unit:$t:$r")
    done
done
for s in $(seq 0 $(( SEEDS - 1 ))); do
    CASES+=("fuzz:fuzz_lancius:$(( SEED_START + s ))")
done

# Collect the binaries the plan needs, so one missing build does not turn
# into a wall of confusing "no such file" errors.
# Debug mode builds into its own directory before the dependency check, so
# the check is satisfied by the debug build rather than the release one.
# Skipped for --list/--dry-run: those describe the plan and must not compile.
if [ "$SANITIZE" != none ] && [ "$DRY_RUN" -eq 0 ] && [ "$LIST_ONLY" -eq 0 ]; then
    BUILD_DIR="${BUILD_DIR:-$REPO_DIR/build-debug}"
    if ! build_debug_binaries "$BUILD_DIR"; then
        echo "run_stress.sh: debug build failed" >&2
        exit 3
    fi
fi

REQUIRED=("${STRESS_BINS[@]}")
MISSING=()
if [ -n "$BUILD_DIR" ]; then
    for b in "${REQUIRED[@]}"; do
        [ -x "$BUILD_DIR/$b" ] || MISSING+=("$b")
    done
else
    for b in "${REQUIRED[@]}"; do
        [ -x "./$b" ] || MISSING+=("$b")
    done
fi

if [ "$LIST_ONLY" -eq 1 ]; then
    echo "plan (tier=$TIER seeds=$SEEDS repeat=$REPEAT jobs=$JOBS timeout=${TIMEOUT_S}s)"
    printf '  %s\n' "${CASES[@]}"
    if [ "$SANITIZE" != none ]; then
        echo
        echo "would build debug binaries (sanitize=$SANITIZE) into: ${BUILD_DIR:-$REPO_DIR/build-debug}"
    fi
    if [ ${#MISSING[@]} -gt 0 ]; then
        echo
        echo "MISSING BINARIES: ${MISSING[*]}"
        echo "build them with: make ${REQUIRED[*]}"
    fi
    exit 0
fi

if [ ${#MISSING[@]} -gt 0 ]; then
    echo "run_stress.sh: missing binaries: ${MISSING[*]}" >&2
    echo "  build them first:  make ${REQUIRED[*]}" >&2
    exit 3
fi

# ------------------------------------------------------------------ logging
TS="$(date -u +%Y%m%dT%H%M%SZ)"
LOG_DIR="$LOG_ROOT/$TS"
mkdir -p "$LOG_DIR" || { echo "cannot create $LOG_DIR" >&2; exit 3; }

if [ -n "$BUILD_DIR" ]; then
    BUILD_DIR="$(cd "$BUILD_DIR" && pwd)"
fi

# ------------------------------------------------- debug: cores and gdb
#
# Core dumps: raise the limit for this shell (and children). Note that this
# kernel pipes cores to apport via /proc/sys/kernel/core_pattern, so the
# file itself may not appear in CORE_DIR -- apport is what receives it.
if [ "$DEBUG" -eq 1 ] || [ "$CORES_ONLY" -eq 1 ]; then
    CORE_DIR="$LOG_DIR/cores"
    mkdir -p "$CORE_DIR" || { echo "cannot create $CORE_DIR" >&2; exit 3; }
    ulimit -c unlimited 2>/dev/null || true
    HARD_LIMIT="$(ulimit -Hc 2>/dev/null || echo unknown)"
    SOFT_LIMIT="$(ulimit -Sc 2>/dev/null || echo unknown)"
fi

command -v gdb >/dev/null 2>&1 && HAVE_GDB=1

# ------------------------------------------------- debug: sanitizer options
#
# UBSan by default PRINTS AND CONTINUES, which would make an undefined-behaviour
# finding a silent pass -- exactly the false-green shape this repo keeps
# getting bitten by. halt_on_error + no-recover turns it into a real failure.
if [ "$SANITIZE" != none ]; then
    export ASAN_OPTIONS="detect_leaks=1:abort_on_error=1:halt_on_error=1:strict_string_checks=1:detect_stack_use_after_return=1:print_stacktrace=1"
    export UBSAN_OPTIONS="halt_on_error=1:abort_on_error=1:print_stacktrace=1"
    export LSAN_OPTIONS="report_objects=1"
fi

# Sanitizer builds are slower; a timeout tuned for -O3 will produce false
# TIMEOUT verdicts rather than real ones.
if [ "$SANITIZE" != none ] && [ "$TIMEOUT_S" -eq 1800 ]; then
    TIMEOUT_S=5400
fi

# Binaries under test: the debug build dir if there is one, else the repo root.
if [ -n "$BUILD_DIR" ]; then BIN_PREFIX="$BUILD_DIR/"; else BIN_PREFIX="./"; fi

SUMMARY="$LOG_DIR/summary.tsv"
# Header in both serial and parallel mode so the tsv is always parseable.
printf 'kind\tname\targ\tverdict\tlog\n' > "$SUMMARY" 2>/dev/null || true

echo "=============================================================="
echo " Lancius stress run"
echo " tier=$TIER  seeds=$SEEDS  repeat=$REPEAT  jobs=$JOBS  timeout=${TIMEOUT_S}s"
echo " logs: $LOG_DIR"
if [ -n "$BUILD_DIR" ]; then
    echo " build: $BUILD_DIR (sanitize=$SANITIZE -O$OPT_LEVEL, release binaries untouched)"
else
    echo " build: repository root (release binaries)"
fi
if [ "$SANITIZE" != none ]; then
    echo " ASAN_OPTIONS=$ASAN_OPTIONS"
    echo " UBSAN_OPTIONS=$UBSAN_OPTIONS"
fi
if [ -n "$CORE_DIR" ]; then
    echo " cores: soft=$SOFT_LIMIT hard=$HARD_LIMIT -> $CORE_DIR"
    echo " gdb:   $([ "$HAVE_GDB" -eq 1 ] && command -v gdb || echo 'not installed')"
fi
echo "=============================================================="

# Record the exact environment so a failure is reproducible later.
{
    echo "timestamp: $TS"
    echo "tier=$TIER seeds=$SEEDS repeat=$REPEAT jobs=$JOBS timeout=$TIMEOUT_S"
    echo "sanitize=$SANITIZE build_dir=${BUILD_DIR:-<repo root>}"
    echo "ASAN_OPTIONS=${ASAN_OPTIONS:-}"
    echo "UBSAN_OPTIONS=${UBSAN_OPTIONS:-}"
    echo "--- uname ---"
    uname -a
    echo "--- compiler ---"
    ${CC:-cc} --version 2>/dev/null | head -1
    echo "--- git ---"
    git rev-parse --short HEAD 2>/dev/null || echo "(not a git repo)"
    git status --porcelain 2>/dev/null | head -20
} > "$LOG_DIR/environment.txt" 2>&1

TOTAL=${#CASES[@]}
DONE=0
FAILED=0
FAILED_NAMES=()

# A case is a failure if the binary exits non-zero, if it is killed by the
# timeout, or if its output contains a failure marker. The marker scan is
# belt-and-braces: every one of these binaries aggregates its own results
# into the exit code, but "prints FAIL, exits 0" is the exact failure shape
# this repo has been bitten by before, so it is not left to trust alone.
FAIL_MARKERS='❌ FAIL|\[FUZZ FAIL\]|FAIL:'

run_case() {
    local kind="$1" name="$2" arg="$3"
    local log="$LOG_DIR/${kind}_${name}_${arg}.log"
    local start end rc status verdict

    start=$(date +%s)
    if [ "$kind" = "unit" ]; then
        printf '  [%*d/%d] %-14s run %s ... ' "$WIDTH" "$DONE" "$TOTAL" "$name" "$arg"
    else
        printf '  [%*d/%d] %-14s seed %-8s ... ' "$WIDTH" "$DONE" "$TOTAL" "$name" "$arg"
    fi

    if [ "$DRY_RUN" -eq 1 ]; then
        echo "DRY-RUN"
        DONE=$(( DONE + 1 ))
        return 0
    fi

    # A core file (if the kernel produces one) lands beside the log.
    if [ -n "$CORE_DIR" ]; then
        ( cd "$CORE_DIR" && ulimit -c unlimited 2>/dev/null
          if [ "$kind" = "unit" ]; then
              timeout --signal=TERM "$TIMEOUT_S" "${BIN_PREFIX}${name}" >"$log" 2>&1
          else
              timeout --signal=TERM "$TIMEOUT_S" "${BIN_PREFIX}${name}" "$arg" >"$log" 2>&1
          fi
        )
        rc=$?
        mv "$CORE_DIR"/core* "$log".core 2>/dev/null || true
    elif [ "$kind" = "unit" ]; then
        timeout --signal=TERM "$TIMEOUT_S" "${BIN_PREFIX}$name" >"$log" 2>&1
        rc=$?
    else
        timeout --signal=TERM "$TIMEOUT_S" "${BIN_PREFIX}$name" "$arg" >"$log" 2>&1
        rc=$?
    fi
    end=$(date +%s)

    if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
        status="TIMEOUT"; verdict="FAIL"
    elif [ "$rc" -ne 0 ]; then
        status="exit=$rc"; verdict="FAIL"
    elif grep -Eq "$FAIL_MARKERS" "$log"; then
        # Non-zero-free but the run reported a failure: surface it loudly
        # rather than calling it green.
        status="exit=0+FAILTEXT"; verdict="FAIL"
    else
        status="ok"; verdict="PASS"
    fi

    printf '%-4s (%ss)  %s\n' "$verdict" "$(( end - start ))" "$status"
    printf '%s\t%s\t%s\t%s\t%s\n' "$kind" "$name" "$arg" "$verdict" "$log" >> "$SUMMARY"

    if [ "$verdict" = "FAIL" ]; then
        DONE=$(( DONE + 1 ))
        FAILED=$(( FAILED + 1 ))
        FAILED_NAMES+=("${kind}:${name}:${arg}")
        report_failure "$kind" "$name" "$arg" "$log" "$status"
        if [ "$KEEP_GOING" -eq 0 ]; then
            echo
            echo "fail-fast: stopping after first failure. see $log"
            finish 1
        fi
    else
        DONE=$(( DONE + 1 ))
        if [ "$VERBOSE" -eq 1 ] && [ "$kind" = fuzz ]; then
            printf '         last: %s\n' "$(grep -E "PASSED \| [0-9]+ FAILED" "$log" | tail -1)"
        fi
    fi
    return 0
}

# On failure, surface the diagnostic that actually matters: a sanitizer
# report, or a gdb command file to re-run the case under a debugger.
report_failure() {
    local kind="$1" name="$2" arg="$3" log="$4" status="$5"

    if [ "$SANITIZE" != none ] && grep -qE "ERROR: (Address|Leak)Sanitizer|runtime error:" "$log" 2>/dev/null; then
        echo "         sanitizer report (first lines):"
        grep -nE "ERROR: (Address|Leak)Sanitizer|runtime error:|SUMMARY:" "$log" \
            | head -4 | sed 's/^/           /'
    fi

    if [ "$HAVE_GDB" -eq 1 ]; then
        local gd="${log%.log}.gdb.cmds"
        {
            # This file is fed to gdb with -x, so it must contain only gdb
            # commands. `file` makes it self-contained: no need to be in the
            # build directory.
            if [ "$kind" = unit ]; then
                echo "file ${BIN_PREFIX}${name}"
            else
                echo "file ${BIN_PREFIX}${name}"
                echo "set args $arg"
            fi
            echo "set pagination off"
            echo "set confirm off"
            echo "run"
            echo "echo \\n=== backtrace ===\\n"
            echo "bt full"
            echo "echo \\n=== all threads ===\\n"
            echo "thread apply all bt"
            echo "echo \\n=== registers ===\\n"
            echo "info registers"
            echo "quit"
        } > "$gd"
        echo "         debugger recipe: gdb --batch -x $gd"
    else
        echo "         gdb not installed; no debugger recipe generated"
    fi

    if [ -n "$CORE_DIR" ]; then
        local cores
        cores="$(ls -1 "$CORE_DIR" 2>/dev/null | wc -l)"
        echo "         core files in $CORE_DIR: $cores"
        echo "         (note: this kernel pipes cores to apport, so they may not appear here)"
    fi
}

finish() {
    local rc="${1:-0}"
    echo
    echo "--------------------------------------------------------------"
    printf ' ran %d case(s), %d failure(s) in %ss\n' "$DONE" "$FAILED" \
        "$(( $(date +%s) - T_START ))"
    if [ "$FAILED" -gt 0 ]; then
        echo " FAILURES:"
        for n in "${FAILED_NAMES[@]}"; do echo "   - $n"; done
        echo " logs: $LOG_DIR"
        [ "$SUMMARY" ] && echo " summary: $SUMMARY"
    else
        echo " all cases passed"
        [ "$DRY_RUN" -eq 0 ] && echo " logs: $LOG_DIR"
    fi
    echo "--------------------------------------------------------------"
    exit "$rc"
}

T_START=$(date +%s)
WIDTH=${#TOTAL}

if [ "$JOBS" -eq 1 ]; then
    for entry in "${CASES[@]}"; do
        IFS=':' read -r kind name arg <<< "$entry"
        run_case "$kind" "$name" "$arg"
    done
else
    # Parallel mode: each case writes its own log; verdict collection is done
    # afterwards from the summary file so the counters stay single-threaded.
    echo " (parallel: $JOBS at a time)"
    for entry in "${CASES[@]}"; do
        IFS=':' read -r kind name arg <<< "$entry"
        log="$LOG_DIR/${kind}_${name}_${arg}.log"
        [ "$DRY_RUN" -eq 1 ] && { echo "  DRY-RUN $kind $name $arg"; continue; }
        ( if [ "$kind" = "unit" ]; then
              timeout --signal=TERM "$TIMEOUT_S" "${BIN_PREFIX}$name" >"$log" 2>&1
          else
              timeout --signal=TERM "$TIMEOUT_S" "${BIN_PREFIX}$name" "$arg" >"$log" 2>&1
          fi
          echo "$?" > "$log.rc" ) &
        while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n 2>/dev/null || sleep 0.2; done
        echo "  started $kind $name $arg"
    done
    wait
    if [ "$DRY_RUN" -eq 0 ]; then
        # header already written at startup
        while IFS=':' read -r kind name arg; do
            log="$LOG_DIR/${kind}_${name}_${arg}.log"
            rcfile="$log.rc"
            if [ -f "$rcfile" ]; then rc=$(cat "$rcfile"); else rc=99; fi
            if [ "$rc" -eq 124 ] || [ "$rc" -eq 137 ]; then
                verdict="FAIL"; st="TIMEOUT"
            elif [ "$rc" -ne 0 ]; then
                verdict="FAIL"; st="exit=$rc"
            elif grep -Eq "$FAIL_MARKERS" "$log" 2>/dev/null; then
                verdict="FAIL"; st="exit=0+FAILTEXT"
            else
                verdict="PASS"; st="ok"
            fi
            DONE=$(( DONE + 1 ))
            printf '  %-4s %-14s %-8s %s\n' "$verdict" "$name" "$arg" "$st"
            printf '%s\t%s\t%s\t%s\t%s\n' "$kind" "$name" "$arg" "$verdict" "$log" >> "$SUMMARY"
            [ "$verdict" = "FAIL" ] && { FAILED=$(( FAILED + 1 )); FAILED_NAMES+=("${kind}:${name}:${arg}"); }
        done < <(printf '%s\n' "${CASES[@]}")
    fi
fi

if [ "$DRY_RUN" -eq 1 ]; then
    echo
    echo "dry run: $TOTAL case(s) planned, nothing executed."
    exit 0
fi

[ "$FAILED" -eq 0 ] && finish 0 || finish 1