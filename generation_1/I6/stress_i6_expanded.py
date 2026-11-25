#!/usr/bin/env python3
"""Crossed, per-case I6 stress test over all 150 math algorithms.

Each sampled parameter set is tested against every realizable verdict level.
The harness records overall and local-step predictions, condition accuracy,
confusion matrices, sampled parameter coverage, and compressed case manifests.
"""
import argparse
import csv
import gzip
import json
import os
import random
import sys
from collections import Counter, defaultdict
from fractions import Fraction as F

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "I1"))
sys.path.insert(0, os.path.join(HERE, "..", "I3"))
sys.path.insert(0, os.path.join(REPO, "tools", "prm"))
sys.path.insert(0, REPO)

import run_big_i6 as big
import verifier5 as V
from bench_gen1 import LEVELS, LVLNAME, fwd_batch, load_gen1
from algos import ALGORITHMS
from algos_gen1_ext import EXT_ALGOS
from algos_gen1_big import EXT2_ALGOS
from algos_i3 import EXT3_ALGOS

ALL150 = list(ALGORITHMS) + list(EXT_ALGOS) + list(EXT2_ALGOS) + list(EXT3_ALGOS)
INPUT_SOURCES = ("algorithm_builder", "explicit_edge")
DIMENSIONS = (
    "algorithm", "family", "input_source", "target_level", "candidate_type",
    "edge_case", "depth_bucket", "sampled_coefficient_boundary",
    "coefficient_sign", "coefficient_zero", "coefficient_fractional",
    "coefficient_magnitude", "output_sign", "output_zero",
    "output_fractional", "step_pattern", "fault_size_bucket",
)
GROUPINGS = [(d,) for d in DIMENSIONS] + [
    ("algorithm", "target_level"),
    ("algorithm", "input_source"),
    ("algorithm", "candidate_type"),
    ("algorithm", "depth_bucket"),
    ("algorithm", "coefficient_sign"),
    ("algorithm", "coefficient_magnitude"),
    ("input_source", "target_level"),
    ("candidate_type", "target_level"),
    ("coefficient_sign", "target_level"),
    ("depth_bucket", "target_level"),
    ("algorithm", "input_source", "target_level"),
    ("algorithm", "candidate_type", "target_level"),
    ("algorithm", "coefficient_sign", "target_level"),
]
STEP_EXTRA_DIMENSIONS = ("step_position", "step_status", "step_class")
STEP_GROUPINGS = GROUPINGS + [(d,) for d in STEP_EXTRA_DIMENSIONS] + [
    ("algorithm", "step_position"),
    ("algorithm", "step_status"),
    ("target_level", "step_position"),
    ("candidate_type", "step_status"),
    ("algorithm", "step_class"),
    ("algorithm", "target_level", "step_position"),
    ("algorithm", "candidate_type", "step_status"),
]
CASE_COLUMNS = (
    "case_id", "input_id", "algorithm", "family", "input_source",
    "target_level", "candidate_type", "edge_case", "depth_bucket",
    "sampled_coefficient_boundary",
    "coefficient_sign", "coefficient_zero", "coefficient_fractional",
    "coefficient_magnitude", "output_sign", "output_zero",
    "output_fractional", "step_pattern", "fault_size_bucket",
    "base_params_json", "params_json",
    "expected_steps_json", "reported_steps_json", "reported_structure_json",
)


def numeric(v):
    return isinstance(v, (int, float, F)) and not isinstance(v, bool)


def as_fraction(v):
    if isinstance(v, F):
        return v
    if isinstance(v, int):
        return F(v)
    return F(str(v))


def value_text(v):
    if v is None:
        return ""
    q = as_fraction(v)
    return str(q.numerator) if q.denominator == 1 else f"{q.numerator}/{q.denominator}"


def build_stress_cell(algo, seed):
    rng = big.CellRng(seed)
    last = None
    rejected = 0
    for _ in range(24):
        try:
            params0, steps = algo.build(rng)
        except Exception as exc:
            last = type(exc).__name__
            continue
        try:
            exact_values = [F(step["value"]) for step in steps]
            if steps and len(steps) <= V.MAX_STEPS and len(exact_values) == len(steps):
                candidates = []
                for li, level in enumerate(LEVELS):
                    rep, struct, kind = V.make_candidate(
                        algo, params0, steps,
                        big.CellRng(seed + 1 + li * 104729), level)
                    V.check(algo, params0, steps, rep, struct)
                    candidates.append((level, rep, struct, kind))
                return params0, steps, candidates, rejected
            last = "invalid step count"
        except Exception as exc:
            rejected += 1
            last = type(exc).__name__
    raise ValueError("could not build a complete valid input (%s; rejected=%d)"
                     % (last, rejected))


def sign_profile(values):
    vals = [as_fraction(v) for v in values]
    pos = any(v > 0 for v in vals)
    neg = any(v < 0 for v in vals)
    zero = any(v == 0 for v in vals)
    if not vals:
        return "no_numeric_values"
    if zero and not pos and not neg:
        return "zero_only"
    if pos and neg:
        return "mixed_signs_with_zero" if zero else "mixed_signs"
    if pos:
        return "positive_with_zero" if zero else "positive_only"
    return "negative_with_zero" if zero else "negative_only"


def magnitude_bucket(values):
    mag = max((abs(as_fraction(v)) for v in values), default=F(0))
    if mag < 1:
        return "<1"
    if mag < 10:
        return "1_to_lt10"
    if mag < 100:
        return "10_to_lt100"
    return "100_plus"


def depth_bucket(n):
    if n == 1:
        return "1"
    if n <= 4:
        return "2_to_4"
    if n <= 8:
        return "5_to_8"
    return "9_to_12"


def fault_bucket(kind, chk):
    if kind == "none":
        return "none"
    if kind == "undecidable":
        return "missing_step"
    metric = chk["struct_rel"] if kind == "fatal" else chk["max_rel"]
    if metric <= 0.05:
        return "le_5pct"
    if metric < 0.25:
        return "5_to_25pct"
    if metric < 1:
        return "25_to_100pct"
    return "ge_100pct"


def scenario_tags(algo, params, steps, chk, target_level, kind,
                  input_source, edge_case, sampled_boundary="pending"):
    coef = [params[k] for k in algo.structure_keys
            if k in params and numeric(params[k])]
    outputs = [s["value"] for s in steps]
    step_signs = [1 if as_fraction(v) > 0 else -1 if as_fraction(v) < 0 else 0
                  for v in outputs]
    signs = {v for v in step_signs if v}
    if not signs:
        pattern = "all_zero"
    elif len(signs) == 2:
        pattern = "sign_changes" if any(
            a * b < 0 for a, b in zip(step_signs, step_signs[1:])) else "both_signs"
    else:
        pattern = "positive_only" if 1 in signs else "negative_only"
    coeff = [as_fraction(v) for v in coef]
    out = [as_fraction(v) for v in outputs]
    return {
        "algorithm": algo.name,
        "family": algo.family,
        "input_source": input_source,
        "sampled_coefficient_boundary": sampled_boundary,
        "target_level": f"{target_level:+.1f}",
        "candidate_type": kind,
        "edge_case": edge_case,
        "depth_bucket": depth_bucket(len(steps)),
        "coefficient_sign": sign_profile(coef),
        "coefficient_zero": "yes" if any(v == 0 for v in coeff) else "no",
        "coefficient_fractional": "yes" if any(v.denominator != 1 for v in coeff) else "no",
        "coefficient_magnitude": magnitude_bucket(coef),
        "output_sign": sign_profile(outputs),
        "output_zero": "yes" if any(v == 0 for v in out) else "no",
        "output_fractional": "yes" if any(v.denominator != 1 for v in out) else "no",
        "step_pattern": pattern,
        "fault_size_bucket": fault_bucket(kind, chk),
    }


def encode_axis(values):
    categories = sorted(set(values))
    lut = {v: i for i, v in enumerate(categories)}
    return np.fromiter((lut[v] for v in values), dtype=np.int32, count=len(values)), categories


def group_rows(model, row_type, axis_names, axes_encoded, pred, truth, verdict,
               row_indices=None):
    rows = []
    abs_err = np.abs(verdict - np.asarray(LEVELS, dtype=np.float64)[truth])
    sq_err = abs_err * abs_err
    groupings = GROUPINGS if row_type == "overall" else STEP_GROUPINGS
    selected_axes = {}
    local_axes = set(STEP_EXTRA_DIMENSIONS) | {"row_class"}
    for name in axis_names:
        codes, categories = axes_encoded[name]
        if row_indices is not None and name not in local_axes:
            codes = codes[row_indices]
        selected_axes[name] = (codes, categories)
    gates_ok = True
    for grouping in groupings:
        codes = np.zeros(len(truth), dtype=np.int64)
        product = 1
        for axis in grouping:
            arr, categories = selected_axes[axis]
            codes = codes * len(categories) + arr
            product *= len(categories)
        observed, inverse = np.unique(codes, return_inverse=True)
        n = np.bincount(inverse)
        correct = np.bincount(inverse, weights=(pred == truth).astype(np.int64))
        mae_sum = np.bincount(inverse, weights=abs_err)
        mse_sum = np.bincount(inverse, weights=sq_err)
        for group_id, code in enumerate(observed):
            remainder = int(code)
            decoded = [None] * len(grouping)
            for pos in range(len(grouping) - 1, -1, -1):
                _, cats = selected_axes[grouping[pos]]
                decoded[pos] = cats[remainder % len(cats)]
                remainder //= len(cats)
            cnt = int(n[group_id])
            accuracy = float(correct[group_id] / cnt)
            mae = float(mae_sum[group_id] / cnt)
            mse = float(mse_sum[group_id] / cnt)
            gate = mae < 1e-3 and mse < 1e-3
            gates_ok &= gate
            condition = "|".join("%s=%s" % pair
                                 for pair in zip(grouping, decoded))
            rows.append((model, row_type, "+".join(grouping), condition,
                         cnt, int(correct[group_id]), 100.0 * accuracy,
                         mae, mse, "PASS" if gate else "FAIL"))
    return rows, gates_ok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs-per-algorithm", type=int, default=1000,
                    help="valid parameter sets sampled from each algorithm's own builder")
    ap.add_argument("--seed", type=int, default=20261127)
    ap.add_argument("--models", default=big.DEFAULT_EXPERTS)
    ap.add_argument("--musd", default=big.DEFAULT_MUSD)
    ap.add_argument("--outdir", default="temp/i6-expanded-stress")
    ap.add_argument("--strict-targets", action="store_true",
                    help="exit nonzero if any reported condition has MAE/MSE >= 1e-3")
    ap.add_argument("--require-perfect-accuracy", action="store_true",
                    help="also exit nonzero if any aggregate or condition accuracy is below 100%%")
    args = ap.parse_args()
    draws = args.inputs_per_algorithm
    if draws < 1:
        ap.error("--inputs-per-algorithm must be positive")
    if not os.path.isfile(args.musd):
        ap.error("I6 normalizer not found: " + args.musd)
    os.makedirs(args.outdir, exist_ok=True)

    rng = random.Random(args.seed)
    overall_bytes = bytearray()
    step_bytes = bytearray()
    case_columns = {key: [] for key in CASE_COLUMNS}
    step_to_case = []
    step_indices = []
    step_truth = []
    step_positions = []
    step_status = []
    step_values = []
    report_values = []
    build_coverage = Counter()
    dropped_levels = Counter()
    candidate_failures = Counter()
    rejected_build_trials = Counter()
    build_failures = Counter()
    sample_param_stats = defaultdict(lambda: {"n": 0, "zero": 0, "negative": 0,
                                               "positive": 0, "fractional": 0,
                                               "minimum": None, "maximum": None,
                                               "unique": set()})
    case_count = 0
    input_count = 0

    def record_input_params(algo, params, source):
        for key, value in params.items():
            if not numeric(value):
                continue
            v = as_fraction(value)
            st = sample_param_stats[(algo.name, key, source)]
            st["n"] += 1
            st["zero"] += v == 0
            st["negative"] += v < 0
            st["positive"] += v > 0
            st["fractional"] += v.denominator != 1
            st["minimum"] = v if st["minimum"] is None else min(st["minimum"], v)
            st["maximum"] = v if st["maximum"] is None else max(st["maximum"], v)
            st["unique"].add(v)

    def add_scenario(input_id, algo, params, steps, rep, struct, kind,
                     target_level, input_source, edge_case):
        nonlocal case_count
        chk = V.check(algo, params, steps, rep, struct)
        if not V.level_realized(algo, params, steps, rep, struct, target_level):
            dropped_levels[(algo.name, input_source, target_level)] += 1
            return False
        tags = scenario_tags(algo, params, steps, chk, target_level, kind,
                             input_source, edge_case)
        overall = V.row_features_i6(algo, chk, -1)
        if V.i6_label_from_features(overall) != LEVELS.index(target_level):
            raise AssertionError("I6 overall label provenance mismatch: %s" % algo.name)
        overall_bytes.extend(bytes(int(v) for v in overall))
        case_id = case_count
        case_count += 1
        expected = [s["value"] for s in steps]
        expected_text = [value_text(v) for v in expected]
        report_text = [value_text(v) for v in rep]
        struct_text = {k: value_text(struct[k]) for k in algo.structure_keys
                       if k in struct and numeric(struct[k])}
        record = {
            "case_id": case_id,
            "input_id": input_id,
            "algorithm": algo.name,
            "family": algo.family,
            "input_source": input_source,
            "target_level": tags["target_level"],
            "candidate_type": kind,
            "edge_case": edge_case,
            "depth_bucket": tags["depth_bucket"],
            "sampled_coefficient_boundary": tags["sampled_coefficient_boundary"],
            "coefficient_sign": tags["coefficient_sign"],
            "coefficient_zero": tags["coefficient_zero"],
            "coefficient_fractional": tags["coefficient_fractional"],
            "coefficient_magnitude": tags["coefficient_magnitude"],
            "output_sign": tags["output_sign"],
            "output_zero": tags["output_zero"],
            "output_fractional": tags["output_fractional"],
            "step_pattern": tags["step_pattern"],
            "fault_size_bucket": tags["fault_size_bucket"],
            "base_params_json": json.dumps(
                {k: value_text(v) for k, v in params.items()},
                sort_keys=True, separators=(",", ":")),
            "params_json": json.dumps({k: value_text(v) for k, v in params.items()},
                                       sort_keys=True, separators=(",", ":")),
            "expected_steps_json": json.dumps(expected_text, separators=(",", ":")),
            "reported_steps_json": json.dumps(report_text, separators=(",", ":")),
            "reported_structure_json": json.dumps(struct_text, sort_keys=True,
                                                   separators=(",", ":")),
        }
        for key in CASE_COLUMNS:
            case_columns[key].append(record[key])
        for si, s in enumerate(steps):
            row = V.row_features_i6(algo, chk, si)
            label = V.step_label(target_level, kind, chk, si)
            if V.i6_label_from_features(row) != label:
                raise AssertionError("I6 step label provenance mismatch: %s step %d"
                                     % (algo.name, si))
            step_bytes.extend(bytes(int(v) for v in row))
            step_to_case.append(case_id)
            step_indices.append(si)
            step_truth.append(label)
            if len(steps) == 1:
                step_positions.append("single")
            elif si == 0:
                step_positions.append("first")
            elif si == len(steps) - 1:
                step_positions.append("final")
            else:
                step_positions.append("middle")
            if chk["per_step"][si]["missing"]:
                step_status.append("missing")
            elif chk["per_step"][si]["exact"]:
                step_status.append("exact")
            else:
                step_status.append("wrong")
            step_values.append(expected_text[si])
            report_values.append(report_text[si])
        return True

    print("Expanded I6 stress: %d algorithms x %d valid inputs x %d verdict labels" %
          (len(ALL150), draws, len(LEVELS)), flush=True)
    for ai, algo in enumerate(ALL150):
        for _ in range(draws):
            seed = rng.randrange(2 ** 30)
            try:
                params, steps, candidates, rejected = build_stress_cell(algo, seed)
            except Exception as exc:
                build_failures[(algo.name, type(exc).__name__)] += 1
                continue
            rejected_build_trials[algo.name] += rejected
            build_coverage[algo.name] += 1
            record_input_params(algo, params, "algorithm_builder")
            input_id = input_count
            input_count += 1
            for level, rep, struct, kind in candidates:
                add_scenario(input_id, algo, params, steps, rep, struct, kind,
                             level, "algorithm_builder", "randomized")
        if (ai + 1) % 15 == 0:
            print("  generated inputs for %d/%d algorithms (%d labelled cases, %d step rows)" %
                  (ai + 1, len(ALL150), case_count, len(step_truth)), flush=True)

    # Explicit edge inputs from the I6 corpus generator: empty derivation,
    # identical points, zero-valued identity, and large coefficients.
    from gen_data_i6 import edge_rows
    edge_rng = big.CellRng(args.seed ^ 0xED6E)
    for algo, params, steps, fixed_rep, fixed_struct, edge_name, fixed_level in edge_rows(edge_rng):
        record_input_params(algo, params, "explicit_edge")
        input_id = input_count
        input_count += 1
        levels = [fixed_level] if fixed_rep is not None else LEVELS
        if not levels or levels == [None]:
            levels = LEVELS
        for li, level in enumerate(levels):
            if level is None:
                level = LEVELS[li]
            if fixed_rep is not None:
                rep, struct, kind = fixed_rep, fixed_struct, edge_name
            else:
                edge_seed = args.seed ^ sum(
                    (j + 1) * ord(ch)
                    for j, ch in enumerate(algo.name + edge_name))
                try:
                    rep, struct, kind = V.make_candidate(
                        algo, params, steps, big.CellRng(edge_seed + li * 104729), level)
                except Exception as exc:
                    candidate_failures[(algo.name, "explicit_edge", level,
                                        type(exc).__name__)] += 1
                    continue
            add_scenario(input_id, algo, params, steps, rep, struct, kind, level,
                         "explicit_edge", edge_name)

    # Exact arithmetic boundary cases that the corpus's shared edge helper
    # does not currently emit: very large coefficients, small rationals, and
    # a valid all-zero arithmetic sequence.
    arithmetic = next(a for a in ALL150 if a.name == "arithmetic_series")
    arithmetic_edges = (
        ("large_coefficients", {"a": F(99991), "d": F(99989), "n": 13}),
        ("small_fractional_coefficients", {"a": F(1, 1000), "d": F(-1, 3), "n": 4}),
        ("zero_coefficients", {"a": F(0), "d": F(0), "n": 1}),
    )
    for edge_name, params in arithmetic_edges:
        steps = arithmetic._steps(params)
        record_input_params(arithmetic, params, "explicit_edge")
        input_id = input_count
        input_count += 1
        for li, level in enumerate(LEVELS):
            rep, struct, kind = V.make_candidate(
                arithmetic, params, steps,
                big.CellRng(args.seed ^ (li * 65537) ^ sum(ord(c) for c in edge_name)),
                level)
            V.check(arithmetic, params, steps, rep, struct)
            add_scenario(input_id, arithmetic, params, steps, rep, struct, kind,
                         level, "explicit_edge", edge_name)

    if case_count == 0:
        raise SystemExit("no usable scenarios generated")
    algo_by_name = {algo.name: algo for algo in ALL150}
    first_case_by_input = {}
    for case_i, input_id in enumerate(case_columns["input_id"]):
        first_case_by_input.setdefault(input_id, case_i)
    boundary_by_input = {}
    for input_id, case_i in first_case_by_input.items():
        source = case_columns["input_source"][case_i]
        if source == "explicit_edge":
            boundary_by_input[input_id] = "explicit_edge"
            continue
        algo_name = case_columns["algorithm"][case_i]
        params = json.loads(case_columns["params_json"][case_i])
        at_min = at_max = False
        for key in algo_by_name[algo_name].structure_keys:
            if key not in params:
                continue
            stats = sample_param_stats.get((algo_name, key, "algorithm_builder"))
            if not stats:
                continue
            text_value = params[key]
            value = F(text_value) if "/" in text_value else F(int(text_value))
            at_min |= value == stats["minimum"]
            at_max |= value == stats["maximum"]
        if at_min and at_max:
            boundary_by_input[input_id] = "observed_min_and_max"
        elif at_min:
            boundary_by_input[input_id] = "observed_sample_min"
        elif at_max:
            boundary_by_input[input_id] = "observed_sample_max"
        else:
            boundary_by_input[input_id] = "interior"
    for case_i, input_id in enumerate(case_columns["input_id"]):
        case_columns["sampled_coefficient_boundary"][case_i] = boundary_by_input[input_id]

    n_step = len(step_truth)
    x_overall = np.frombuffer(overall_bytes, dtype=np.uint8).reshape(case_count, V.FEAT_DIM)
    x_step = np.frombuffer(step_bytes, dtype=np.uint8).reshape(n_step, V.FEAT_DIM)
    y_overall = np.asarray([LEVELS.index(float(v)) for v in case_columns["target_level"]],
                          dtype=np.int8)
    y_step = np.asarray(step_truth, dtype=np.int8)
    step_case_array = np.asarray(step_to_case, dtype=np.int32)

    # Load each current I6 expert with its serialized architecture.
    import bench_gen1 as B
    mu, sd = big.read_musd(args.musd)
    experts = {}
    for family, filename in big.FAMILY_MODEL_MAP.items():
        path = os.path.join(args.models, filename)
        if not os.path.isfile(path):
            raise SystemExit("missing expert model: " + path)
        h1, h2, _ = big.model_banner(path)
        B.H1G, B.H2G = h1, h2
        experts[family] = load_gen1(path)

    def infer(model, features):
        pred = np.empty(len(features), dtype=np.int8)
        conf = np.empty(len(features), dtype=np.float32)
        verdict = np.empty(len(features), dtype=np.float64)
        lv = np.asarray(LEVELS, dtype=np.float64)
        for start in range(0, len(features), 20000):
            end = min(len(features), start + 20000)
            normalized = big.normalize_i6(features[start:end], mu, sd)
            probs = big.softmax(fwd_batch(model, normalized))
            pred[start:end] = np.argmax(probs, axis=1)
            conf[start:end] = np.max(probs, axis=1)
            verdict[start:end] = np.clip(probs @ lv,
                                         -1.0 + big.VERDICT_EPS,
                                         1.0 - big.VERDICT_EPS)
        return pred, conf, verdict

    predictions = {}
    summaries = {}
    for family, model in experts.items():
        po, co, vo = infer(model, x_overall)
        ps, cs, vs = infer(model, x_step)
        predictions[family] = (po, co, vo, ps, cs, vs)
        summaries[family] = {
            "overall_n": int(len(y_overall)),
            "overall_correct": int((po == y_overall).sum()),
            "overall_accuracy_pct": float(100 * (po == y_overall).mean()),
            "overall_MAE": float(np.mean(np.abs(vo - np.asarray(LEVELS)[y_overall]))),
            "overall_MSE": float(np.mean((vo - np.asarray(LEVELS)[y_overall]) ** 2)),
            "step_n": int(len(y_step)),
            "step_correct": int((ps == y_step).sum()),
            "step_accuracy_pct": float(100 * (ps == y_step).mean()),
            "step_MAE": float(np.mean(np.abs(vs - np.asarray(LEVELS)[y_step]))),
            "step_MSE": float(np.mean((vs - np.asarray(LEVELS)[y_step]) ** 2)),
        }

    case_axis_values = {name: case_columns[name] for name in DIMENSIONS}
    case_axis_values["row_class"] = [LVLNAME[LEVELS[int(i)]] for i in y_overall]
    case_axes_encoded = {name: encode_axis(values)
                         for name, values in case_axis_values.items()}
    step_axes_encoded = dict(case_axes_encoded)
    step_axes_encoded["step_position"] = encode_axis(step_positions)
    step_axes_encoded["step_status"] = encode_axis(step_status)
    step_axis_classes = [LVLNAME[LEVELS[int(i)]] for i in y_step]
    step_axes_encoded["step_class"] = encode_axis(step_axis_classes)
    step_axes_encoded["row_class"] = step_axes_encoded["step_class"]

    condition_rows = []
    condition_gates_pass = True
    perfect_conditions = True
    for family, (po, _, vo, ps, _, vs) in predictions.items():
        rows, gates = group_rows(family, "overall",
                                 tuple(case_axis_values), case_axes_encoded,
                                 po, y_overall, vo)
        condition_rows.extend(rows)
        condition_gates_pass &= gates
        perfect_conditions &= all(row[6] == 100.0 for row in rows)
        rows, gates = group_rows(family, "local-step",
                                 tuple(step_axes_encoded), step_axes_encoded,
                                 ps, y_step, vs,
                                 row_indices=step_case_array)
        condition_rows.extend(rows)
        condition_gates_pass &= gates
        perfect_conditions &= all(row[6] == 100.0 for row in rows)

    outdir = args.outdir
    accuracy_path = os.path.join(outdir, "condition_accuracy.csv")
    with open(accuracy_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("expert", "row_type", "group_by", "condition", "n",
                    "correct", "accuracy_pct", "continuous_MAE", "continuous_MSE",
                    "1e-3_gate"))
        w.writerows(condition_rows)

    # Per-class confusion matrices and class recall/precision/F1.
    class_path = os.path.join(outdir, "class_accuracy.csv")
    with open(class_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("expert", "row_type", "true_class", "n", "correct",
                    "recall_pct", "precision_pct", "f1_pct"))
        for family, (po, _, _, ps, _, _) in predictions.items():
            for row_type, pred, truth in (("overall", po, y_overall),
                                          ("local-step", ps, y_step)):
                for cls, level in enumerate(LEVELS):
                    tp = int(((pred == cls) & (truth == cls)).sum())
                    support = int((truth == cls).sum())
                    predicted = int((pred == cls).sum())
                    recall = tp / support if support else 0.0
                    precision = tp / predicted if predicted else 0.0
                    f1 = 2 * recall * precision / (recall + precision) if recall + precision else 0.0
                    w.writerow((family, row_type, LVLNAME[level], support, tp,
                                100 * recall, 100 * precision, 100 * f1))

    # Per-parameter-vector accuracy: each input is evaluated against all
    # realizable verdict labels, so this reports exact per-input robustness.
    input_id_values = np.asarray(case_columns["input_id"], dtype=np.int64)
    unique_inputs, case_input_group = np.unique(input_id_values, return_inverse=True)
    input_overall_n = np.bincount(case_input_group)
    step_input_group = case_input_group[step_case_array]
    input_step_n = np.bincount(step_input_group, minlength=len(unique_inputs))
    input_case_index = np.full(len(unique_inputs), len(case_input_group), dtype=np.int64)
    np.minimum.at(input_case_index, case_input_group,
                  np.arange(len(case_input_group), dtype=np.int64))
    input_correct_counts = {}
    for family, (po, _, _, ps, _, _) in predictions.items():
        input_correct_counts[family] = (
            np.bincount(case_input_group,
                        weights=(po == y_overall).astype(np.int64),
                        minlength=len(unique_inputs)).astype(np.int64),
            np.bincount(step_input_group,
                        weights=(ps == y_step).astype(np.int64),
                        minlength=len(unique_inputs)).astype(np.int64))
    labels_by_input = defaultdict(set)
    for case_i, group_i in enumerate(case_input_group):
        labels_by_input[int(group_i)].add(case_columns["target_level"][case_i])
    input_accuracy_path = os.path.join(outdir, "input_accuracy.csv.gz")
    with gzip.open(input_accuracy_path, "wt", newline="", encoding="utf-8") as f:
        fieldnames = ["input_id", "algorithm", "input_source",
                      "sampled_coefficient_boundary", "base_params_json", "params_json",
                      "verdict_labels_tested", "overall_cases", "local_step_rows"]
        for family in sorted(predictions):
            fieldnames += [family + "_overall_correct", family + "_overall_accuracy_pct",
                           family + "_step_correct", family + "_step_accuracy_pct"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for group_i, input_id in enumerate(unique_inputs):
            ci = int(input_case_index[group_i])
            row = {
                "input_id": int(input_id),
                "algorithm": case_columns["algorithm"][ci],
                "input_source": case_columns["input_source"][ci],
                "sampled_coefficient_boundary":
                    case_columns["sampled_coefficient_boundary"][ci],
                "base_params_json": case_columns["base_params_json"][ci],
                "params_json": case_columns["params_json"][ci],
                "verdict_labels_tested": ";".join(sorted(labels_by_input[group_i])),
                "overall_cases": int(input_overall_n[group_i]),
                "local_step_rows": int(input_step_n[group_i]),
            }
            for family in predictions:
                overall_counts, step_counts = input_correct_counts[family]
                oc = int(overall_counts[group_i])
                sc = int(step_counts[group_i])
                row[family + "_overall_correct"] = oc
                row[family + "_overall_accuracy_pct"] = (
                    100.0 * oc / int(input_overall_n[group_i]))
                row[family + "_step_correct"] = sc
                row[family + "_step_accuracy_pct"] = (
                    100.0 * sc / int(input_step_n[group_i])
                    if input_step_n[group_i] else 0.0)
            w.writerow(row)

    # Exact sampled coefficient/input ranges for each algorithm.
    param_path = os.path.join(outdir, "parameter_coverage.csv")
    with open(param_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("algorithm", "parameter", "source", "n", "unique_values", "minimum",
                    "maximum", "zero_n", "negative_n", "positive_n", "fractional_n"))
        for (algo, key, source), st in sorted(sample_param_stats.items()):
            w.writerow((algo, key, source, st["n"], len(st["unique"]),
                        value_text(st["minimum"]), value_text(st["maximum"]),
                        st["zero"], st["negative"], st["positive"], st["fractional"]))

    strata_path = os.path.join(outdir, "strata_coverage.csv")
    with open(strata_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(("algorithm", "input_source", "target_level",
                    "overall_cases", "local_step_rows"))
        counts = Counter((case_columns["algorithm"][i],
                          case_columns["input_source"][i],
                          case_columns["target_level"][i])
                         for i in range(case_count))
        local_counts = Counter((case_columns["algorithm"][int(ci)],
                                case_columns["input_source"][int(ci)],
                                case_columns["target_level"][int(ci)])
                               for ci in step_case_array)
        count_map = {key: n for key, n in counts.items()}
        for algo in ALL150:
            for source in INPUT_SOURCES:
                for level in LEVELS:
                    matching = [key for key in count_map
                                if key[0] == algo.name and key[1] == source
                                and key[2] == f"{level:+.1f}"]
                    if not matching:
                        w.writerow((algo.name, source, f"{level:+.1f}", 0, 0))
                    for key in matching:
                        w.writerow((*key, count_map[key], local_counts[key]))

    # Full scenario and step prediction manifests, compressed for size.
    case_path = os.path.join(outdir, "scenario_cases.csv.gz")
    with gzip.open(case_path, "wt", newline="", encoding="utf-8") as f:
        fieldnames = list(CASE_COLUMNS)
        for family in sorted(predictions):
            fieldnames += [family + "_predicted", family + "_confidence",
                           family + "_correct"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for i in range(case_count):
            row = {key: case_columns[key][i] for key in CASE_COLUMNS}
            for family, (po, co, _, _, _, _) in predictions.items():
                row[family + "_predicted"] = LVLNAME[LEVELS[int(po[i])]]
                row[family + "_confidence"] = "%.9g" % float(co[i])
                row[family + "_correct"] = int(po[i] == y_overall[i])
            w.writerow(row)

    step_path = os.path.join(outdir, "step_cases.csv.gz")
    with gzip.open(step_path, "wt", newline="", encoding="utf-8") as f:
        fieldnames = ["case_id", "input_id", "algorithm", "family", "target_level", "step_index",
                      "step_position", "step_status", "step_class", "expected_value",
                      "reported_value", "candidate_type", "input_source"]
        for family in sorted(predictions):
            fieldnames += [family + "_predicted", family + "_confidence",
                           family + "_correct"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for si, case_id in enumerate(step_to_case):
            ci = int(case_id)
            row = {
                "case_id": ci,
                "input_id": case_columns["input_id"][ci],
                "algorithm": case_columns["algorithm"][ci],
                "family": case_columns["family"][ci],
                "target_level": case_columns["target_level"][ci],
                "step_index": step_indices[si],
                "step_position": step_positions[si],
                "step_status": step_status[si],
                "step_class": LVLNAME[LEVELS[int(y_step[si])]],
                "expected_value": step_values[si],
                "reported_value": report_values[si],
                "candidate_type": case_columns["candidate_type"][ci],
                "input_source": case_columns["input_source"][ci],
            }
            for family, (_, _, _, ps, cs, _) in predictions.items():
                row[family + "_predicted"] = LVLNAME[LEVELS[int(ps[si])]]
                row[family + "_confidence"] = "%.9g" % float(cs[si])
                row[family + "_correct"] = int(ps[si] == y_step[si])
            w.writerow(row)

    summary_path = os.path.join(outdir, "summary.txt")
    overall_gate = all(m["overall_MAE"] < 1e-3 and m["overall_MSE"] < 1e-3
                       for m in summaries.values())
    step_gate = all(m["step_MAE"] < 1e-3 and m["step_MSE"] < 1e-3
                    for m in summaries.values())
    perfect_aggregate = all(m["overall_accuracy_pct"] == 100.0
                            and m["step_accuracy_pct"] == 100.0
                            for m in summaries.values())
    edge_input_ids = {case_columns["input_id"][i] for i in range(case_count)
                      if case_columns["edge_case"][i] != "randomized"}
    with open(summary_path, "w") as f:
        f.write("I6 expanded crossed stress suite\n")
        f.write("seed=%d algorithms=%d valid_builder_inputs_per_algorithm=%d\n" %
                (args.seed, len(ALL150), draws))
        f.write("random_parameter_sets=%d explicit_edge_input_sets=%d\n" %
                (sum(build_coverage.values()), len(edge_input_ids)))
        f.write("design=each valid parameter set crossed with all five verdict labels\n")
        f.write("overall_cases=%d local_step_rows=%d explicit_edge_cases=%d\n" %
                (case_count, n_step,
                 sum(1 for i in range(case_count)
                     if case_columns["edge_case"][i] != "randomized")))
        f.write("continuous_gate_overall=%s local_step=%s all_condition_slices=%s\n" %
                ("PASS" if overall_gate else "FAIL",
                 "PASS" if step_gate else "FAIL",
                 "PASS" if condition_gates_pass else "FAIL"))
        f.write("perfect_accuracy_overall_and_step=%s all_condition_slices=%s\n" %
                ("PASS" if perfect_aggregate else "FAIL",
                 "PASS" if perfect_conditions else "FAIL"))
        f.write("condition_rows=%d\n" % len(condition_rows))
        f.write("build_failures=%d rejected_builder_trials=%d "
                "candidate_failures=%d "
                "unrealizable_labels_dropped=%d\n" %
                (sum(build_failures.values()), sum(rejected_build_trials.values()),
                 sum(candidate_failures.values()),
                 sum(dropped_levels.values())))
        f.write("\nPer-expert totals (accuracy percentages are exact class agreement):\n")
        for family, metrics in sorted(summaries.items()):
            f.write("%s overall=%d/%d (%.6f%%) MAE=%.9g MSE=%.9g; "
                    "local_step=%d/%d (%.6f%%) MAE=%.9g MSE=%.9g\n" %
                    (family, metrics["overall_correct"], metrics["overall_n"],
                     metrics["overall_accuracy_pct"], metrics["overall_MAE"],
                     metrics["overall_MSE"], metrics["step_correct"],
                     metrics["step_n"], metrics["step_accuracy_pct"],
                     metrics["step_MAE"], metrics["step_MSE"]))
        f.write("\nSuccessfully generated builder inputs by algorithm:\n")
        for key, value in sorted(build_coverage.items()):
            f.write("%s inputs=%d\n" % (key, value))
        f.write("\nRejected builder/candidate trials by algorithm:\n")
        for key, value in sorted(rejected_build_trials.items()):
            f.write("%s rejected=%d\n" % (key, value))
        f.write("\nUnrealizable target requests by algorithm/input source/level:\n")
        for key, value in sorted(dropped_levels.items()):
            f.write("%s source=%s level=%+.1f dropped=%d\n" % (*key, value))
        f.write("\nCandidate failures by algorithm/source/level/error:\n")
        for key, value in sorted(candidate_failures.items()):
            f.write("%s source=%s level=%+.1f error=%s failed=%d\n" % (*key, value))
        f.write("\nFiles: condition_accuracy.csv, class_accuracy.csv, "
                "input_accuracy.csv.gz, parameter_coverage.csv, strata_coverage.csv, "
                "scenario_cases.csv.gz, step_cases.csv.gz\n")

    print("summary -> %s" % summary_path, flush=True)
    print("condition metrics -> %s (%d rows)" % (accuracy_path, len(condition_rows)),
          flush=True)
    for family, metrics in sorted(summaries.items()):
        print("%s overall %.6f%% (%d/%d), step %.6f%% (%d/%d), "
              "MAE/MSE %.3g/%.3g overall and %.3g/%.3g step" %
              (family, metrics["overall_accuracy_pct"], metrics["overall_correct"],
               metrics["overall_n"], metrics["step_accuracy_pct"],
               metrics["step_correct"], metrics["step_n"], metrics["overall_MAE"],
               metrics["overall_MSE"], metrics["step_MAE"], metrics["step_MSE"]),
              flush=True)
    print("gates: overall=%s step=%s every_condition=%s perfect_accuracy=%s" %
          (overall_gate, step_gate, condition_gates_pass,
           perfect_aggregate and perfect_conditions), flush=True)
    if args.strict_targets and not (overall_gate and step_gate and condition_gates_pass):
        raise SystemExit(3)
    if args.require_perfect_accuracy and not (perfect_aggregate and perfect_conditions):
        raise SystemExit(4)


if __name__ == "__main__":
    main()
