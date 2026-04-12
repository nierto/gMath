#!/usr/bin/env python3
"""
Generate mpmath-verified reference data for DECIMAL transcendentals.

Unlike binary transcendentals which store values in Q-format (value × 2^N),
decimal transcendentals store values in scaled-integer form (value × 10^DP).

Output format: Rust const arrays with inputs and expected outputs at the
profile's compute-tier decimal places (DECIMAL_COMPUTE_DP):

    Q16.16 (realtime):   compute_dp =   9  → fits in i64
    Q32.32 (compact):    compute_dp =  19  → fits in i128
    Q64.64 (embedded):   compute_dp =  38  → I256 (4 × u64 words)
    Q128.128 (balanced): compute_dp =  77  → I512 (8 × u64 words)
    Q256.256 (scientific): compute_dp = 154 → I1024 (16 × u64 words)

Reference inputs mirror the binary FASC generator:
  - exp: [-20, 20]
  - ln:  (0.0001, 10000]
  - sqrt: [0.0001, 10000]
  - sin/cos: [-10, 10] (multiples of π emphasized)
  - tan: [-1.5, 1.5]
  - atan: [-100, 100] (log density near 0)
  - atan2: quadrant-spanning pairs
  - asin/acos: [-0.999, 0.999]
  - sinh/cosh/tanh: [-5, 5]
  - asinh: [-100, 100]
  - acosh: [1.001, 100]
  - atanh: (-0.999, 0.999)

Usage:
    python3 scripts/generate_decimal_transcendental_refs.py
"""

import os
import random
import math
from mpmath import mp, mpf, pi, e, sqrt, log, exp, sin, cos, tan, atan, atan2
from mpmath import asin, acos, sinh, cosh, tanh, asinh, acosh, atanh, power, nint

# Precision WAY above any profile's compute dp (154)
mp.dps = 300

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "tests", "data")

random.seed(42)

# ═══════════════════════════════════════════════════════════════════
# Scaled integer conversion
# ═══════════════════════════════════════════════════════════════════

def to_scaled_int(value, dp):
    """Convert mpf value to integer scaled by 10^dp, rounded."""
    scale = mpf(10) ** dp
    scaled = value * scale
    return int(nint(scaled))


def to_twos_complement(val, total_bits):
    """Convert signed int to two's complement unsigned."""
    if val >= 0:
        return val
    return val + (1 << total_bits)


def to_u64_words(val, total_bits):
    """Convert signed int to little-endian u64 words."""
    tc = to_twos_complement(val, total_bits)
    n_words = total_bits // 64
    return [(tc >> (64 * i)) & 0xFFFFFFFFFFFFFFFF for i in range(n_words)]


def format_i64(val):
    return f"{val}i64"


def format_i128(val):
    if val >= 0:
        return f"0x{val:032X}_i128"
    tc = to_twos_complement(val, 128)
    return f"0x{tc:032X}_u128 as i128"


def format_u64_array(words):
    parts = [f"0x{w:016X}" for w in words]
    return f"[{', '.join(parts)}]"


def format_expected(val, total_bits):
    if total_bits == 64:
        return format_i64(val)
    if total_bits == 128:
        return format_i128(val)
    words = to_u64_words(val, total_bits)
    return format_u64_array(words)


def type_signature(total_bits):
    if total_bits == 64:
        return "i64"
    if total_bits == 128:
        return "i128"
    n_words = total_bits // 64
    return f"[u64; {n_words}]"


# ═══════════════════════════════════════════════════════════════════
# Point generators — mirror binary FASC generator style
# ═══════════════════════════════════════════════════════════════════

def edge_points(lo, hi, n_edge=30):
    edges = [lo, hi]
    for delta in [1e-8, 1e-6, 1e-4, 1e-2, 0.1, 0.5]:
        if lo + delta < hi:
            edges.append(lo + delta)
        if hi - delta > lo:
            edges.append(hi - delta)
    if lo < 0 < hi:
        for v in [0, 1e-10, -1e-10, 1e-6, -1e-6, 1e-3, -1e-3]:
            if lo <= v <= hi:
                edges.append(v)
    for v in [1, -1, 2, -2, 0.5, -0.5]:
        if lo <= v <= hi:
            edges.append(v)
    edges = sorted(set(edges))
    return edges[:n_edge]


def uniform_grid(lo, hi, n):
    if n <= 1:
        return [(lo + hi) / 2]
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def log_spaced(lo, hi, n):
    if lo <= 0:
        lo = 1e-10
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    return [math.exp(log_lo + (log_hi - log_lo) * i / (n - 1)) for i in range(n)]


def log_density_near_zero(lo, hi, n):
    points = []
    if hi > 0:
        pos = log_spaced(max(1e-6, lo if lo > 0 else 1e-6), hi, n // 2)
        points.extend(pos)
    if lo < 0:
        neg = [-x for x in log_spaced(max(1e-6, -hi if hi < 0 else 1e-6), -lo, n // 2)]
        points.extend(neg)
    for v in [0, 1e-10, -1e-10, 1e-6, -1e-6]:
        if lo <= v <= hi:
            points.append(v)
    points = sorted(set(points))
    return points[:n]


# ═══════════════════════════════════════════════════════════════════
# Per-function point sets
# ═══════════════════════════════════════════════════════════════════

def gen_exp_points(n=300):
    edges = edge_points(-20, 20)
    # Highlights near zero (exp is sensitive to ULP near 1)
    for v in [0, 1e-15, -1e-15, 1e-10, -1e-10, 0.1, -0.1, 0.5, -0.5, 1, -1, 2, -2, 5, -5, 10, -10, 15, -15]:
        edges.append(v)
    grid = uniform_grid(-20, 20, n - len(edges))
    pts = sorted(set(edges + grid))[:n]
    return [(mpf(v), f"exp_{i}") for i, v in enumerate(pts)]


def gen_ln_points(n=300):
    edges = [0.0001, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0,
             1.001, 1.01, 1.1, 2.0, float(e), 10.0, 100.0, 1000.0, 10000.0]
    # Near 1 (challenging for ln)
    for delta in [1e-10, 1e-6, 1e-4, 1e-2]:
        edges.append(1.0 + delta)
        edges.append(1.0 - delta)
    grid = log_spaced(0.0001, 10000, n - len(edges))
    pts = sorted(set(edges + grid))[:n]
    return [(mpf(v), f"ln_{i}") for i, v in enumerate(pts)]


def gen_sqrt_points(n=300):
    edges = [0.0, 1e-10, 1e-6, 0.0001, 0.001, 0.01, 0.25, 0.5, 1.0, 2.0,
             3.0, 4.0, 9.0, 10.0, 16.0, 25.0, 100.0, 1000.0, 10000.0]
    grid = log_spaced(0.0001, 10000, n - len(edges))
    pts = sorted(set(edges + grid))[:n]
    return [(mpf(v), f"sqrt_{i}") for i, v in enumerate(pts)]


def gen_sin_points(n=300):
    edges = edge_points(-10, 10)
    pi_f = float(pi)
    # Multiples of π/6, π/4, π/3, π/2, π, 2π, 3π
    for k in [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]:
        edges.append(k * pi_f / 6)
        edges.append(k * pi_f / 4)
        edges.append(k * pi_f / 3)
        edges.append(k * pi_f / 2)
        edges.append(k * pi_f)
    # Near zero
    for v in [1e-15, 1e-10, 1e-5, -1e-15, -1e-10, -1e-5]:
        edges.append(v)
    edges = [v for v in edges if -10 <= v <= 10]
    grid = uniform_grid(-10, 10, n - len(edges))
    pts = sorted(set(edges + grid))[:n]
    return [(mpf(v), f"sin_{i}") for i, v in enumerate(pts)]


def gen_cos_points(n=300):
    # Same domain as sin
    edges = edge_points(-10, 10)
    pi_f = float(pi)
    for k in [-6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6]:
        edges.append(k * pi_f / 6)
        edges.append(k * pi_f / 4)
        edges.append(k * pi_f / 3)
        edges.append(k * pi_f / 2)
        edges.append(k * pi_f)
    for v in [1e-15, 1e-10, 1e-5, -1e-15, -1e-10, -1e-5]:
        edges.append(v)
    edges = [v for v in edges if -10 <= v <= 10]
    grid = uniform_grid(-10, 10, n - len(edges))
    pts = sorted(set(edges + grid))[:n]
    return [(mpf(v), f"cos_{i}") for i, v in enumerate(pts)]


def gen_atan_points(n=300):
    edges = [-100, -50, -10, -5, -2, -1, -0.5, -0.1, -0.01, -0.001, 0,
             0.001, 0.01, 0.1, 0.5, 1, 2, 5, 10, 50, 100]
    # Near ±1 is the boundary case (atan(1) = π/4 exactly)
    for delta in [1e-10, 1e-6, 1e-4, 1e-2]:
        edges.append(1.0 + delta)
        edges.append(1.0 - delta)
        edges.append(-1.0 + delta)
        edges.append(-1.0 - delta)
    grid = log_density_near_zero(-100, 100, n - len(edges))
    pts = sorted(set(edges + grid))[:n]
    return [(mpf(v), f"atan_{i}") for i, v in enumerate(pts)]


def gen_atan2_points(n=200):
    result = []
    # Standard quadrant points
    edge_pairs = [
        (1, 0), (0, 1), (-1, 0), (0, -1),  # axes
        (1, 1), (-1, 1), (-1, -1), (1, -1),  # diagonals
        (0.001, 1), (1, 0.001), (-0.001, 1), (1, -0.001),  # near axes
        (3, 4), (4, 3), (-3, 4), (3, -4),  # Pythagorean
        (0.5, 0.5), (10, 10), (100, 1), (1, 100),
    ]
    for i, (y, x) in enumerate(edge_pairs):
        result.append((mpf(y), mpf(x), f"atan2_{i}"))
    for i in range(n - len(result)):
        angle = random.uniform(-math.pi, math.pi)
        radius = math.exp(random.uniform(-3, 5))
        y = radius * math.sin(angle)
        x = radius * math.cos(angle)
        result.append((mpf(y), mpf(x), f"atan2_{len(result)}"))
    return result[:n]


def gen_asin_points(n=300):
    edges = [-0.999, -0.99, -0.9, -0.5, 0, 0.5, 0.9, 0.99, 0.999,
             -0.001, 0.001, -0.1, 0.1]
    for _ in range(30):
        edges.append(1 - 10 ** random.uniform(-6, -1))
        edges.append(-1 + 10 ** random.uniform(-6, -1))
    grid = uniform_grid(-0.999, 0.999, n - len(edges))
    pts = sorted(set([v for v in edges + grid if -0.9999 <= v <= 0.9999]))[:n]
    return [(mpf(v), f"asin_{i}") for i, v in enumerate(pts)]


def gen_acos_points(n=300):
    edges = [-0.999, -0.99, -0.9, -0.5, 0, 0.5, 0.9, 0.99, 0.999,
             -0.001, 0.001, -0.1, 0.1]
    for _ in range(30):
        edges.append(1 - 10 ** random.uniform(-6, -1))
        edges.append(-1 + 10 ** random.uniform(-6, -1))
    grid = uniform_grid(-0.999, 0.999, n - len(edges))
    pts = sorted(set([v for v in edges + grid if -0.9999 <= v <= 0.9999]))[:n]
    return [(mpf(v), f"acos_{i}") for i, v in enumerate(pts)]


def gen_tan_points(n=300):
    lo, hi = -1.5, 1.5
    edges = edge_points(lo, hi)
    # Near π/4 ≈ 0.785
    pi_f = float(pi)
    edges.append(pi_f / 4)
    edges.append(-pi_f / 4)
    edges.append(pi_f / 6)
    edges.append(-pi_f / 6)
    grid = uniform_grid(lo, hi, n - len(edges))
    pts = sorted(set(edges + grid))[:n]
    return [(mpf(v), f"tan_{i}") for i, v in enumerate(pts)]


def gen_sinh_points(n=200):
    edges = edge_points(-5, 5)
    grid = uniform_grid(-5, 5, n - len(edges))
    pts = sorted(set(edges + grid))[:n]
    return [(mpf(v), f"sinh_{i}") for i, v in enumerate(pts)]


def gen_cosh_points(n=200):
    return [(v, label.replace("sinh", "cosh")) for v, label in gen_sinh_points(n)]


def gen_tanh_points(n=200):
    return [(v, label.replace("sinh", "tanh")) for v, label in gen_sinh_points(n)]


# Financial-specific stress points
def gen_financial_points():
    """Points that stress financial formulas (compound interest, rates, etc.)."""
    points = []
    # Compound interest: (1 + r/n)^(nt) where r is small
    # ln version: nt × ln(1 + r/n)
    # Common rates: 0.01, 0.025, 0.05, 0.1, 0.15
    for rate in [0.001, 0.005, 0.01, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20]:
        points.append((mpf(1 + rate), f"fin_ln_1plus{rate}"))
        points.append((mpf(1 + rate/12), f"fin_ln_1plus{rate}over12"))
        points.append((mpf(1 + rate/365), f"fin_ln_1plus{rate}over365"))
    return points


# ═══════════════════════════════════════════════════════════════════
# Reference writing
# ═══════════════════════════════════════════════════════════════════

def write_unary_refs(f, const_name, points, mp_func, compute_dp, total_bits):
    """Write: (input_scaled, expected_scaled, label)"""
    type_sig = type_signature(total_bits)
    max_val = (1 << (total_bits - 1)) - 1
    min_val = -(1 << (total_bits - 1))

    f.write(f"#[allow(dead_code)]\n")
    f.write(f"pub const {const_name}: &[({type_sig}, {type_sig}, &str)] = &[\n")
    count = 0
    skipped = 0
    for item in points:
        value, label = item
        try:
            input_scaled = to_scaled_int(value, compute_dp)
            if input_scaled > max_val or input_scaled < min_val:
                skipped += 1
                continue
            result = mp_func(value)
            expected = to_scaled_int(result, compute_dp)
            if expected > max_val or expected < min_val:
                skipped += 1
                continue
            in_fmt = format_expected(input_scaled, total_bits)
            out_fmt = format_expected(expected, total_bits)
            f.write(f"    ({in_fmt}, {out_fmt}, \"{label}\"),\n")
            count += 1
        except Exception as ex:
            print(f"  WARNING: {const_name} {label}: {ex}")
            skipped += 1
    f.write("];\n\n")
    return count, skipped


def write_binary_refs(f, const_name, points, mp_func, compute_dp, total_bits):
    """Write: (a_scaled, b_scaled, expected_scaled, label)"""
    type_sig = type_signature(total_bits)
    max_val = (1 << (total_bits - 1)) - 1
    min_val = -(1 << (total_bits - 1))

    f.write(f"#[allow(dead_code)]\n")
    f.write(f"pub const {const_name}: &[({type_sig}, {type_sig}, {type_sig}, &str)] = &[\n")
    count = 0
    skipped = 0
    for item in points:
        a_val, b_val, label = item
        try:
            a_scaled = to_scaled_int(a_val, compute_dp)
            b_scaled = to_scaled_int(b_val, compute_dp)
            if not (min_val <= a_scaled <= max_val and min_val <= b_scaled <= max_val):
                skipped += 1
                continue
            result = mp_func(a_val, b_val)
            expected = to_scaled_int(result, compute_dp)
            if expected > max_val or expected < min_val:
                skipped += 1
                continue
            a_fmt = format_expected(a_scaled, total_bits)
            b_fmt = format_expected(b_scaled, total_bits)
            out_fmt = format_expected(expected, total_bits)
            f.write(f"    ({a_fmt}, {b_fmt}, {out_fmt}, \"{label}\"),\n")
            count += 1
        except Exception as ex:
            print(f"  WARNING: {const_name} {label}: {ex}")
            skipped += 1
    f.write("];\n\n")
    return count, skipped


# (profile_name, compute_dp, total_bits)
PROFILES = [
    ("q16_16",     9,    64),
    ("q32_32",    19,   128),
    ("q64_64",    38,   256),
    ("q128_128",  77,   512),
    ("q256_256", 154,  1024),
]


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"mpmath precision: {mp.dps} decimal places")

    # Generate shared point sets (domain-bounded, profile-independent)
    print("Generating test points...")
    exp_pts = gen_exp_points(300)
    ln_pts = gen_ln_points(300) + gen_financial_points()
    sqrt_pts = gen_sqrt_points(300)
    sin_pts = gen_sin_points(300)
    cos_pts = gen_cos_points(300)
    tan_pts = gen_tan_points(250)
    atan_pts = gen_atan_points(300)
    atan2_pts = gen_atan2_points(200)
    asin_pts = gen_asin_points(250)
    acos_pts = gen_acos_points(250)
    sinh_pts = gen_sinh_points(200)
    cosh_pts = gen_cosh_points(200)
    tanh_pts = gen_tanh_points(200)

    unary_funcs = [
        ("DECIMAL_EXP_REFS",   exp_pts,   exp),
        ("DECIMAL_LN_REFS",    ln_pts,    log),
        ("DECIMAL_SQRT_REFS",  sqrt_pts,  sqrt),
        ("DECIMAL_SIN_REFS",   sin_pts,   sin),
        ("DECIMAL_COS_REFS",   cos_pts,   cos),
        ("DECIMAL_TAN_REFS",   tan_pts,   tan),
        ("DECIMAL_ATAN_REFS",  atan_pts,  atan),
        ("DECIMAL_ASIN_REFS",  asin_pts,  asin),
        ("DECIMAL_ACOS_REFS",  acos_pts,  acos),
        ("DECIMAL_SINH_REFS",  sinh_pts,  sinh),
        ("DECIMAL_COSH_REFS",  cosh_pts,  cosh),
        ("DECIMAL_TANH_REFS",  tanh_pts,  tanh),
    ]

    binary_funcs = [
        ("DECIMAL_ATAN2_REFS", atan2_pts, atan2),
    ]

    grand_total = 0
    for profile_name, compute_dp, total_bits in PROFILES:
        filename = os.path.join(DATA_DIR, f"decimal_refs_{profile_name}.rs")
        print(f"\n{'='*60}")
        print(f"Generating {filename}  (compute_dp={compute_dp}, bits={total_bits})")
        print(f"{'='*60}")

        total_count = 0
        total_skipped = 0

        with open(filename, "w") as f:
            f.write(f"// AUTO-GENERATED by scripts/generate_decimal_transcendental_refs.py\n")
            f.write(f"// Profile: {profile_name}\n")
            f.write(f"// Compute DP: {compute_dp}  (values scaled by 10^{compute_dp})\n")
            f.write(f"// Storage type: {type_signature(total_bits)} ({total_bits} bits)\n")
            f.write(f"// mpmath precision: {mp.dps} decimal places\n")
            f.write(f"//\n")
            f.write(f"// Unary format: (input_scaled, expected_scaled, label)\n")
            f.write(f"// Binary format: (a_scaled, b_scaled, expected_scaled, label)\n\n")

            for const_name, points, mp_func in unary_funcs:
                print(f"  {const_name}: {len(points)} points...", end="")
                cnt, skip = write_unary_refs(f, const_name, points, mp_func, compute_dp, total_bits)
                print(f" -> {cnt} written, {skip} skipped")
                total_count += cnt
                total_skipped += skip

            for const_name, points, mp_func in binary_funcs:
                print(f"  {const_name}: {len(points)} points...", end="")
                cnt, skip = write_binary_refs(f, const_name, points, mp_func, compute_dp, total_bits)
                print(f" -> {cnt} written, {skip} skipped")
                total_count += cnt
                total_skipped += skip

        print(f"  Done: {total_count} refs, {total_skipped} skipped")
        grand_total += total_count

    print(f"\nAll 5 profiles: {grand_total} total references")


if __name__ == "__main__":
    main()
