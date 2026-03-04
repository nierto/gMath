#!/usr/bin/env python3
"""
Generate ULP reference data for all 17 transcendental functions.

Uses mpmath at 200+ decimal places to produce exact Q-format reference values
as Rust const arrays. Three output files, one per compute path:

  tests/data/ulp_refs_q64_64.rs    — (i128, i128, &str) tuples
  tests/data/ulp_refs_q128_128.rs  — ([u64; 4], [u64; 4], &str) tuples
  tests/data/ulp_refs_q256_256.rs  — ([u64; 8], [u64; 8], &str) tuples

Encoding: Q_N.N fixed-point as two's complement integer = round(value * 2^N).
Input values are also encoded as Q_N.N.
"""

import os
import sys
from mpmath import mp, mpf, pi, e, sqrt, log, exp, sin, cos, tan, atan, atan2
from mpmath import asin, acos, sinh, cosh, tanh, asinh, acosh, atanh, power, nint

mp.dps = 250  # 250 decimal places, well above 77 needed for Q256.256

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "tests", "data")


# ═══════════════════════════════════════════════════════════════════
# Test points for each function
# ═══════════════════════════════════════════════════════════════════

# Constants at full precision
PI = pi
E = e
SQRT2 = sqrt(2)
SQRT3 = sqrt(3)
LN2 = log(2)

# For functions taking a single argument
EXP_POINTS = [
    ("0", mpf(0)),
    ("0.5", mpf("0.5")),
    ("1.0", mpf(1)),
    ("2.0", mpf(2)),
    ("-1.0", mpf(-1)),
    ("-0.5", mpf("-0.5")),
    ("0.001", mpf("0.001")),
    ("ln2", LN2),
    ("3.0", mpf(3)),
    ("10.0", mpf(10)),
    ("-10.0", mpf(-10)),
    ("0.999", mpf("0.999")),
]

LN_POINTS = [
    ("0.001", mpf("0.001")),
    ("0.1", mpf("0.1")),
    ("0.5", mpf("0.5")),
    ("1.0", mpf(1)),
    ("1.001", mpf("1.001")),
    ("2.0", mpf(2)),
    ("e", E),
    ("3.0", mpf(3)),
    ("10.0", mpf(10)),
    ("100.0", mpf(100)),
    ("1024.0", mpf(1024)),
]

SQRT_POINTS = [
    ("0.25", mpf("0.25")),
    ("0.5", mpf("0.5")),
    ("1.0", mpf(1)),
    ("2.0", mpf(2)),
    ("3.0", mpf(3)),
    ("4.0", mpf(4)),
    ("5.0", mpf(5)),
    ("10.0", mpf(10)),
    ("100.0", mpf(100)),
    ("0.01", mpf("0.01")),
    ("1.0001", mpf("1.0001")),
]

SIN_POINTS = [
    ("0.5", mpf("0.5")),
    ("1.0", mpf(1)),
    ("-1.0", mpf(-1)),
    ("0.001", mpf("0.001")),
    ("pi/6", PI / 6),
    ("pi/4", PI / 4),
    ("pi/3", PI / 3),
    ("pi/2", PI / 2),
    ("pi", PI),
    ("3.0", mpf(3)),
    ("10.0", mpf(10)),
]

COS_POINTS = [
    ("0.5", mpf("0.5")),
    ("1.0", mpf(1)),
    ("-1.0", mpf(-1)),
    ("0.001", mpf("0.001")),
    ("pi/6", PI / 6),
    ("pi/4", PI / 4),
    ("pi/3", PI / 3),
    ("pi/2", PI / 2),
    ("pi", PI),
    ("3.0", mpf(3)),
    ("10.0", mpf(10)),
]

TAN_POINTS = [
    ("0.5", mpf("0.5")),
    ("1.0", mpf(1)),
    ("-1.0", mpf(-1)),
    ("0.001", mpf("0.001")),
    ("pi/4", PI / 4),
    ("pi/6", PI / 6),
    ("pi/3", PI / 3),
    ("1.5", mpf("1.5")),
    ("-0.5", mpf("-0.5")),
    ("0.25", mpf("0.25")),
]

ATAN_POINTS = [
    ("0.5", mpf("0.5")),
    ("1.0", mpf(1)),
    ("-1.0", mpf(-1)),
    ("0.001", mpf("0.001")),
    ("2.0", mpf(2)),
    ("10.0", mpf(10)),
    ("100.0", mpf(100)),
    ("-10.0", mpf(-10)),
    ("tan_pi_8", tan(PI / 8)),
    ("0.1", mpf("0.1")),
]

# (label, y, x)
ATAN2_POINTS = [
    ("1_1", mpf(1), mpf(1)),
    ("1_0", mpf(1), mpf(0)),
    ("0_1", mpf(0), mpf(1)),
    ("-1_1", mpf(-1), mpf(1)),
    ("1_-1", mpf(1), mpf(-1)),
    ("0_-1", mpf(0), mpf(-1)),
    ("sqrt3_1", SQRT3, mpf(1)),
    ("1_sqrt3", mpf(1), SQRT3),
    ("-1_-1", mpf(-1), mpf(-1)),
    ("0.001_1", mpf("0.001"), mpf(1)),
]

ASIN_POINTS = [
    ("0.5", mpf("0.5")),
    ("-0.5", mpf("-0.5")),
    ("0.001", mpf("0.001")),
    ("0.1", mpf("0.1")),
    ("sqrt2/2", SQRT2 / 2),
    ("sqrt3/2", SQRT3 / 2),
    ("0.999", mpf("0.999")),
    ("0.9", mpf("0.9")),
    ("-0.9", mpf("-0.9")),
    ("0.25", mpf("0.25")),
]

ACOS_POINTS = [
    ("0.5", mpf("0.5")),
    ("-0.5", mpf("-0.5")),
    ("0.001", mpf("0.001")),
    ("0.1", mpf("0.1")),
    ("sqrt2/2", SQRT2 / 2),
    ("sqrt3/2", SQRT3 / 2),
    ("0.999", mpf("0.999")),
    ("0.9", mpf("0.9")),
    ("-0.9", mpf("-0.9")),
    ("0.25", mpf("0.25")),
]

SINH_POINTS = [
    ("0.5", mpf("0.5")),
    ("1.0", mpf(1)),
    ("2.0", mpf(2)),
    ("-1.0", mpf(-1)),
    ("0.001", mpf("0.001")),
    ("3.0", mpf(3)),
    ("5.0", mpf(5)),
    ("-3.0", mpf(-3)),
    ("ln2", LN2),
    ("0.1", mpf("0.1")),
]

COSH_POINTS = [
    ("0.5", mpf("0.5")),
    ("1.0", mpf(1)),
    ("2.0", mpf(2)),
    ("-1.0", mpf(-1)),
    ("0.001", mpf("0.001")),
    ("3.0", mpf(3)),
    ("5.0", mpf(5)),
    ("-3.0", mpf(-3)),
    ("ln2", LN2),
    ("0.1", mpf("0.1")),
]

TANH_POINTS = [
    ("0.5", mpf("0.5")),
    ("1.0", mpf(1)),
    ("2.0", mpf(2)),
    ("-1.0", mpf(-1)),
    ("-2.0", mpf(-2)),
    ("0.001", mpf("0.001")),
    ("3.0", mpf(3)),
    ("5.0", mpf(5)),
    ("-0.5", mpf("-0.5")),
    ("0.1", mpf("0.1")),
]

ASINH_POINTS = [
    ("0.5", mpf("0.5")),
    ("1.0", mpf(1)),
    ("2.0", mpf(2)),
    ("-1.0", mpf(-1)),
    ("0.001", mpf("0.001")),
    ("10.0", mpf(10)),
    ("-10.0", mpf(-10)),
    ("0.1", mpf("0.1")),
    ("100.0", mpf(100)),
    ("0.25", mpf("0.25")),
]

ACOSH_POINTS = [
    ("1.001", mpf("1.001")),
    ("1.5", mpf("1.5")),
    ("2.0", mpf(2)),
    ("3.0", mpf(3)),
    ("5.0", mpf(5)),
    ("10.0", mpf(10)),
    ("100.0", mpf(100)),
    ("1.0001", mpf("1.0001")),
    ("e", E),
    ("1.1", mpf("1.1")),
]

ATANH_POINTS = [
    ("0.5", mpf("0.5")),
    ("-0.5", mpf("-0.5")),
    ("0.9", mpf("0.9")),
    ("-0.9", mpf("-0.9")),
    ("0.001", mpf("0.001")),
    ("0.1", mpf("0.1")),
    ("-0.1", mpf("-0.1")),
    ("0.99", mpf("0.99")),
    ("0.25", mpf("0.25")),
    ("0.75", mpf("0.75")),
]

# (label, base, exponent)
POW_POINTS = [
    ("2^10", mpf(2), mpf(10)),
    ("2^0.5", mpf(2), mpf("0.5")),
    ("10^3", mpf(10), mpf(3)),
    ("2.0^3.5", mpf("2.0"), mpf("3.5")),
    ("1.5^2.5", mpf("1.5"), mpf("2.5")),
    ("0.5^2", mpf("0.5"), mpf(2)),
    ("0.5^3", mpf("0.5"), mpf(3)),
    ("3^0.5", mpf(3), mpf("0.5")),
    ("1.001^100", mpf("1.001"), mpf(100)),
    ("e^1", E, mpf(1)),
]


# ═══════════════════════════════════════════════════════════════════
# Q-format conversion utilities
# ═══════════════════════════════════════════════════════════════════

def to_qformat(value, frac_bits):
    """Convert mpf value to Q-format integer: round(value * 2^frac_bits)."""
    scaled = value * mpf(2) ** frac_bits
    return int(nint(scaled))


def to_twos_complement(val, total_bits):
    """Convert signed Python int to two's complement unsigned integer."""
    if val >= 0:
        return val
    return val + (1 << total_bits)


def to_u64_words(val, total_bits):
    """Convert two's complement value to little-endian u64 words."""
    tc = to_twos_complement(val, total_bits)
    n_words = total_bits // 64
    words = []
    for i in range(n_words):
        words.append((tc >> (64 * i)) & 0xFFFFFFFFFFFFFFFF)
    return words


def format_i128(val):
    """Format a Q64.64 value as Rust i128 literal."""
    if val >= 0:
        return f"0x{val:032X}_i128"
    tc = to_twos_complement(val, 128)
    # Use wrapping cast from u128
    return f"0x{tc:032X}_u128 as i128"


def format_u64_array(words, n):
    """Format as Rust [u64; N] array literal."""
    parts = [f"0x{w:016X}" for w in words]
    return f"[{', '.join(parts)}]"


# ═══════════════════════════════════════════════════════════════════
# Code generation for each profile
# ═══════════════════════════════════════════════════════════════════

def generate_single_arg_refs(func_name, mp_func, points, frac_bits, total_bits):
    """Generate reference data for a single-argument function."""
    entries = []
    for label, input_val in points:
        try:
            input_q = to_qformat(input_val, frac_bits)
            result_val = mp_func(input_val)
            expected_q = to_qformat(result_val, frac_bits)
            entries.append((label, input_q, expected_q))
        except Exception as ex:
            print(f"  WARNING: {func_name}({label}) failed: {ex}", file=sys.stderr)
    return entries


def generate_two_arg_refs(func_name, mp_func, points, frac_bits, total_bits):
    """Generate reference data for a two-argument function (atan2, pow)."""
    entries = []
    for item in points:
        label = item[0]
        arg1 = item[1]  # y for atan2, base for pow
        arg2 = item[2]  # x for atan2, exp for pow
        try:
            input1_q = to_qformat(arg1, frac_bits)
            input2_q = to_qformat(arg2, frac_bits)
            result_val = mp_func(arg1, arg2)
            expected_q = to_qformat(result_val, frac_bits)
            entries.append((label, input1_q, input2_q, expected_q))
        except Exception as ex:
            print(f"  WARNING: {func_name}({label}) failed: {ex}", file=sys.stderr)
    return entries


def write_q64_64(f, func_name, entries, two_arg=False):
    """Write Q64.64 reference data as (i128, i128, &str) or (i128, i128, i128, &str)."""
    name = func_name.upper() + "_REFS"
    if two_arg:
        f.write(f"const {name}: &[(i128, i128, i128, &str)] = &[\n")
        for label, in1, in2, exp in entries:
            f.write(f"    ({format_i128(in1)}, {format_i128(in2)}, {format_i128(exp)}, \"{label}\"),\n")
    else:
        f.write(f"const {name}: &[(i128, i128, &str)] = &[\n")
        for label, inp, exp in entries:
            f.write(f"    ({format_i128(inp)}, {format_i128(exp)}, \"{label}\"),\n")
    f.write("];\n\n")


def write_wide(f, func_name, entries, n_words, two_arg=False):
    """Write Q128.128 or Q256.256 reference data as ([u64;N], ..., &str)."""
    name = func_name.upper() + "_REFS"
    total_bits = n_words * 64
    if two_arg:
        f.write(f"const {name}: &[([u64; {n_words}], [u64; {n_words}], [u64; {n_words}], &str)] = &[\n")
        for label, in1, in2, exp in entries:
            w1 = format_u64_array(to_u64_words(in1, total_bits), n_words)
            w2 = format_u64_array(to_u64_words(in2, total_bits), n_words)
            we = format_u64_array(to_u64_words(exp, total_bits), n_words)
            f.write(f"    ({w1}, {w2}, {we}, \"{label}\"),\n")
    else:
        f.write(f"const {name}: &[([u64; {n_words}], [u64; {n_words}], &str)] = &[\n")
        for label, inp, exp in entries:
            wi = format_u64_array(to_u64_words(inp, total_bits), n_words)
            we = format_u64_array(to_u64_words(exp, total_bits), n_words)
            f.write(f"    ({wi}, {we}, \"{label}\"),\n")
    f.write("];\n\n")


# ═══════════════════════════════════════════════════════════════════
# Main generation
# ═══════════════════════════════════════════════════════════════════

SINGLE_ARG_FUNCS = [
    ("exp",   exp,   EXP_POINTS),
    ("ln",    log,   LN_POINTS),
    ("sqrt",  sqrt,  SQRT_POINTS),
    ("sin",   sin,   SIN_POINTS),
    ("cos",   cos,   COS_POINTS),
    ("tan",   tan,   TAN_POINTS),
    ("atan",  atan,  ATAN_POINTS),
    ("asin",  asin,  ASIN_POINTS),
    ("acos",  acos,  ACOS_POINTS),
    ("sinh",  sinh,  SINH_POINTS),
    ("cosh",  cosh,  COSH_POINTS),
    ("tanh",  tanh,  TANH_POINTS),
    ("asinh", asinh, ASINH_POINTS),
    ("acosh", acosh, ACOSH_POINTS),
    ("atanh", atanh, ATANH_POINTS),
]

TWO_ARG_FUNCS = [
    ("atan2", atan2, ATAN2_POINTS),
    ("pow",   power, POW_POINTS),
]

PROFILES = [
    # (name, frac_bits, total_bits, n_words)
    ("q64_64",   64,  128, 2),
    ("q128_128", 128, 256, 4),
    ("q256_256", 256, 512, 8),
]


def generate_profile(profile_name, frac_bits, total_bits, n_words):
    """Generate one reference file."""
    filename = os.path.join(DATA_DIR, f"ulp_refs_{profile_name}.rs")
    print(f"Generating {filename} (Q{frac_bits}.{frac_bits}, {total_bits}-bit)...")

    with open(filename, "w") as f:
        f.write(f"// AUTO-GENERATED by scripts/generate_ulp_references.py\n")
        f.write(f"// Q{frac_bits}.{frac_bits} ({total_bits}-bit two's complement)\n")
        f.write(f"// mpmath precision: {mp.dps} decimal places\n")
        f.write(f"//\n")
        f.write(f"// Format: (input_qformat, expected_output_qformat, label)\n")
        f.write(f"// All values are exact Q{frac_bits}.{frac_bits} representations:\n")
        f.write(f"//   value_q = round(value_real * 2^{frac_bits})\n\n")

        # Single-arg functions
        for func_name, mp_func, points in SINGLE_ARG_FUNCS:
            print(f"  {func_name}: {len(points)} points...")
            entries = generate_single_arg_refs(func_name, mp_func, points, frac_bits, total_bits)

            if profile_name == "q64_64":
                write_q64_64(f, func_name, entries)
            else:
                write_wide(f, func_name, entries, n_words)

        # Two-arg functions
        for func_name, mp_func, points in TWO_ARG_FUNCS:
            print(f"  {func_name}: {len(points)} points...")
            entries = generate_two_arg_refs(func_name, mp_func, points, frac_bits, total_bits)

            if profile_name == "q64_64":
                write_q64_64(f, func_name, entries, two_arg=True)
            else:
                write_wide(f, func_name, entries, n_words, two_arg=True)

    print(f"  Done: {filename}")


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    for profile_name, frac_bits, total_bits, n_words in PROFILES:
        generate_profile(profile_name, frac_bits, total_bits, n_words)

    total_points = sum(len(p) for _, _, p in SINGLE_ARG_FUNCS) + sum(len(p) for _, _, p in TWO_ARG_FUNCS)
    print(f"\nTotal: {total_points} test points × {len(PROFILES)} profiles = {total_points * len(PROFILES)} validated comparisons")


if __name__ == "__main__":
    main()
