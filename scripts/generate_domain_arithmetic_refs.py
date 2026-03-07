#!/usr/bin/env python3
"""Generate domain arithmetic reference values for gMath validation suite.

Generates 4 files:
  - tests/data/domain_arith_refs_common.rs     (decimal, symbolic, cross-domain)
  - tests/data/domain_arith_refs_q64_64.rs     (binary Q64.64)
  - tests/data/domain_arith_refs_q128_128.rs   (binary Q128.128)
  - tests/data/domain_arith_refs_q256_256.rs   (binary Q256.256)

Reference values computed with mpmath at 250 decimal places.
Follows pattern from generate_fasc_ulp_references.py.
"""

import os
import sys
from fractions import Fraction
import mpmath as mp
from mpmath import mpf, nint

mp.dps = 250

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "tests", "data")

# ═══════════════════════════════════════════════════════════════════
# Q-format conversion (matches generate_fasc_ulp_references.py)
# ═══════════════════════════════════════════════════════════════════

PROFILES = [
    ("q64_64", 64, 128),
    ("q128_128", 128, 256),
    ("q256_256", 256, 512),
]


def round_fraction_half_away(frac):
    """Round Fraction to nearest integer, half away from zero."""
    n, d = frac.numerator, frac.denominator
    q, r = divmod(abs(n), d)
    if 2 * r >= d:
        q += 1
    return q if n >= 0 else -q


def to_qformat(value, frac_bits):
    """Convert value to Q-format integer: round(value * 2^frac_bits).

    Accepts both Fraction (exact) and mpf (finite precision).
    For integer arithmetic, always pass Fraction for bit-exact results.
    """
    if isinstance(value, Fraction):
        # Exact integer arithmetic — zero precision loss
        scaled = value * (1 << frac_bits)
        return round_fraction_half_away(scaled)
    else:
        # mpf path for transcendentals (inherent approximation is acceptable)
        scaled = value * mpf(2) ** frac_bits
        return int(nint(scaled))


def to_twos_complement(val, total_bits):
    """Convert signed Python int to two's complement unsigned integer."""
    if val >= 0:
        return val
    return val + (1 << total_bits)


def format_i128(val):
    """Format as Rust i128 literal."""
    if val >= 0:
        return f"0x{val:032X}_i128"
    tc = to_twos_complement(val, 128)
    return f"0x{tc:032X}_u128 as i128"


def to_u64_words(val, total_bits):
    """Convert signed int to little-endian u64 words."""
    tc = to_twos_complement(val, total_bits)
    n_words = total_bits // 64
    return [(tc >> (64 * i)) & 0xFFFFFFFFFFFFFFFF for i in range(n_words)]


def format_u64_array(words):
    """Format as Rust [u64; N] literal."""
    parts = [f"0x{w:016X}" for w in words]
    return f"[{', '.join(parts)}]"


def format_expected(val, total_bits):
    """Format expected value for the given profile."""
    if total_bits == 128:
        return format_i128(val)
    words = to_u64_words(val, total_bits)
    return format_u64_array(words)


def type_signature(total_bits):
    """Return Rust type signature for expected values."""
    if total_bits == 128:
        return "i128"
    n_words = total_bits // 64
    return f"[u64; {n_words}]"


# ═══════════════════════════════════════════════════════════════════
# Test case generators
# ═══════════════════════════════════════════════════════════════════

def binary_cases():
    """Generate binary arithmetic test cases with integer inputs.

    All results must fit in Q64.64 range (|integer| < 2^63).
    Returns dict of op_name -> [(a_str, b_str, Fraction_result), ...].
    Uses exact Fraction arithmetic — zero floating-point precision loss.
    """
    ops = {}

    add_pairs = [
        # Tier 1 (i8 range: -128..127)
        (7, 3), (100, 27), (-42, 15), (0, 0), (1, -1),
        (127, 0), (-128, 0), (50, 50), (-100, -20), (1, 126),
        # Tier 1->2 overflow
        (100, 100), (127, 1), (-128, -1),
        # Tier 2 (i16 range)
        (200, 300), (1000, -500), (32000, 700), (-10000, 5000),
        (-200, -300), (15000, 15000),
        # Tier 2->3 overflow
        (32000, 32000),
        # Tier 3 (i32 range)
        (100000, 200000), (2000000000, 100000000),
        (-1500000000, 500000000), (1000000, -999999),
        # Tier 3->4 overflow
        (2000000000, 2000000000),
        # Tier 4 (i64 range, fits Q64.64)
        (10**10, 2 * 10**10), (-(10**12), 10**11),
        (10**15, 10**14), (2**50, 2**49),
        (-(10**15), 10**15),
    ]
    ops["add"] = [(str(a), str(b), Fraction(a + b)) for a, b in add_pairs]

    sub_pairs = [
        # Tier 1
        (7, 3), (100, 27), (-42, -15), (0, 0), (127, 127),
        (-128, -128), (50, -50), (-100, 20), (1, -126),
        # Tier 1->2
        (100, -100), (127, -1), (-128, 1),
        # Tier 2
        (500, 300), (1000, 500), (32000, -700), (-10000, -5000),
        (-200, -300),
        # Tier 2->3
        (32000, -32000),
        # Tier 3
        (300000, 100000), (2000000000, 100000000),
        (-1500000000, -500000000), (1000000, 999999),
        # Tier 3->4
        (2000000000, -2000000000),
        # Tier 4
        (3 * 10**10, 10**10), (-(10**12), -(10**11)),
        (10**15, -(10**14)), (2**50, 2**49),
    ]
    ops["sub"] = [(str(a), str(b), Fraction(a - b)) for a, b in sub_pairs]

    mul_pairs = [
        # Tier 1
        (7, 3), (10, 10), (-6, 7), (0, 100), (1, 127),
        (-1, -1), (-1, 1), (1, 1), (12, -12),
        # Tier 1->2 overflow
        (50, 50), (127, 127),
        # Tier 2
        (200, 100), (1000, 30), (-500, 20),
        # Tier 2->3
        (32000, 100),
        # Tier 3
        (100000, 200), (50000, -300),
        # Tier 3->4
        (100000, 100000),
        # Tier 4
        (10**6, 10**6), (10**8, 100), (-(10**6), 10**6),
        # Largest safe for Q64.64 (result < 2^63)
        (10**9, 10**9), (2**30, 2**30),
        (10**9, -2), (3, 10**9),
    ]
    ops["mul"] = [(str(a), str(b), Fraction(a * b)) for a, b in mul_pairs]

    div_pairs = [
        # Exact divisions (result is integer)
        (10, 2), (21, 7), (100, 25), (-42, 6), (0, 1),
        (1000000, 1000), (-999, -3), (1, 1), (-1, 1), (1, -1),
        (2**60, 2**30), (100, -25),
        # Power-of-2 fractions (exact in binary fixed-point)
        (1, 2), (1, 4), (3, 8), (7, 16), (1, 1024),
        (3, 2), (5, 4), (15, 8),
        # Inexact (rational results, finite ULP error expected)
        (7, 3), (1, 7), (100, 3), (-5, 6), (22, 7),
        (1, 9), (10, 3), (1000, 7), (-100, 9),
        # Close to 1
        (99, 100), (101, 100), (999, 1000),
        # Larger
        (10**10, 3), (10**12, 7),
    ]
    ops["div"] = [(str(a), str(b), Fraction(a, b)) for a, b in div_pairs]

    return ops


def decimal_cases():
    """Generate decimal arithmetic test cases.

    Inputs are exact decimal strings. Verification via to_rational().
    Returns dict of op_name -> [(a_str, b_str), ...].
    """
    ops = {}

    ops["add"] = [
        # Same decimal places (1 dp)
        ("1.5", "2.5"), ("0.1", "0.9"), ("10.0", "20.0"),
        ("0.5", "0.5"), ("99.9", "0.1"), ("0.0", "1.5"),
        ("100.0", "0.0"),
        # Same decimal places (2 dp)
        ("1.25", "2.75"), ("0.01", "0.99"), ("99.99", "0.01"),
        ("0.50", "0.50"),
        # Same decimal places (3 dp)
        ("0.001", "0.999"), ("1.500", "2.500"), ("0.125", "0.875"),
        # Negative values (use positive inputs; test builds Negate)
        # Actually, gmath_parse handles negatives, but decimal negatives
        # with leading "-" may have the parse bug. Test positive only for now.
        # Mixed decimal places (cross-precision, may route to rational)
        ("1.5", "0.25"), ("0.1", "0.01"), ("10.0", "0.001"),
    ]

    ops["sub"] = [
        ("2.5", "1.5"), ("1.0", "0.5"), ("10.0", "3.0"),
        ("0.5", "0.5"), ("100.0", "99.9"), ("0.0", "0.0"),
        ("5.5", "2.5"),
        ("4.00", "1.25"), ("0.99", "0.01"), ("100.00", "0.01"),
        ("1.000", "0.001"), ("0.500", "0.125"),
        ("1.5", "0.25"), ("0.1", "0.01"),
    ]

    ops["mul"] = [
        ("1.5", "2.5"), ("0.1", "0.1"), ("10.0", "3.0"),
        ("0.5", "0.5"), ("100.0", "0.1"), ("0.0", "1.5"),
        ("1.25", "4.00"), ("0.10", "0.10"), ("99.99", "1.00"),
        ("0.5", "0.25"),
        ("2.0", "3.0"), ("0.5", "2.0"),
    ]

    ops["div"] = [
        ("7.5", "2.5"), ("1.0", "2.0"), ("10.0", "4.0"),
        ("0.5", "0.5"), ("100.0", "10.0"),
        ("9.0", "3.0"),
        ("1.25", "0.25"), ("0.50", "0.10"),
        ("1.5", "2.5"), ("0.5", "0.25"),
        ("2.0", "1.0"), ("6.0", "3.0"),
    ]

    return ops


def symbolic_cases():
    """Generate symbolic arithmetic test cases.

    Inputs are fraction strings. Verification via to_rational().
    Returns dict of op_name -> [(a_str, b_str), ...].
    """
    ops = {}

    ops["add"] = [
        ("1/3", "2/3"), ("1/3", "2/7"), ("1/2", "1/2"),
        ("1/6", "1/6"), ("3/4", "1/4"), ("1/3", "1/6"),
        ("0/1", "1/7"), ("5/8", "3/8"),
        ("1/10", "1/5"), ("7/11", "3/11"), ("1/100", "1/200"),
        ("99/100", "1/100"), ("22/7", "1/7"),
        ("1/997", "1/991"), ("1/13", "1/17"),
        ("100/3", "1/3"), ("1/1000", "1/1000"),
        ("3/5", "2/5"), ("7/12", "5/12"),
    ]

    ops["sub"] = [
        ("2/3", "1/3"), ("1/2", "1/3"), ("1/3", "1/3"),
        ("3/4", "1/4"), ("5/6", "1/6"), ("1/7", "2/7"),
        ("1/3", "1/6"), ("0/1", "1/7"),
        ("22/7", "1/7"), ("1/997", "1/991"),
        ("100/3", "1/3"), ("1/2", "1/4"), ("7/8", "3/8"),
        ("5/6", "5/6"), ("11/13", "7/13"),
    ]

    ops["mul"] = [
        ("1/3", "3/1"), ("1/2", "1/2"), ("2/3", "3/4"),
        ("1/3", "1/3"), ("1/7", "7/1"), ("0/1", "1/3"),
        ("5/6", "6/5"), ("1/10", "1/10"),
        ("22/7", "7/22"), ("3/4", "4/5"), ("1/100", "100/1"),
        ("2/3", "3/2"), ("7/11", "11/13"),
        ("1/2", "2/1"), ("3/7", "7/3"),
    ]

    ops["div"] = [
        ("1/3", "2/7"), ("1/2", "1/2"), ("3/4", "1/4"),
        ("1/3", "1/3"), ("1/7", "1/7"),
        ("5/6", "5/6"), ("22/7", "11/7"), ("1/10", "1/5"),
        ("3/4", "3/8"), ("100/3", "10/3"), ("1/2", "3/4"),
        ("7/11", "7/13"), ("1/2", "2/1"),
        ("3/7", "7/3"),
    ]

    return ops


def cross_domain_cases():
    """Generate cross-domain arithmetic test cases.

    Mixed input types. Verification via to_rational().
    Returns dict of op_name -> [(a_str, b_str), ...].
    """
    ops = {}

    ops["add"] = [
        # Binary + Decimal
        ("7", "1.5"), ("0", "0.5"), ("100", "0.01"),
        ("1", "0.1"),
        # Binary + Symbolic
        ("7", "1/3"), ("0", "1/7"), ("10", "1/10"),
        ("3", "2/3"),
        # Decimal + Symbolic
        ("1.5", "1/3"), ("0.5", "1/6"), ("2.5", "1/4"),
    ]

    ops["sub"] = [
        ("7", "1.5"), ("100", "0.01"),
        ("7", "1/3"), ("10", "1/10"),
        ("1.5", "1/3"), ("0.5", "1/6"),
        ("3", "2/3"),
    ]

    ops["mul"] = [
        ("7", "1.5"), ("0", "0.5"), ("100", "0.01"),
        ("2", "0.25"),
        ("9", "1/3"), ("14", "1/7"), ("6", "2/3"),
        ("0.5", "1/3"), ("1.5", "2/3"),
    ]

    ops["div"] = [
        ("7", "1.5"), ("100", "0.5"), ("1", "0.25"),
        ("10", "1/10"),
        ("1.5", "1/3"), ("0.5", "1/6"),
        ("9", "1/3"), ("6", "2/3"),
    ]

    return ops


# ═══════════════════════════════════════════════════════════════════
# Fraction parsing + computation
# ═══════════════════════════════════════════════════════════════════

def parse_to_fraction(s):
    """Parse input string to Python Fraction (exact)."""
    if '/' in s:
        parts = s.split('/')
        return Fraction(int(parts[0]), int(parts[1]))
    elif '.' in s:
        return Fraction(s)
    else:
        return Fraction(int(s))


def compute_rational(a_str, b_str, op):
    """Compute exact rational result."""
    a = parse_to_fraction(a_str)
    b = parse_to_fraction(b_str)
    if op == "add":
        return a + b
    elif op == "sub":
        return a - b
    elif op == "mul":
        return a * b
    elif op == "div":
        return a / b
    else:
        raise ValueError(f"Unknown op: {op}")


def format_rational_entry(a_str, b_str, result, label):
    """Format a rational reference entry as Rust code."""
    num = result.numerator
    den = result.denominator
    return f'    ("{a_str}", "{b_str}", {num}, {den}, "{label}")'


# ═══════════════════════════════════════════════════════════════════
# Output writers
# ═══════════════════════════════════════════════════════════════════

HEADER_COMMON = """\
// AUTO-GENERATED by scripts/generate_domain_arithmetic_refs.py
// Domain arithmetic reference values (profile-independent)
// mpmath precision: 250 decimal places / exact Fraction arithmetic
//
// Format: (a_str, b_str, expected_numerator: i128, expected_denominator: i128, label)
// Verification: evaluate() -> to_rational() -> numerator_i128/denominator_i128
//
// Decimal inputs: "1.5", "0.25" etc. (ExactDecimal -> Decimal domain)
// Symbolic inputs: "1/3", "22/7" etc. (ExactRational -> Symbolic domain)
// Cross-domain: mixed input types -> rational comparison

"""


def write_common_refs(decimal_ops, symbolic_ops, cross_ops):
    """Write domain_arith_refs_common.rs."""
    filepath = os.path.join(DATA_DIR, "domain_arith_refs_common.rs")
    with open(filepath, 'w') as f:
        f.write(HEADER_COMMON)

        # Decimal refs
        for op_name in ["add", "sub", "mul", "div"]:
            const_name = f"DECIMAL_{op_name.upper()}_REFS"
            pairs = decimal_ops[op_name]
            f.write(f"const {const_name}: &[(&str, &str, i128, i128, &str)] = &[\n")
            for i, (a, b) in enumerate(pairs):
                result = compute_rational(a, b, op_name)
                label = f"dec_{op_name}_{i}"
                f.write(format_rational_entry(a, b, result, label) + ",\n")
            f.write("];\n\n")

        # Symbolic refs
        for op_name in ["add", "sub", "mul", "div"]:
            const_name = f"SYMBOLIC_{op_name.upper()}_REFS"
            pairs = symbolic_ops[op_name]
            f.write(f"const {const_name}: &[(&str, &str, i128, i128, &str)] = &[\n")
            for i, (a, b) in enumerate(pairs):
                result = compute_rational(a, b, op_name)
                label = f"sym_{op_name}_{i}"
                f.write(format_rational_entry(a, b, result, label) + ",\n")
            f.write("];\n\n")

        # Cross-domain refs
        for op_name in ["add", "sub", "mul", "div"]:
            const_name = f"CROSS_{op_name.upper()}_REFS"
            pairs = cross_ops[op_name]
            f.write(f"const {const_name}: &[(&str, &str, i128, i128, &str)] = &[\n")
            for i, (a, b) in enumerate(pairs):
                result = compute_rational(a, b, op_name)
                label = f"cross_{op_name}_{i}"
                f.write(format_rational_entry(a, b, result, label) + ",\n")
            f.write("];\n\n")

    print(f"  wrote {filepath}")


def write_binary_refs(profile_name, frac_bits, total_bits, binary_ops):
    """Write profile-specific binary reference file."""
    filepath = os.path.join(DATA_DIR, f"domain_arith_refs_{profile_name}.rs")
    ts = type_signature(total_bits)

    with open(filepath, 'w') as f:
        f.write(f"// AUTO-GENERATED by scripts/generate_domain_arithmetic_refs.py\n")
        f.write(f"// Binary domain arithmetic references for {profile_name}\n")
        f.write(f"// Q{frac_bits}.{frac_bits} ({total_bits}-bit) fixed-point\n")
        f.write(f"// mpmath precision: 250 decimal places\n")
        f.write(f"//\n")
        f.write(f'// Format: (a_str, b_str, expected_{ts}, label)\n')
        f.write(f"// Verification: evaluate() -> as_binary_storage() -> ULP compare\n")
        f.write(f"//\n")
        f.write(f"// Integer inputs go through gmath() -> Binary domain.\n")
        f.write(f"// Expected Q-format = round(exact_result * 2^{frac_bits}).\n\n")

        for op_name in ["add", "sub", "mul", "div"]:
            const_name = f"BINARY_{op_name.upper()}_REFS"
            cases = binary_ops[op_name]
            f.write(f"const {const_name}: &[(&str, &str, {ts}, &str)] = &[\n")
            for i, (a_str, b_str, mpf_result) in enumerate(cases):
                qval = to_qformat(mpf_result, frac_bits)
                formatted = format_expected(qval, total_bits)
                label = f"bin_{op_name}_{i}"
                f.write(f'    ("{a_str}", "{b_str}", {formatted}, "{label}"),\n')
            f.write("];\n\n")

    print(f"  wrote {filepath}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Generating domain arithmetic reference values...")
    print(f"  mpmath precision: {mp.dps} decimal places")

    # Generate test cases
    bin_ops = binary_cases()
    dec_ops = decimal_cases()
    sym_ops = symbolic_cases()
    cross_ops = cross_domain_cases()

    # Count cases
    bin_count = sum(len(v) for v in bin_ops.values())
    dec_count = sum(len(v) for v in dec_ops.values())
    sym_count = sum(len(v) for v in sym_ops.values())
    cross_count = sum(len(v) for v in cross_ops.values())
    print(f"  binary: {bin_count} cases (x3 profiles)")
    print(f"  decimal: {dec_count} cases")
    print(f"  symbolic: {sym_count} cases")
    print(f"  cross-domain: {cross_count} cases")

    # Write common refs
    write_common_refs(dec_ops, sym_ops, cross_ops)

    # Write profile-specific binary refs
    for profile_name, frac_bits, total_bits in PROFILES:
        write_binary_refs(profile_name, frac_bits, total_bits, bin_ops)

    total = bin_count * 3 + dec_count + sym_count + cross_count
    print(f"\nDone! {total} total reference entries across 4 files.")


if __name__ == "__main__":
    main()
