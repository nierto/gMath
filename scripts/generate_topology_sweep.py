#!/usr/bin/env python3
"""Generate topology sweep reference values for gMath fractal topology mapping.

Creates a systematic grid of test cases covering the full input space for each
arithmetic operation. Each mathematical value is tested through MULTIPLE domain
representations (integer string, decimal string, fraction string) so the Rust
sweep can determine which domain gives 0 ULP for each point.

This data serves dual purpose:
  1. Validation: 1000+ test cases per operation against mpmath 250-digit truth
  2. Topology:   Empirical domain mapping for fractal topology routing tables

Output:
  tests/data/topology_sweep_common.rs     (rational refs for all representations)
  tests/data/topology_sweep_q64_64.rs     (binary Q64.64 refs)
  tests/data/topology_sweep_q128_128.rs   (binary Q128.128 refs)
  tests/data/topology_sweep_q256_256.rs   (binary Q256.256 refs)

Deterministic precedence for ties (when multiple domains give 0 ULP):
  Symbolic > Decimal > Binary > Ternary
  (Encodes exactness preference, not speed — speed is non-deterministic)

Run: python3 scripts/generate_topology_sweep.py
Time: ~1-5 minutes (mpmath at 250 digits)
"""

import os
import sys
import math
import itertools
import random
from fractions import Fraction
import mpmath as mp
from mpmath import mpf, nint

mp.dps = 250

# Maximum pairs per operation per domain (controls output size)
# 10000 pairs × 4 ops × ~80 bytes/line ≈ 3.2MB per domain file
MAX_PAIRS_PER_OP = 10000
RNG_SEED = 42  # deterministic for reproducibility

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "tests", "data")

# Domain precedence (higher = preferred when tied at 0 ULP)
DOMAIN_SYMBOLIC = 3
DOMAIN_DECIMAL = 2
DOMAIN_BINARY = 1
DOMAIN_TERNARY = 0

PROFILES = [
    ("q64_64", 64, 128),
    ("q128_128", 128, 256),
    ("q256_256", 256, 512),
]

# Maximum integer magnitude per profile (before overflow)
PROFILE_MAX_INT = {
    "q64_64": 2**63 - 1,         # ~9.2e18
    "q128_128": 2**127 - 1,      # ~1.7e38
    "q256_256": 2**255 - 1,      # ~5.8e76
}


# ═══════════════════════════════════════════════════════════════════
# Q-format conversion (matches existing generate_domain_arithmetic_refs.py)
# ═══════════════════════════════════════════════════════════════════

def round_fraction_half_away(frac):
    """Round Fraction to nearest integer, half away from zero."""
    n, d = frac.numerator, frac.denominator
    q, r = divmod(abs(n), d)
    if 2 * r >= d:
        q += 1
    return q if n >= 0 else -q


def to_qformat(value, frac_bits):
    """Convert value to Q-format integer: round(value * 2^frac_bits)."""
    if isinstance(value, Fraction):
        scaled = value * (1 << frac_bits)
        return round_fraction_half_away(scaled)
    else:
        scaled = value * mpf(2) ** frac_bits
        return int(nint(scaled))


def to_twos_complement(val, total_bits):
    if val >= 0:
        return val
    return val + (1 << total_bits)


def format_i128(val):
    if val >= 0:
        return f"0x{val:032X}_i128"
    tc = to_twos_complement(val, 128)
    return f"0x{tc:032X}_u128 as i128"


def to_u64_words(val, total_bits):
    tc = to_twos_complement(val, total_bits)
    n_words = total_bits // 64
    return [(tc >> (64 * i)) & 0xFFFFFFFFFFFFFFFF for i in range(n_words)]


def format_u64_array(words):
    parts = [f"0x{w:016X}" for w in words]
    return f"[{', '.join(parts)}]"


def format_expected(val, total_bits):
    if total_bits == 128:
        return format_i128(val)
    return format_u64_array(to_u64_words(val, total_bits))


def type_signature(total_bits):
    if total_bits == 128:
        return "i128"
    return f"[u64; {total_bits // 64}]"


# ═══════════════════════════════════════════════════════════════════
# Grid value generators
# ═══════════════════════════════════════════════════════════════════

def integer_axis():
    """Generate integer axis values for binary domain testing.

    Covers: zero, small, tier boundaries, powers of 2/10/3, primes, large values.
    Returns list of (int_value, str_representation) tuples.
    """
    values = set()

    # Zero
    values.add(0)

    # Small integers (tier 1: i8 range)
    for v in [1, 2, 3, 5, 7, 10, 13, 42, 100, 127]:
        values.add(v)
        values.add(-v)

    # Tier boundaries
    for v in [128, 255, 256, 32767, 32768, 65535, 65536]:
        values.add(v)
        values.add(-v)

    # Powers of 2 (binary-natural)
    for exp in range(1, 61):
        v = 2**exp
        if v <= 2**62:  # stay safely below Q64.64 overflow
            values.add(v)

    # Powers of 10 (decimal-natural)
    for exp in range(1, 19):
        v = 10**exp
        values.add(v)

    # Powers of 3 (ternary-natural)
    for exp in range(1, 39):
        v = 3**exp
        if v <= 10**18:
            values.add(v)

    # Primes (symbolic-natural)
    for p in [7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61,
              67, 71, 73, 79, 83, 89, 97, 101, 127, 251, 509, 997, 1009]:
        values.add(p)
        values.add(-p)

    # Specific interesting values
    for v in [42, 255, 1000, 9999, 10001, 100000, 999999, 1000001]:
        values.add(v)
        values.add(-v)

    # Extended range: large powers of 2 near tier boundaries
    for v in [2**62, 2**63 - 1]:  # near i64::MAX
        values.add(v)
        values.add(-v)

    # Extended range: large powers of 10
    for v in [10**15, 10**18]:
        values.add(v)
        values.add(-v)

    # Extended range: near i32::MAX/MIN
    for v in [2**31 - 1, 2**31, 2**32 - 1, 2**32]:
        values.add(v)
        values.add(-v)

    # Sort for reproducibility
    return sorted(values)


def decimal_axis():
    """Generate decimal string axis values for decimal domain testing.

    Covers: various decimal precisions, exact and near-exact values.
    Returns list of (Fraction_value, str_representation) tuples.
    """
    entries = []
    seen = set()

    # 1 decimal place
    for v in ["0.0", "0.1", "0.2", "0.3", "0.4", "0.5", "0.6", "0.7", "0.8", "0.9",
              "1.0", "1.5", "2.0", "2.5", "3.0", "5.0", "7.5", "9.9",
              "10.0", "20.0", "25.0", "50.0", "99.9", "100.0"]:
        if v not in seen:
            entries.append((Fraction(v), v))
            seen.add(v)

    # 2 decimal places
    for v in ["0.01", "0.05", "0.10", "0.25", "0.33", "0.50", "0.75", "0.99",
              "1.00", "1.25", "1.50", "2.50", "3.14", "9.99", "10.00",
              "12.50", "25.00", "50.00", "99.99", "100.00"]:
        if v not in seen:
            entries.append((Fraction(v), v))
            seen.add(v)

    # 3 decimal places
    for v in ["0.001", "0.010", "0.100", "0.125", "0.250", "0.333", "0.500",
              "0.625", "0.750", "0.875", "0.999", "1.000", "1.500", "2.500",
              "3.141", "9.999", "10.000", "99.999"]:
        if v not in seen:
            entries.append((Fraction(v), v))
            seen.add(v)

    # Extended: high-precision decimals (15 decimal places)
    for v in ["0.000000000000001",  # 1e-15
              "0.123456789012345",  # 15-digit mantissa
              "9.999999999999999",  # near 10
              "999999999999.999",   # large with 3 decimal places
              "123456789.123456789"]:  # 9+9 digits
        if v not in seen:
            entries.append((Fraction(v), v))
            seen.add(v)

    # Extended: large decimals
    for v in ["1000.0", "10000.0", "100000.0", "999999.999", "1000000.0"]:
        if v not in seen:
            entries.append((Fraction(v), v))
            seen.add(v)

    return entries


def fraction_axis():
    """Generate fraction string axis values for symbolic domain testing.

    Covers: unit fractions, simple rationals, coprime pairs, near-integer fractions.
    Returns list of (Fraction_value, str_representation) tuples.
    """
    entries = []
    seen = set()

    # Unit fractions 1/n
    for n in range(1, 51):
        s = f"1/{n}"
        if s not in seen:
            entries.append((Fraction(1, n), s))
            seen.add(s)

    # Common fractions a/b (small, coprime)
    for a, b in [(2,3), (3,4), (3,5), (2,5), (4,5), (5,6), (3,7), (4,7), (5,7), (6,7),
                 (3,8), (5,8), (7,8), (2,9), (4,9), (5,9), (7,9), (8,9),
                 (3,10), (7,10), (9,10), (2,11), (3,11), (5,11), (7,11),
                 (1,13), (5,13), (7,13), (11,13),
                 (1,17), (5,17), (11,17),
                 (22,7), (99,100), (100,3), (355,113), (1,997), (1,991)]:
        if math.gcd(a, b) == a // math.gcd(a, b) * 0 + math.gcd(a, b):  # always true, just compute
            s = f"{a}/{b}"
            if s not in seen:
                entries.append((Fraction(a, b), s))
                seen.add(s)

    # Zero
    entries.append((Fraction(0), "0/1"))

    # Negative fractions
    for a, b in [(1,3), (2,3), (1,7), (3,4), (22,7)]:
        s = f"-{a}/{b}"
        entries.append((Fraction(-a, b), s))

    # Extended: larger numerators and denominators
    for a, b in [(1000,3), (999,1000), (1,10000), (7919,7), (9973,11),
                 (1,9973), (100,97), (355,113), (103993,33102)]:
        s = f"{a}/{b}"
        if s not in seen:
            entries.append((Fraction(a, b), s))
            seen.add(s)

    # Extended: near-integer fractions
    for a, b in [(99,100), (999,1000), (9999,10000), (101,100), (1001,1000)]:
        s = f"{a}/{b}"
        if s not in seen:
            entries.append((Fraction(a, b), s))
            seen.add(s)

    return entries


# ═══════════════════════════════════════════════════════════════════
# Test case generation with overflow filtering
# ═══════════════════════════════════════════════════════════════════

def fits_in_i128(frac):
    """Check if a Fraction's num and den fit in i128/u128."""
    return abs(frac.numerator) < 2**127 and frac.denominator < 2**128 and frac.denominator > 0


def result_fits_profile(result_frac, profile_name):
    """Check if result fits in the given profile's Q-format range."""
    max_int = PROFILE_MAX_INT[profile_name]
    # Result magnitude must be < max_int (integer part)
    return abs(result_frac) < max_int


def deterministic_sample(cases, max_count, rng, priority_indices=None):
    """Deterministically sample max_count cases, preserving priority entries.

    Args:
        cases: list of (a_str, b_str, result, label) tuples
        max_count: maximum number to keep
        rng: seeded Random instance
        priority_indices: set of indices that must be kept (edge cases, boundaries)
    Returns:
        sampled list, re-labeled with sequential indices
    """
    if len(cases) <= max_count:
        return cases

    if priority_indices is None:
        priority_indices = set()

    # Always keep priority entries
    priority = [cases[i] for i in sorted(priority_indices) if i < len(cases)]
    remaining_budget = max_count - len(priority)

    if remaining_budget <= 0:
        # More priority than budget — sample from priority
        return rng.sample(priority, max_count)

    # Sample from non-priority entries
    non_priority_indices = [i for i in range(len(cases)) if i not in priority_indices]
    sampled_indices = rng.sample(non_priority_indices, min(remaining_budget, len(non_priority_indices)))
    sampled = priority + [cases[i] for i in sorted(sampled_indices)]

    return sampled


def generate_binary_cases(integers):
    """Generate binary arithmetic test cases from integer axis.

    Returns dict of op -> [(a_str, b_str, Fraction_result, label), ...]
    """
    rng = random.Random(RNG_SEED)
    ops = {"add": [], "sub": [], "mul": [], "div": []}

    # For manageable size: use a representative subset for multiplication
    # Include all values up to 10^12, plus tier boundary values above that
    subset = [v for v in integers if abs(v) <= 10**12 or abs(v) in {2**31-1, 2**31, 2**32-1, 2**32}]
    full = integers

    # Identify priority values (zero, boundaries, extremes)
    priority_values = {0, 1, -1, 127, -128, 255, 256, 32767, -32768, 65535, 65536}

    # Addition: full × full (filter by result range)
    all_add = []
    add_priorities = set()
    for a in full:
        for b in full:
            result = Fraction(a + b)
            if abs(int(result)) < 2**62:  # safe for Q64.64
                idx = len(all_add)
                if a in priority_values or b in priority_values:
                    add_priorities.add(idx)
                all_add.append((str(a), str(b), result, f"bin_add_{idx}"))
    ops["add"] = deterministic_sample(all_add, MAX_PAIRS_PER_OP, rng, add_priorities)
    # Re-label
    ops["add"] = [(a, b, r, f"bin_add_{i}") for i, (a, b, r, _) in enumerate(ops["add"])]

    # Subtraction: full × full
    all_sub = []
    sub_priorities = set()
    for a in full:
        for b in full:
            result = Fraction(a - b)
            if abs(int(result)) < 2**62:
                idx = len(all_sub)
                if a in priority_values or b in priority_values:
                    sub_priorities.add(idx)
                all_sub.append((str(a), str(b), result, f"bin_sub_{idx}"))
    ops["sub"] = deterministic_sample(all_sub, MAX_PAIRS_PER_OP, rng, sub_priorities)
    ops["sub"] = [(a, b, r, f"bin_sub_{i}") for i, (a, b, r, _) in enumerate(ops["sub"])]

    # Multiplication: use subset to avoid overflow
    all_mul = []
    mul_priorities = set()
    for a in subset:
        for b in subset:
            result = Fraction(a * b)
            if abs(int(result)) < 2**62:
                idx = len(all_mul)
                if a in priority_values or b in priority_values:
                    mul_priorities.add(idx)
                all_mul.append((str(a), str(b), result, f"bin_mul_{idx}"))
    ops["mul"] = deterministic_sample(all_mul, MAX_PAIRS_PER_OP, rng, mul_priorities)
    ops["mul"] = [(a, b, r, f"bin_mul_{i}") for i, (a, b, r, _) in enumerate(ops["mul"])]

    # Division: full ÷ nonzero
    nonzero = [v for v in full if v != 0]
    all_div = []
    div_priorities = set()
    for a in full:
        for b in nonzero:
            result = Fraction(a, b)
            if abs(result) < 2**62:
                idx = len(all_div)
                if a in priority_values or b in priority_values:
                    div_priorities.add(idx)
                all_div.append((str(a), str(b), result, f"bin_div_{idx}"))
    ops["div"] = deterministic_sample(all_div, MAX_PAIRS_PER_OP, rng, div_priorities)
    ops["div"] = [(a, b, r, f"bin_div_{i}") for i, (a, b, r, _) in enumerate(ops["div"])]

    return ops


def generate_decimal_cases(decimals):
    """Generate decimal arithmetic test cases.

    Returns dict of op -> [(a_str, b_str, Fraction_result, label), ...]
    """
    ops = {"add": [], "sub": [], "mul": [], "div": []}
    idx = {"add": 0, "sub": 0, "mul": 0, "div": 0}

    for (a_frac, a_str), (b_frac, b_str) in itertools.product(decimals, repeat=2):
        for op in ["add", "sub", "mul"]:
            if op == "add":
                result = a_frac + b_frac
            elif op == "sub":
                result = a_frac - b_frac
            else:
                result = a_frac * b_frac

            if fits_in_i128(result):
                label = f"dec_{op}_{idx[op]}"
                ops[op].append((a_str, b_str, result, label))
                idx[op] += 1

        # Division: skip zero divisor
        if b_frac != 0:
            result = a_frac / b_frac
            if fits_in_i128(result):
                label = f"dec_div_{idx['div']}"
                ops["div"].append((a_str, b_str, result, label))
                idx["div"] += 1

    return ops


def generate_symbolic_cases(fractions):
    """Generate symbolic arithmetic test cases.

    Returns dict of op -> [(a_str, b_str, Fraction_result, label), ...]
    """
    rng = random.Random(RNG_SEED + 1)  # different seed from binary
    ops = {"add": [], "sub": [], "mul": [], "div": []}

    # Priority: unit fractions, zero, simple rationals
    priority_strings = {"0/1", "1/1", "1/2", "1/3", "1/7", "1/10", "22/7", "99/100"}

    for op_name in ["add", "sub", "mul", "div"]:
        all_cases = []
        priorities = set()

        for (a_frac, a_str), (b_frac, b_str) in itertools.product(fractions, repeat=2):
            if op_name == "div" and b_frac == 0:
                continue

            if op_name == "add":
                result = a_frac + b_frac
            elif op_name == "sub":
                result = a_frac - b_frac
            elif op_name == "mul":
                result = a_frac * b_frac
            else:
                result = a_frac / b_frac

            if fits_in_i128(result):
                idx = len(all_cases)
                if a_str in priority_strings or b_str in priority_strings:
                    priorities.add(idx)
                all_cases.append((a_str, b_str, result, f"sym_{op_name}_{idx}"))

        sampled = deterministic_sample(all_cases, MAX_PAIRS_PER_OP, rng, priorities)
        ops[op_name] = [(a, b, r, f"sym_{op_name}_{i}") for i, (a, b, r, _) in enumerate(sampled)]

    return ops


def generate_cross_domain_cases(integers, decimals, fractions):
    """Generate cross-domain test cases.

    Tests: integer×decimal, integer×fraction, decimal×fraction
    Returns dict of op -> [(a_str, b_str, Fraction_result, label), ...]
    """
    ops = {"add": [], "sub": [], "mul": [], "div": []}
    idx = {"add": 0, "sub": 0, "mul": 0, "div": 0}

    # Use subsets for cross-domain (otherwise combinatorial explosion)
    int_sub = [(Fraction(v), str(v)) for v in integers if abs(v) <= 10000]
    dec_sub = decimals[:40]  # first 40 decimals (was 20)
    frac_sub = fractions[:40]  # first 40 fractions (was 20)

    cross_pairs = [
        (int_sub, dec_sub, "id"),   # integer × decimal
        (int_sub, frac_sub, "if"),  # integer × fraction
        (dec_sub, frac_sub, "df"),  # decimal × fraction
    ]

    for left_vals, right_vals, prefix in cross_pairs:
        for (a_frac, a_str), (b_frac, b_str) in itertools.product(left_vals, right_vals):
            for op in ["add", "sub", "mul"]:
                if op == "add":
                    result = a_frac + b_frac
                elif op == "sub":
                    result = a_frac - b_frac
                else:
                    result = a_frac * b_frac

                if fits_in_i128(result):
                    label = f"cross_{prefix}_{op}_{idx[op]}"
                    ops[op].append((a_str, b_str, result, label))
                    idx[op] += 1

            if b_frac != 0:
                result = a_frac / b_frac
                if fits_in_i128(result):
                    label = f"cross_{prefix}_div_{idx['div']}"
                    ops["div"].append((a_str, b_str, result, label))
                    idx["div"] += 1

    return ops


# ═══════════════════════════════════════════════════════════════════
# Output writers
# ═══════════════════════════════════════════════════════════════════

HEADER = """\
// AUTO-GENERATED by scripts/generate_topology_sweep.py
// Topology sweep reference values — DO NOT EDIT
//
// Purpose: Dual-use validation + fractal topology domain mapping
// Truth:   mpmath at 250 decimal digits / exact Python Fraction arithmetic
//
// Deterministic precedence for domain ties: Symbolic > Decimal > Binary > Ternary
// (correctness-only — timing is non-deterministic and never encoded)
//
// {description}

"""


def write_rational_refs(filepath, description, all_ops, domain_prefix):
    """Write rational comparison references (common across profiles)."""
    with open(filepath, 'w') as f:
        f.write(HEADER.format(description=description))

        for op_name in ["add", "sub", "mul", "div"]:
            cases = all_ops.get(op_name, [])
            const_name = f"{domain_prefix}_{op_name.upper()}_REFS"
            f.write(f"const {const_name}: &[(&str, &str, i128, i128, &str)] = &[\n")
            for (a_str, b_str, result, label) in cases:
                num = result.numerator
                den = result.denominator
                f.write(f'    ("{a_str}", "{b_str}", {num}, {den}, "{label}"),\n')
            f.write("];\n\n")

    print(f"  wrote {filepath} ({sum(len(v) for v in all_ops.values())} entries)")


def write_binary_refs(filepath, profile_name, frac_bits, total_bits, binary_ops):
    """Write profile-specific binary Q-format references."""
    ts = type_signature(total_bits)

    with open(filepath, 'w') as f:
        f.write(HEADER.format(
            description=f"Binary domain Q{frac_bits}.{frac_bits} ({total_bits}-bit) refs for {profile_name}"
        ))

        for op_name in ["add", "sub", "mul", "div"]:
            cases = binary_ops.get(op_name, [])
            const_name = f"TOPO_BINARY_{op_name.upper()}_REFS"
            f.write(f"const {const_name}: &[(&str, &str, {ts}, &str)] = &[\n")
            for (a_str, b_str, result, label) in cases:
                qval = to_qformat(result, frac_bits)
                formatted = format_expected(qval, total_bits)
                f.write(f'    ("{a_str}", "{b_str}", {formatted}, "{label}"),\n')
            f.write("];\n\n")

    count = sum(len(v) for v in binary_ops.values())
    print(f"  wrote {filepath} ({count} entries)")


# ═══════════════════════════════════════════════════════════════════
# Grid metadata for topology mapping
# ═══════════════════════════════════════════════════════════════════

def write_grid_metadata(filepath, integers, decimals, fractions):
    """Write grid metadata for post-sweep topology table generation."""
    with open(filepath, 'w') as f:
        f.write(HEADER.format(
            description="Grid metadata for topology table generation"
        ))

        # Integer bucket boundaries (log-scale magnitudes)
        magnitudes = sorted(set(abs(v) for v in integers if v != 0))
        f.write(f"/// {len(integers)} integer axis values\n")
        f.write(f"/// Magnitude range: 0 to {max(abs(v) for v in integers)}\n")
        f.write(f"/// Positive values: {len([v for v in integers if v > 0])}\n")
        f.write(f"/// Negative values: {len([v for v in integers if v < 0])}\n")
        f.write(f"/// Zero: {'yes' if 0 in integers else 'no'}\n\n")

        f.write(f"/// {len(decimals)} decimal axis values\n")
        f.write(f"/// Range: {decimals[0][1]} to {decimals[-1][1]}\n\n")

        f.write(f"/// {len(fractions)} fraction axis values\n")
        f.write(f"/// Denominators: 1 to {max(f.denominator for f, _ in fractions)}\n\n")

        # Precedence order documentation
        f.write("/// Domain precedence for tie-breaking (higher = preferred):\n")
        f.write("/// Symbolic (3) > Decimal (2) > Binary (1) > Ternary (0)\n")
        f.write("///\n")
        f.write("/// When multiple domains give 0 ULP for an (input, operation) pair,\n")
        f.write("/// the domain with highest precedence is selected. This is a\n")
        f.write("/// mathematical property choice, NOT a performance choice.\n")
        f.write("/// Speed varies by environment; precedence is deterministic forever.\n")

    print(f"  wrote {filepath}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 70)
    print("gMath Topology Sweep Reference Generator")
    print("=" * 70)
    print(f"  mpmath precision: {mp.dps} decimal places")
    print(f"  Precedence: Symbolic > Decimal > Binary > Ternary")
    print()

    # Generate axis values
    print("Generating axis values...")
    integers = integer_axis()
    decimals = decimal_axis()
    fractions = fraction_axis()
    print(f"  integers:  {len(integers)} values ({min(integers)} to {max(integers)})")
    print(f"  decimals:  {len(decimals)} values ({decimals[0][1]} to {decimals[-1][1]})")
    print(f"  fractions: {len(fractions)} values")
    print()

    # Generate test cases per domain
    print("Generating test cases (this may take a minute)...")

    print("  binary arithmetic (integer × integer cross-product)...")
    binary_ops = generate_binary_cases(integers)
    bin_count = sum(len(v) for v in binary_ops.values())
    for op, cases in binary_ops.items():
        print(f"    {op}: {len(cases)} cases")

    print("  decimal arithmetic (decimal × decimal cross-product)...")
    decimal_ops = generate_decimal_cases(decimals)
    dec_count = sum(len(v) for v in decimal_ops.values())
    for op, cases in decimal_ops.items():
        print(f"    {op}: {len(cases)} cases")

    print("  symbolic arithmetic (fraction × fraction cross-product)...")
    symbolic_ops = generate_symbolic_cases(fractions)
    sym_count = sum(len(v) for v in symbolic_ops.values())
    for op, cases in symbolic_ops.items():
        print(f"    {op}: {len(cases)} cases")

    print("  cross-domain arithmetic...")
    cross_ops = generate_cross_domain_cases(integers, decimals, fractions)
    cross_count = sum(len(v) for v in cross_ops.values())
    for op, cases in cross_ops.items():
        print(f"    {op}: {len(cases)} cases")

    total = bin_count + dec_count + sym_count + cross_count
    print(f"\n  TOTAL: {total} test cases across all domains")
    print()

    # Write output files
    print("Writing reference files...")

    # Common rational refs (all domains except binary ULP)
    write_rational_refs(
        os.path.join(DATA_DIR, "topology_sweep_decimal.rs"),
        "Decimal domain arithmetic references",
        decimal_ops, "TOPO_DECIMAL"
    )
    write_rational_refs(
        os.path.join(DATA_DIR, "topology_sweep_symbolic.rs"),
        "Symbolic domain arithmetic references",
        symbolic_ops, "TOPO_SYMBOLIC"
    )
    write_rational_refs(
        os.path.join(DATA_DIR, "topology_sweep_cross.rs"),
        "Cross-domain arithmetic references",
        cross_ops, "TOPO_CROSS"
    )

    # Profile-specific binary refs
    for profile_name, frac_bits, total_bits in PROFILES:
        write_binary_refs(
            os.path.join(DATA_DIR, f"topology_sweep_{profile_name}.rs"),
            profile_name, frac_bits, total_bits, binary_ops
        )

    # Grid metadata
    write_grid_metadata(
        os.path.join(DATA_DIR, "topology_sweep_grid.rs"),
        integers, decimals, fractions
    )

    print()
    print("=" * 70)
    print(f"Done! {total} total reference entries.")
    print()
    print("Next steps:")
    print("  1. Run the Rust sweep per profile:")
    print("     GMATH_PROFILE=embedded   cargo test --test topology_sweep -- --nocapture 2>sweep_embedded.log")
    print("     GMATH_PROFILE=balanced   cargo test --test topology_sweep -- --nocapture 2>sweep_balanced.log")
    print("     GMATH_PROFILE=scientific cargo test --test topology_sweep -- --nocapture 2>sweep_scientific.log")
    print("  2. Post-process sweep logs into topology tables")
    print("  3. Commit topology tables to src/fixed_point/router/fractal_topology/")
    print("=" * 70)


if __name__ == "__main__":
    main()
