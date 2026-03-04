#!/usr/bin/env python3
"""
Generate ZASC pipeline ULP reference data with 1000+ samples per function.

Unlike generate_ulp_references.py (which outputs Q-format integers for direct
binary function testing), this script outputs STRING inputs suitable for the
ZASC pipeline: gmath("1.5").exp() → evaluate → as_binary_storage → ULP compare.

Output: tests/data/zasc_ulp_refs_{q64_64,q128_128,q256_256}.rs
Format: (input_str, expected_value, label) for unary functions
        (a_str, b_str, expected_value, label) for binary functions

Q64.64:   expected = i128
Q128.128: expected = [u64; 4]  (I256 little-endian words)
Q256.256: expected = [u64; 8]  (I512 little-endian words)

Uses mpmath at 250 decimal places → Q_N.N hex = round(value * 2^N).

Critical: Integer exponents use "10" (no decimal point) so CompactShadow gives
denominator=1, triggering the pow integer fast path. "10.0" would give (100,10)
→ slow path, since from_rational() does NOT GCD-reduce.
"""

import os
import sys
import random
import math
from mpmath import mp, mpf, pi, e, sqrt, log, exp, sin, cos, tan, atan, atan2
from mpmath import asin, acos, sinh, cosh, tanh, asinh, acosh, atanh, power, nint

mp.dps = 250  # well above 77 digits needed for Q256.256

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "tests", "data")

# Seed for reproducibility
random.seed(42)


# ═══════════════════════════════════════════════════════════════════
# Q-format conversion
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

def float_to_str(value, label=None):
    """Convert a Python/mpmath float to a string suitable for gmath().

    Returns a decimal string with limited precision to avoid overflow in
    gmath()'s parse_decimal path: (scaled_value << 64) must fit in i128.
    For Q64.64, we limit to 18 significant digits.

    Special handling: integers must NOT have decimal point (for pow fast path).
    """
    if label is not None:
        return label

    # Use mpmath to format with 18 significant digits.
    # Q64.64 has ~19.3 digits of precision, so 18 sig digits is plenty.
    # More digits cause (scaled << 64) overflow in parse_decimal's i128 path.
    s = mp.nstr(value, 18, strip_zeros=True)

    # Ensure it has a decimal point (so gmath() routes to decimal/binary, not symbolic)
    # but NOT for integers used as pow exponents (those are handled separately)
    if '.' not in s and 'e' not in s and 'E' not in s:
        s = s + ".0"

    # Avoid scientific notation — gmath() may not parse it
    if 'e' in s or 'E' in s:
        # Convert scientific notation to fixed-point
        val = mpf(value)
        # Determine number of decimal places needed
        abs_val = abs(float(val))
        if abs_val == 0:
            return "0.0"
        if abs_val >= 1:
            # Integer digits + some fractional
            int_digits = len(str(int(abs(float(val)))))
            frac_digits = max(1, 18 - int_digits)
        else:
            # Small number: need enough zeros + significant digits
            # e.g., 0.0001234 → need ~7 decimal places
            import math as pymath
            leading_zeros = -int(pymath.floor(pymath.log10(abs_val)))
            frac_digits = leading_zeros + 17
            frac_digits = min(frac_digits, 18)  # cap total decimal places
        s = mp.nstr(val, 18, strip_zeros=True)
        # If still in sci notation, use fixed-point format
        if 'e' in s or 'E' in s:
            fmt = f"{{:.{frac_digits}f}}"
            s = fmt.format(float(mp.nstr(val, 20)))
            # Strip trailing zeros but keep at least one decimal digit
            if '.' in s:
                s = s.rstrip('0')
                if s.endswith('.'):
                    s = s + '0'

    return s


# ═══════════════════════════════════════════════════════════════════
# Domain-specific point generators
# ═══════════════════════════════════════════════════════════════════

def edge_points_uniform(lo, hi, n_edge=20):
    """Generate edge cases for a uniform domain [lo, hi]."""
    edges = []
    # Boundary points
    edges.append(lo)
    edges.append(hi)
    # Near boundaries
    span = hi - lo
    for delta in [1e-6, 1e-4, 1e-2, 0.1]:
        if lo + delta < hi:
            edges.append(lo + delta)
        if hi - delta > lo:
            edges.append(hi - delta)
    # Near zero (if in range)
    if lo < 0 < hi:
        for v in [0, 1e-8, -1e-8, 1e-4, -1e-4, 0.001, -0.001]:
            if lo <= v <= hi:
                edges.append(v)
    # Near ±1 (if in range)
    for v in [1, -1, 1 - 1e-6, -1 + 1e-6]:
        if lo <= v <= hi:
            edges.append(v)
    # Deduplicate and trim
    edges = sorted(set(edges))
    return edges[:n_edge]


def uniform_grid(lo, hi, n):
    """n uniformly spaced points in [lo, hi]."""
    if n <= 1:
        return [(lo + hi) / 2]
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def log_spaced(lo, hi, n):
    """n log-spaced points in [lo, hi] (both positive)."""
    if lo <= 0:
        lo = 1e-10
    log_lo = math.log(lo)
    log_hi = math.log(hi)
    return [math.exp(log_lo + (log_hi - log_lo) * i / (n - 1)) for i in range(n)]


def log_density_near_zero(lo, hi, n):
    """More density near zero, log-spaced outward."""
    points = []
    # Positive side
    if hi > 0:
        pos = log_spaced(max(1e-6, lo if lo > 0 else 1e-6), hi, n // 2)
        points.extend(pos)
    # Negative side
    if lo < 0:
        neg = [-x for x in log_spaced(max(1e-6, -hi if hi < 0 else 1e-6), -lo, n // 2)]
        points.extend(neg)
    # Near zero
    for v in [0, 1e-8, -1e-8, 1e-6, -1e-6]:
        if lo <= v <= hi:
            points.append(v)
    points = sorted(set(points))
    return points[:n]


# ═══════════════════════════════════════════════════════════════════
# Per-function point generation
# ═══════════════════════════════════════════════════════════════════

def gen_exp_points(n=1000):
    """exp: [-20, 20]. Edge cases + uniform grid."""
    edges = edge_points_uniform(-20, 20)
    grid = uniform_grid(-20, 20, n - len(edges))
    all_pts = sorted(set(edges + grid))[:n]
    result = []
    for v in all_pts:
        s = float_to_str(mpf(v))
        result.append((s, mpf(v), f"exp_{len(result)}"))
    return result


def gen_ln_points(n=1000):
    """ln: (0.0001, 10000]. Edge cases + log-spaced."""
    edges = [0.0001, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 1.0,
             1.001, 1.01, 1.1, 2.0, math.e, 10.0, 100.0, 1000.0, 10000.0]
    grid = log_spaced(0.0001, 10000, n - len(edges))
    all_pts = sorted(set(edges + grid))[:n]
    result = []
    for v in all_pts:
        s = float_to_str(mpf(v))
        result.append((s, mpf(v), f"ln_{len(result)}"))
    return result


def gen_sqrt_points(n=1000):
    """sqrt: [0.0001, 10000]. Edge cases + log-spaced."""
    edges = [0.0001, 0.001, 0.01, 0.25, 0.5, 1.0, 2.0, 3.0, 4.0, 9.0,
             10.0, 16.0, 25.0, 100.0, 1000.0, 10000.0]
    grid = log_spaced(0.0001, 10000, n - len(edges))
    all_pts = sorted(set(edges + grid))[:n]
    result = []
    for v in all_pts:
        s = float_to_str(mpf(v))
        result.append((s, mpf(v), f"sqrt_{len(result)}"))
    return result


def gen_sin_points(n=1000):
    """sin: [-10, 10]. Edge cases + uniform grid."""
    edges = edge_points_uniform(-10, 10)
    # Add multiples of pi
    for k in range(-3, 4):
        edges.append(float(k * pi))
        edges.append(float(k * pi / 2))
        edges.append(float(k * pi / 4))
    edges = [v for v in edges if -10 <= v <= 10]
    grid = uniform_grid(-10, 10, n - len(edges))
    all_pts = sorted(set(edges + grid))[:n]
    result = []
    for v in all_pts:
        s = float_to_str(mpf(v))
        result.append((s, mpf(v), f"sin_{len(result)}"))
    return result


def gen_cos_points(n=1000):
    """cos: [-10, 10]. Same domain as sin."""
    edges = edge_points_uniform(-10, 10)
    for k in range(-3, 4):
        edges.append(float(k * pi))
        edges.append(float(k * pi / 2))
        edges.append(float(k * pi / 4))
    edges = [v for v in edges if -10 <= v <= 10]
    grid = uniform_grid(-10, 10, n - len(edges))
    all_pts = sorted(set(edges + grid))[:n]
    result = []
    for v in all_pts:
        s = float_to_str(mpf(v))
        result.append((s, mpf(v), f"cos_{len(result)}"))
    return result


def gen_tan_points(n=1000):
    """tan: [-1.5, 1.5]. Avoid poles at ±π/2 ≈ ±1.5708."""
    lo, hi = -1.5, 1.5
    edges = edge_points_uniform(lo, hi)
    grid = uniform_grid(lo, hi, n - len(edges))
    all_pts = sorted(set(edges + grid))[:n]
    result = []
    for v in all_pts:
        s = float_to_str(mpf(v))
        result.append((s, mpf(v), f"tan_{len(result)}"))
    return result


def gen_atan_points(n=1000):
    """atan: [-100, 100]. Log-density near 0."""
    edges = [-100, -10, -1, -0.5, 0, 0.5, 1, 10, 100]
    grid = log_density_near_zero(-100, 100, n - len(edges))
    all_pts = sorted(set(edges + grid))[:n]
    result = []
    for v in all_pts:
        s = float_to_str(mpf(v))
        result.append((s, mpf(v), f"atan_{len(result)}"))
    return result


def gen_atan2_points(n=1000):
    """atan2(y,x): quadrant-spanning pairs."""
    result = []
    # Edge cases: axes, diagonals
    edge_pairs = [
        (1, 0), (0, 1), (-1, 0), (0, -1),  # axes
        (1, 1), (-1, 1), (-1, -1), (1, -1),  # diagonals
        (0.001, 1), (1, 0.001), (-0.001, 1), (1, -0.001),  # near axes
        (3, 4), (4, 3), (-3, 4), (3, -4),  # Pythagorean
        (0.5, 0.5), (10, 10), (100, 1), (1, 100),
    ]
    for y, x in edge_pairs:
        ys = float_to_str(mpf(y))
        xs = float_to_str(mpf(x))
        result.append((ys, xs, mpf(y), mpf(x), f"atan2_{len(result)}"))

    # Random quadrant-spanning pairs
    remaining = n - len(result)
    for i in range(remaining):
        # Random angle, random radius
        angle = random.uniform(-math.pi, math.pi)
        radius = math.exp(random.uniform(-3, 5))  # 0.05 to 148
        y = radius * math.sin(angle)
        x = radius * math.cos(angle)
        ys = float_to_str(mpf(y))
        xs = float_to_str(mpf(x))
        result.append((ys, xs, mpf(y), mpf(x), f"atan2_{len(result)}"))

    return result[:n]


def gen_asin_points(n=1000):
    """asin: [-0.999, 0.999]. Extra density near ±1."""
    edges = [-0.999, -0.99, -0.9, -0.5, 0, 0.5, 0.9, 0.99, 0.999,
             -0.9999, 0.9999, -0.001, 0.001, -0.1, 0.1]
    grid = uniform_grid(-0.999, 0.999, n - len(edges))
    # Add extra near ±1
    for i in range(50):
        grid.append(1 - 10 ** random.uniform(-6, -1))
        grid.append(-1 + 10 ** random.uniform(-6, -1))
    all_pts = sorted(set(edges + grid))
    all_pts = [v for v in all_pts if -0.9999 <= v <= 0.9999][:n]
    result = []
    for v in all_pts:
        s = float_to_str(mpf(v))
        result.append((s, mpf(v), f"asin_{len(result)}"))
    return result


def gen_acos_points(n=1000):
    """acos: [-0.999, 0.999]. Same domain as asin."""
    edges = [-0.999, -0.99, -0.9, -0.5, 0, 0.5, 0.9, 0.99, 0.999,
             -0.9999, 0.9999, -0.001, 0.001, -0.1, 0.1]
    grid = uniform_grid(-0.999, 0.999, n - len(edges))
    for i in range(50):
        grid.append(1 - 10 ** random.uniform(-6, -1))
        grid.append(-1 + 10 ** random.uniform(-6, -1))
    all_pts = sorted(set(edges + grid))
    all_pts = [v for v in all_pts if -0.9999 <= v <= 0.9999][:n]
    result = []
    for v in all_pts:
        s = float_to_str(mpf(v))
        result.append((s, mpf(v), f"acos_{len(result)}"))
    return result


def gen_sinh_points(n=1000):
    """sinh: [-5, 5]."""
    edges = edge_points_uniform(-5, 5)
    grid = uniform_grid(-5, 5, n - len(edges))
    all_pts = sorted(set(edges + grid))[:n]
    result = []
    for v in all_pts:
        s = float_to_str(mpf(v))
        result.append((s, mpf(v), f"sinh_{len(result)}"))
    return result


def gen_cosh_points(n=1000):
    """cosh: [-5, 5]."""
    edges = edge_points_uniform(-5, 5)
    grid = uniform_grid(-5, 5, n - len(edges))
    all_pts = sorted(set(edges + grid))[:n]
    result = []
    for v in all_pts:
        s = float_to_str(mpf(v))
        result.append((s, mpf(v), f"cosh_{len(result)}"))
    return result


def gen_tanh_points(n=1000):
    """tanh: [-5, 5]."""
    edges = edge_points_uniform(-5, 5)
    grid = uniform_grid(-5, 5, n - len(edges))
    all_pts = sorted(set(edges + grid))[:n]
    result = []
    for v in all_pts:
        s = float_to_str(mpf(v))
        result.append((s, mpf(v), f"tanh_{len(result)}"))
    return result


def gen_asinh_points(n=1000):
    """asinh: [-100, 100]. Log-density near 0."""
    edges = [-100, -10, -1, 0, 1, 10, 100]
    grid = log_density_near_zero(-100, 100, n - len(edges))
    all_pts = sorted(set(edges + grid))[:n]
    result = []
    for v in all_pts:
        s = float_to_str(mpf(v))
        result.append((s, mpf(v), f"asinh_{len(result)}"))
    return result


def gen_acosh_points(n=1000):
    """acosh: [1.001, 100]. Log-spaced."""
    edges = [1.001, 1.01, 1.1, 1.5, 2.0, 3.0, 5.0, 10.0, 50.0, 100.0,
             1.0001, 1.00001]
    grid = log_spaced(1.001, 100, n - len(edges))
    all_pts = sorted(set(edges + grid))[:n]
    result = []
    for v in all_pts:
        s = float_to_str(mpf(v))
        result.append((s, mpf(v), f"acosh_{len(result)}"))
    return result


def gen_atanh_points(n=1000):
    """atanh: (-0.999, 0.999)."""
    edges = [-0.999, -0.99, -0.9, -0.5, 0, 0.5, 0.9, 0.99, 0.999,
             -0.001, 0.001, -0.1, 0.1]
    grid = uniform_grid(-0.999, 0.999, n - len(edges))
    # Extra density near ±1
    for i in range(50):
        grid.append(1 - 10 ** random.uniform(-6, -1))
        grid.append(-1 + 10 ** random.uniform(-6, -1))
    all_pts = sorted(set(edges + grid))
    all_pts = [v for v in all_pts if -0.9999 <= v <= 0.9999][:n]
    result = []
    for v in all_pts:
        s = float_to_str(mpf(v))
        result.append((s, mpf(v), f"atanh_{len(result)}"))
    return result


def gen_pow_integer_points(n=500):
    """pow with integer exponents. Input strings: base has '.', exp does NOT.

    Critical: exponent must be like "10" not "10.0" so CompactShadow sees den=1
    and the pow integer fast path is triggered.
    """
    result = []
    # Fixed edge cases
    edge_cases = [
        (2.0, 0), (2.0, 1), (2.0, 10), (2.0, 20), (2.0, -1), (2.0, -10),
        (3.0, 0), (3.0, 5), (3.0, 10), (3.0, -5),
        (0.5, 1), (0.5, 10), (0.5, 20), (0.5, -5),
        (10.0, 0), (10.0, 1), (10.0, 5), (10.0, -3),
        (1.5, 10), (1.5, -10),
    ]
    for base, exp_int in edge_cases:
        base_s = float_to_str(mpf(base))
        exp_s = str(exp_int)  # NO decimal point!
        result.append((base_s, exp_s, mpf(base), mpf(exp_int), f"pow_int_{len(result)}"))

    # Random integer-exponent pairs
    remaining = n - len(result)
    for i in range(remaining):
        base = random.uniform(0.5, 10.0)
        exp_int = random.randint(-10, 20)
        # Check result won't overflow Q64.64 (max ~9.2e18)
        try:
            ref_val = float(power(mpf(base), mpf(exp_int)))
            if abs(ref_val) > 1e18 or abs(ref_val) < 1e-18:
                # Skip extreme results
                base = random.uniform(0.8, 3.0)
                exp_int = random.randint(-5, 10)
        except:
            continue
        base_s = float_to_str(mpf(base))
        exp_s = str(exp_int)
        result.append((base_s, exp_s, mpf(base), mpf(exp_int), f"pow_int_{len(result)}"))

    return result[:n]


def gen_pow_fractional_points(n=500):
    """pow with fractional exponents. Both inputs have decimal points."""
    result = []
    # Fixed edge cases
    edge_cases = [
        (2.0, 0.5), (4.0, 0.5), (9.0, 0.5),  # sqrt via pow
        (2.0, 0.25), (2.0, 0.1), (2.0, 1.5),
        (math.e, 1.0), (math.e, 2.0), (math.e, 0.5),
        (10.0, 0.5), (10.0, 0.1), (10.0, -0.5),
        (0.5, 0.5), (0.5, 1.5), (0.5, 2.5),
        (3.0, 1.5), (5.0, 0.3), (7.0, 0.7),
    ]
    for base, exp_frac in edge_cases:
        base_s = float_to_str(mpf(base))
        exp_s = float_to_str(mpf(exp_frac))
        result.append((base_s, exp_s, mpf(base), mpf(exp_frac), f"pow_frac_{len(result)}"))

    # Random fractional-exponent pairs
    remaining = n - len(result)
    for i in range(remaining):
        base = random.uniform(0.1, 10.0)
        exp_frac = random.uniform(-3.0, 3.0)
        # Ensure fractional (not integer)
        if exp_frac == int(exp_frac):
            exp_frac += 0.01
        # Check result won't overflow
        try:
            ref_val = float(power(mpf(base), mpf(exp_frac)))
            if abs(ref_val) > 1e18 or abs(ref_val) < 1e-18:
                base = random.uniform(0.5, 5.0)
                exp_frac = random.uniform(-1.0, 2.0)
        except:
            continue
        base_s = float_to_str(mpf(base))
        exp_s = float_to_str(mpf(exp_frac))
        result.append((base_s, exp_s, mpf(base), mpf(exp_frac), f"pow_frac_{len(result)}"))

    return result[:n]


# ═══════════════════════════════════════════════════════════════════
# Output generation
# ═══════════════════════════════════════════════════════════════════

def write_unary_refs(f, const_name, points, mp_func, frac_bits, total_bits):
    """Write unary reference: (input_str, expected_value, label).

    CRITICAL: Compute reference from the STRING, not the original mpf value.
    The ZASC pipeline will parse the string, so the reference must account for
    any rounding introduced by float_to_str's 18-digit limit.
    """
    type_sig = type_signature(total_bits)
    max_val = (1 << (total_bits - 1)) - 1
    min_val = -(1 << (total_bits - 1))

    f.write(f"const {const_name}: &[(&str, {type_sig}, &str)] = &[\n")
    count = 0
    skipped = 0
    for item in points:
        input_str, _mp_val, label = item
        try:
            # Re-parse string at 250 decimal places — this is what gmath() sees
            str_val = mpf(input_str)
            result = mp_func(str_val)
            expected = to_qformat(result, frac_bits)
            if expected > max_val or expected < min_val:
                skipped += 1
                continue
            formatted = format_expected(expected, total_bits)
            f.write(f"    (\"{input_str}\", {formatted}, \"{label}\"),\n")
            count += 1
        except Exception as ex:
            print(f"  WARNING: {const_name} input={input_str}: {ex}", file=sys.stderr)
            skipped += 1
    f.write("];\n\n")
    return count, skipped


def write_binary_refs(f, const_name, points, mp_func, frac_bits, total_bits):
    """Write binary reference: (a_str, b_str, expected_value, label).

    CRITICAL: Compute reference from the STRINGS, not original mpf values.
    """
    type_sig = type_signature(total_bits)
    max_val = (1 << (total_bits - 1)) - 1
    min_val = -(1 << (total_bits - 1))

    f.write(f"const {const_name}: &[(&str, &str, {type_sig}, &str)] = &[\n")
    count = 0
    skipped = 0
    for item in points:
        a_str, b_str, _mp_a, _mp_b, label = item
        try:
            str_a = mpf(a_str)
            str_b = mpf(b_str)
            result = mp_func(str_a, str_b)
            expected = to_qformat(result, frac_bits)
            if expected > max_val or expected < min_val:
                skipped += 1
                continue
            formatted = format_expected(expected, total_bits)
            f.write(f"    (\"{a_str}\", \"{b_str}\", {formatted}, \"{label}\"),\n")
            count += 1
        except Exception as ex:
            print(f"  WARNING: {const_name} a={a_str} b={b_str}: {ex}", file=sys.stderr)
            skipped += 1
    f.write("];\n\n")
    return count, skipped


PROFILES = [
    ("q64_64",   64,  128),
    ("q128_128", 128, 256),
    ("q256_256", 256, 512),
]


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"mpmath precision: {mp.dps} decimal places")

    # Generate all point sets once (shared across profiles)
    print("Generating test points...")
    exp_pts = gen_exp_points(1000)
    ln_pts = gen_ln_points(1000)
    sqrt_pts = gen_sqrt_points(1000)
    sin_pts = gen_sin_points(1000)
    cos_pts = gen_cos_points(1000)
    tan_pts = gen_tan_points(1000)
    atan_pts = gen_atan_points(1000)
    atan2_pts = gen_atan2_points(1000)
    asin_pts = gen_asin_points(1000)
    acos_pts = gen_acos_points(1000)
    sinh_pts = gen_sinh_points(1000)
    cosh_pts = gen_cosh_points(1000)
    tanh_pts = gen_tanh_points(1000)
    asinh_pts = gen_asinh_points(1000)
    acosh_pts = gen_acosh_points(1000)
    atanh_pts = gen_atanh_points(1000)
    pow_int_pts = gen_pow_integer_points(500)
    pow_frac_pts = gen_pow_fractional_points(500)

    unary_funcs = [
        ("ZASC_EXP_REFS", exp_pts, exp),
        ("ZASC_LN_REFS", ln_pts, log),
        ("ZASC_SQRT_REFS", sqrt_pts, sqrt),
        ("ZASC_SIN_REFS", sin_pts, sin),
        ("ZASC_COS_REFS", cos_pts, cos),
        ("ZASC_TAN_REFS", tan_pts, tan),
        ("ZASC_ATAN_REFS", atan_pts, atan),
        ("ZASC_ASIN_REFS", asin_pts, asin),
        ("ZASC_ACOS_REFS", acos_pts, acos),
        ("ZASC_SINH_REFS", sinh_pts, sinh),
        ("ZASC_COSH_REFS", cosh_pts, cosh),
        ("ZASC_TANH_REFS", tanh_pts, tanh),
        ("ZASC_ASINH_REFS", asinh_pts, asinh),
        ("ZASC_ACOSH_REFS", acosh_pts, acosh),
        ("ZASC_ATANH_REFS", atanh_pts, atanh),
    ]

    binary_funcs = [
        ("ZASC_ATAN2_REFS", atan2_pts, atan2),
        ("ZASC_POW_INTEGER_REFS", pow_int_pts, power),
        ("ZASC_POW_FRACTIONAL_REFS", pow_frac_pts, power),
    ]

    grand_total = 0

    for profile_name, frac_bits, total_bits in PROFILES:
        filename = os.path.join(DATA_DIR, f"zasc_ulp_refs_{profile_name}.rs")
        n_words = total_bits // 64
        print(f"\n{'='*60}")
        print(f"Generating {filename}")
        print(f"  Q{frac_bits}.{frac_bits} ({total_bits}-bit, {n_words} words)")
        print(f"{'='*60}")

        total_count = 0
        total_skipped = 0

        with open(filename, "w") as f:
            f.write(f"// AUTO-GENERATED by scripts/generate_zasc_ulp_references.py\n")
            f.write(f"// Q{frac_bits}.{frac_bits} ({total_bits}-bit) ZASC pipeline references\n")
            f.write(f"// mpmath precision: {mp.dps} decimal places\n")
            f.write(f"// 1000+ samples per function\n")
            f.write(f"//\n")
            if total_bits == 128:
                f.write(f"// Unary format: (input_str, expected_i128, label)\n")
                f.write(f"// Binary format: (a_str, b_str, expected_i128, label)\n")
            else:
                f.write(f"// Unary format: (input_str, expected_[u64;{n_words}], label)\n")
                f.write(f"// Binary format: (a_str, b_str, expected_[u64;{n_words}], label)\n")
            f.write(f"//\n")
            f.write(f"// Input strings go through gmath() -> full ZASC pipeline.\n")
            f.write(f"// Integer exponents have NO decimal point to trigger pow fast path.\n\n")

            for const_name, points, mp_func in unary_funcs:
                print(f"  {const_name}: {len(points)} points...", end="")
                cnt, skip = write_unary_refs(f, const_name, points, mp_func, frac_bits, total_bits)
                print(f" -> {cnt} written, {skip} skipped")
                total_count += cnt
                total_skipped += skip

            for const_name, points, mp_func in binary_funcs:
                print(f"  {const_name}: {len(points)} points...", end="")
                cnt, skip = write_binary_refs(f, const_name, points, mp_func, frac_bits, total_bits)
                print(f" -> {cnt} written, {skip} skipped")
                total_count += cnt
                total_skipped += skip

        print(f"  Done: {total_count} refs, {total_skipped} skipped")
        grand_total += total_count

    print(f"\nAll {len(PROFILES)} profiles generated: {grand_total} total references")


if __name__ == "__main__":
    main()
