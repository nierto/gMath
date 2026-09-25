#!/usr/bin/env python3
"""
mpmath references for tests/decimal_exp_range_validation.rs.

Decimal exp and the functions built on it (sinh, cosh, tanh), plus the
large-argument asinh/acosh path and the sin/cos range reduction, across the
FULL representable range of each profile:

  * DecimalFixed<D> (i128 storage): every D the decimal tests use per
    profile. The result must fit both i128 at D decimals and the decimal
    compute tier (value * 10^compute_dp inside the compute integer).
  * The engine at the canonical (FASC) storage: decimal_exp at compute dp,
    materialised at DECIMAL_STORAGE_MAX_DP into BinaryStorage.

Expected values are f(x) * 10^D rounded to nearest, ties to even (the decimal
domain's rule), from mpmath at 500 digits. Inputs are exact decimals built
from integers (no binary floats anywhere). Seeded, so the file is
reproducible:

    python3 scripts/generate_decimal_exp_range_refs.py
"""

import os
import random
from mpmath import mp, mpf, exp, sinh, cosh, tanh, asinh, acosh, sin, cos, pi, floor

mp.dps = 500

PROFILES = {
    # cfg, compute dp, compute bits, canonical storage dp, BinaryStorage bits, DecimalFixed D values
    "realtime":   ("q16_16",   9,   64,  4,  32,  [4, 2, 0]),
    "compact":    ("q32_32",   19,  128, 9,  64,  [9, 4, 0]),
    "embedded":   ("q64_64",   38,  256, 19, 128, [19, 9, 0]),
    "balanced":   ("q128_128", 77,  512, 38, 256, [38, 28, 19, 0]),
    "scientific": ("q256_256", 154, 1024, 77, 512, [38, 19, 0]),
}

I128_MAX = 2**127 - 1
FNS = {"exp": exp, "sinh": sinh, "cosh": cosh, "tanh": tanh, "asinh": asinh, "acosh": acosh, "sin": sin, "cos": cos}


def round_half_even(v):
    """Nearest integer to the mpf v, ties to even."""
    f = int(floor(v))
    r = v - f
    if r > mpf(1) / 2 or (r == mpf(1) / 2 and f % 2 == 1):
        f += 1
    return f


def value(x_raw, d):
    return mpf(x_raw) / mpf(10) ** d


def expected(fn, x_raw, d):
    return round_half_even(FNS[fn](value(x_raw, d)) * mpf(10) ** d)


def fits(fn, x_raw, d, c, w):
    """The result fits i128 at d decimals and the compute tier at c decimals."""
    v = FNS[fn](value(x_raw, d))
    cmax = mpf(2 ** (w - 1) - 1) / mpf(10) ** c
    if abs(v) > cmax:
        return False
    return abs(round_half_even(v * mpf(10) ** d)) <= I128_MAX


def input_fits(x_raw, d, c, w):
    return abs(x_raw) * 10 ** (c - d) < 2 ** (w - 1) and abs(x_raw) <= I128_MAX


def largest_raw(fn, d, c, w, hi):
    """Largest x_raw (at d decimals) with fn(x) representable, by bisection."""
    lo = 0
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if fits(fn, mid, d, c, w):
            lo = mid
        else:
            hi = mid - 1
    return lo


def fixed_corpus(fn, d, c, w, rng):
    """Inputs (x_raw at d decimals) for DecimalFixed<d>::fn."""
    one = 10 ** d
    xs = set()
    if fn in ("exp", "sinh", "cosh"):
        top = largest_raw(fn, d, c, w, 2 * w * one)
        bottom = -(w + 4) * one if fn == "exp" else -top
        # every integer across the range (sparser far below zero)
        for k in range(bottom // one, top // one + 1):
            if k >= -80 or k % 16 == 0:
                xs.add(k * one)
        # the boundary and its neighbourhood
        for delta in range(0, 4):
            xs.add(top - delta)
            if fn != "exp":
                xs.add(-(top - delta))
        # random full-precision inputs
        for _ in range(60):
            xs.add(rng.randrange(bottom, top + 1))
        # small magnitudes
        for k in range(1, 6):
            xs.add(k)
            xs.add(-k)
    elif fn == "tanh":
        lim = min(2 * c + 40, 400) * one
        for k in range(-60, 61):
            xs.add(k * one)
        for _ in range(60):
            xs.add(rng.randrange(-lim, lim + 1))
        for k in range(1, 6):
            xs.add(k)
            xs.add(-k)
        # far outside (the old 2x overflowed on narrow compute tiers)
        big = min((2 ** (w - 1) - 1) // 10 ** (c - d), I128_MAX)
        xs.add(big)
        xs.add(-big)
        xs.add(big // 3)
    elif fn in ("sin", "cos"):
        # the whole argument range: powers of ten, random, and inputs next to
        # multiples of pi/2 (tiny reduced argument, the hardest reductions)
        cmax_raw = min((2 ** (w - 1) - 1) // 10 ** (c - d), I128_MAX)
        k = 0
        while 10 ** k <= cmax_raw:
            xs.add(10 ** k)
            xs.add(-(10 ** k))
            k += 1
        xs.add(cmax_raw)
        xs.add(-cmax_raw)
        for _ in range(40):
            xs.add(rng.randrange(-cmax_raw, cmax_raw + 1))
        n = 1
        while True:
            x = int(floor(n * pi / 2 * mpf(10) ** d + mpf(1) / 2))
            if x > cmax_raw:
                break
            xs.add(x)
            xs.add(-x)
            n = n * 7 + 3
    else:  # asinh / acosh: large arguments, where x^2 left the compute tier
        cmax_raw = (2 ** (w - 1) - 1) // 10 ** (c - d)
        cmax_raw = min(cmax_raw, I128_MAX)
        k = d
        while 10 ** k <= cmax_raw:
            xs.add(10 ** k)
            xs.add(10 ** k + rng.randrange(0, 10 ** k))
            if fn == "asinh":
                xs.add(-(10 ** k))
            k += 1
        xs.add(cmax_raw)
        if fn == "asinh":
            xs.add(-cmax_raw)
        for _ in range(20):
            xs.add(rng.randrange(one, cmax_raw))
        if fn == "acosh":
            xs = {x for x in xs if x >= one}
    out = []
    for x in sorted(xs):
        if not input_fits(x, d, c, w):
            continue
        if fn in ("exp", "sinh", "cosh") and not fits(fn, x, d, c, w):
            continue
        out.append((x, expected(fn, x, d)))
    return out


def engine_corpus(s, bs_bits, w, rng):
    """decimal_exp inputs at the canonical storage dp s, results in BinaryStorage."""
    one = 10 ** s
    bs_max = 2 ** (bs_bits - 1) - 1

    def ok(x):
        return abs(x) <= bs_max and abs(round_half_even(exp(value(x, s)) * mpf(10) ** s)) <= bs_max

    lo, hi = 0, 2 * w * one
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if ok(mid):
            lo = mid
        else:
            hi = mid - 1
    top = lo
    bottom = -(w + 4) * one
    xs = set()
    for k in range(bottom // one, top // one + 1):
        if k >= -80 or k % 16 == 0:
            xs.add(k * one)
    for delta in range(0, 4):
        xs.add(top - delta)
    for _ in range(60):
        xs.add(rng.randrange(bottom, top + 1))
    for k in range(1, 6):
        xs.add(k)
        xs.add(-k)
    xs = [x for x in sorted(xs) if abs(x) <= bs_max]
    return [(x, round_half_even(exp(value(x, s)) * mpf(10) ** s)) for x in xs], top


def main():
    rng = random.Random(20260925)
    lines = [
        "// Generated by scripts/generate_decimal_exp_range_refs.py (mpmath, 500 digits).",
        "// Do not edit by hand. Expected values: f(x) * 10^D rounded half to even.",
        "",
    ]
    for name, (cfg, c, w, s, bs_bits, ds) in PROFILES.items():
        lines.append(f"/// {name}: (function, D, x_raw at D, expected raw at D)")
        lines.append(f'#[cfg(table_format = "{cfg}")]')
        lines.append("pub const FIXED_REFS: &[(&str, u8, i128, i128)] = &[")
        for d in ds:
            for fn in ("exp", "sinh", "cosh", "tanh", "asinh", "acosh", "sin", "cos"):
                for x, e in fixed_corpus(fn, d, c, w, rng):
                    lines.append(f'    ("{fn}", {d}, {x}, {e}),')
        lines.append("];")
        engine, top = engine_corpus(s, bs_bits, w, rng)
        lines.append(f"/// {name}: decimal_exp at storage dp {s} (x_raw, expected), BinaryStorage range")
        lines.append(f'#[cfg(table_format = "{cfg}")]')
        lines.append("pub const ENGINE_EXP_REFS: &[(&str, &str)] = &[")
        for x, e in engine:
            lines.append(f'    ("{x}", "{e}"),')
        lines.append("];")
        lines.append(f"/// {name}: the largest x_raw at dp {s} whose exp fits BinaryStorage")
        lines.append(f'#[cfg(table_format = "{cfg}")]')
        lines.append(f'pub const ENGINE_EXP_TOP: &str = "{top}";')
        lines.append("")
    path = os.path.join(os.path.dirname(__file__), "..", "tests", "data", "decimal_exp_range_refs.rs")
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print("wrote", os.path.normpath(path))


if __name__ == "__main__":
    main()
