#!/usr/bin/env python3
"""References for FixedPoint's f64/f32 boundary (tests/data/float_boundary_refs.rs).

For every profile, and realtime at FRAC_BITS 16 and 10:
  - to-float references: raw values whose exact value raw * 2^-F is rounded to
    f64 and f32 (nearest, ties to even; f32 subnormal and infinite where the
    profile's range leaves f32's), computed on exact rationals with
    `fractions.Fraction` and Python's round-half-even. The f64 results are
    cross-checked against CPython's correctly rounded int / int division.
  - from-float vectors: f64 and f32 bit patterns and the raw each must convert
    to (the exact value truncated toward zero), or `None` where the magnitude is
    outside the storage range.
Deterministic: fixed seed, no floating-point arithmetic in the references.

Usage: python3 scripts/generate_float_boundary_refs.py
"""
import random
import struct
from fractions import Fraction

PROFILES = [
    # (name, storage width W, fraction bits F)
    ("REALTIME_F16", 32, 16),
    ("REALTIME_F10", 32, 10),
    ("COMPACT", 64, 32),
    ("EMBEDDED", 128, 64),
    ("BALANCED", 256, 128),
    ("SCIENTIFIC", 512, 256),
]

FORMATS = {"f64": (52, 11), "f32": (23, 8)}  # stored fraction bits, exponent bits


def round_half_even(q):
    """Nearest integer to the Fraction q, ties to even."""
    return round(q)  # Fraction.__round__ without ndigits rounds half to even


def floor_log2(v):
    """e with 2^e <= v < 2^(e+1), for a positive Fraction v."""
    e = v.numerator.bit_length() - v.denominator.bit_length()
    if Fraction(2) ** e > v:
        e -= 1
    elif Fraction(2) ** (e + 1) <= v:
        e += 1
    return e


def to_ieee(v, fmt):
    """IEEE bits of the exact rational v in format fmt, round to nearest even."""
    frac_bits, exp_bits = FORMATS[fmt]
    bias = (1 << (exp_bits - 1)) - 1
    if v == 0:
        return 0
    sign = 1 if v < 0 else 0
    a = abs(v)
    e = floor_log2(a)
    emin = 1 - bias
    quantum_exp = max(e, emin) - frac_bits
    q = round_half_even(a / Fraction(2) ** quantum_exp)
    if q == 0:
        bits = 0
    elif q < (1 << frac_bits):
        bits = q  # subnormal (only reachable when e < emin)
    else:
        if q == (1 << (frac_bits + 1)):
            q >>= 1
            quantum_exp += 1
        biased = quantum_exp + frac_bits + bias
        if biased >= (1 << exp_bits) - 1:
            bits = ((1 << exp_bits) - 1) << frac_bits  # infinity
        else:
            bits = (biased << frac_bits) | (q - (1 << frac_bits))
    return (sign << (frac_bits + exp_bits)) | bits


def f64_value(bits):
    """Exact Fraction of a finite f64 bit pattern."""
    return Fraction(struct.unpack("<d", struct.pack("<Q", bits))[0])


def f32_value(bits):
    return Fraction(struct.unpack("<f", struct.pack("<I", bits))[0])


def truncate_to_raw(v, W, F):
    """v * 2^F truncated toward zero, or None outside the storage range."""
    scaled = v * (1 << F)
    r = int(scaled)  # int() of a Fraction truncates toward zero
    lo, hi = -(1 << (W - 1)), (1 << (W - 1)) - 1
    return r if lo <= r <= hi else None


def raws_for(W, F, rng):
    lo, hi = -(1 << (W - 1)), (1 << (W - 1)) - 1
    raws = {0, 1, -1, 2, -2, 3, -3, hi, lo, lo + 1, hi - 1}
    for k in range(W - 1):
        raws.update({1 << k, -(1 << k)})
    for k in sorted({1, 2, 23, 24, 25, 52, 53, 54, F - 1, F, F + 1, W - 3, W - 2}):
        for r in ((1 << k) - 1, (1 << k) + 1):
            if 0 < r <= hi:
                raws.update({r, -r})
    # rounding ties and carries at 53 / 24 significant bits, at several scales
    for p in (53, 24):
        for j in range(0, W - p - 1, max(1, (W - p - 1) // 12)):
            for r in ((1 << p) + 1, (1 << p) + 3, (1 << (p + 1)) - 1, (1 << (p + 1)) + 1,
                      ((1 << p) + 1) * 2 + 1, (1 << (p + 1)) - 3):
                r <<= j
                if lo <= r <= hi:
                    raws.update({r, -r})
    # f32 subnormal boundary and overflow on the wide profiles
    for e in (-126, -127, -140, -149, -150, 127, 128):
        for d in (-2, -1, 0, 1, 2):
            r = (1 << (e + F)) + d if e + F >= 0 else None
            if r is not None and lo <= r <= hi:
                raws.update({r, -r})
    for _ in range(200):
        n = rng.randrange(1, W)
        r = rng.getrandbits(n) | (1 << (n - 1))
        if rng.random() < 0.5:
            r = -r
        if lo <= r <= hi:
            raws.add(r)
    return sorted(raws)


def float_inputs(W, F, rng, fmt):
    """Bit patterns of f64 (or f32) inputs around this profile's range."""
    frac_bits, exp_bits = FORMATS[fmt]
    pack = (lambda b: b) if fmt == "f64" else (lambda b: b & 0xFFFFFFFF)
    value = f64_value if fmt == "f64" else f32_value
    bias = (1 << (exp_bits - 1)) - 1
    top = W - 1 - F  # storage range is +-2^top
    out = set()
    # exponents at the ends of the range and around the raw step
    edges = sorted({e for c in (-F, 0, top) for e in range(c - 3, c + 3)})
    for e in edges:
        biased = e + bias
        if 1 <= biased < (1 << exp_bits) - 1:
            for fraction in (0, 1, (1 << frac_bits) - 1, 1 << (frac_bits - 1)):
                b = (biased << frac_bits) | fraction
                out.update({b, b | (1 << (frac_bits + exp_bits))})
    for fraction in (1, 2, (1 << frac_bits) - 1):  # subnormals
        out.update({fraction, fraction | (1 << (frac_bits + exp_bits))})
    out.update({0, 1 << (frac_bits + exp_bits)})  # +0, -0
    for _ in range(250):
        e = rng.randrange(-F - 2, top + 3)
        biased = e + bias
        if 1 <= biased < (1 << exp_bits) - 1:
            b = (biased << frac_bits) | rng.getrandbits(frac_bits)
            if rng.random() < 0.5:
                b |= 1 << (frac_bits + exp_bits)
            out.add(b)
    return sorted(pack(b) for b in out)


def rust_raw(r):
    """Signed raw as (negative, hex magnitude) for the test's parser."""
    return f'{"true" if r < 0 else "false"}, "{abs(r):x}"'


def main():
    rng = random.Random(20260919)
    lines = [
        "// Generated by scripts/generate_float_boundary_refs.py - do not edit.",
        "// Exact rational references (fractions.Fraction, round half to even);",
        "// f64 cross-checked against CPython's correctly rounded int / int division.",
        "",
        "/// A raw value (sign, hex magnitude) and its f64 and f32 bit patterns.",
        "pub struct ToFloat { pub negative: bool, pub magnitude: &'static str, pub f64_bits: u64, pub f32_bits: u32 }",
        "",
        "/// An f64 or f32 bit pattern and the raw it converts to (truncated toward",
        "/// zero), or `None` outside the storage range.",
        "pub struct FromFloat { pub bits: u64, pub raw: Option<(bool, &'static str)> }",
        "",
    ]
    total = 0
    for name, W, F in PROFILES:
        to_float = []
        for r in raws_for(W, F, rng):
            v = Fraction(r, 1 << F)
            b64 = to_ieee(v, "f64")
            assert b64 == struct.unpack("<Q", struct.pack("<d", r / (1 << F)))[0], (name, r)
            to_float.append(f"    ToFloat {{ negative: {rust_raw(r).split(',')[0]}, magnitude: \"{abs(r):x}\", "
                            f"f64_bits: 0x{b64:016x}, f32_bits: 0x{to_ieee(v, 'f32'):08x} }},")
        lines.append(f"pub const {name}_TO_FLOAT: &[ToFloat] = &[")
        lines.extend(to_float)
        lines.append("];")
        for fmt, value in (("f64", f64_value), ("f32", f32_value)):
            vectors = []
            for b in float_inputs(W, F, rng, fmt):
                raw = truncate_to_raw(value(b), W, F)
                expected = "None" if raw is None else f"Some(({rust_raw(raw)}))"
                vectors.append(f"    FromFloat {{ bits: 0x{b:x}, raw: {expected} }},")
            lines.append(f"pub const {name}_FROM_{fmt.upper()}: &[FromFloat] = &[")
            lines.extend(vectors)
            lines.append("];")
            total += len(vectors)
        total += len(to_float)
        lines.append("")
    out = "tests/data/float_boundary_refs.rs"
    with open(out, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {out}: {total} references")


if __name__ == "__main__":
    main()
