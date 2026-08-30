#!/usr/bin/env python3
"""Generate independent reference values for tests/data/certified_geometry_refs.rs.

Independent means: computed here in Python's exact integers and fractions,
cross-checked against mpmath at 300 digits, by code that shares nothing with
the Rust implementation. The Rust gates already check the interval and
predicate results against i128 models written alongside the code; this file
is the second opinion the mpmath rule asks for.

What is emitted, per binary profile (F = fractional bits, W = storage bits):

  SQRT      (x_raw, floor_raw, ceil_raw): the certified sqrt endpoints
            floor = isqrt(x_raw << F), ceil = floor + [floor^2 != x_raw << F]
  MUL, DIV  (a_raw, b_raw, floor_raw, ceil_raw): directed endpoints of a*b and
            a/b at the storage scale, including operands near the storage
            maximum so the wide multiply paths are exercised
  PIVOT     (n, floor_raw, ceil_raw): floor and ceil at scale 2^F of the
            EXACT rational last Cholesky pivot of the dyadic A^T A + I matrix
            that tests/pd_verdict_validation.rs builds from its LCG (seed
            0x1D7, n = 23 then n = 50 from one stream); the interval Cholesky
            must enclose it
  ORIENT2D, ORIENT3D, INCIRCLE, INSPHERE
            (coordinates..., sign): configurations scaled and translated to
            within a factor of four of the storage maximum; sign evaluated on
            the actual integers

and profile-independently for the decimal domain (D = decimal places):

  DSQRT, DMUL, DDIV as i128 literals.

Every raw value is emitted as exact little-endian two's-complement bytes of the
storage width (4, 8, 16, 32 or 64 bytes) so that the wide profiles, whose raws
exceed i128, need no string parsing on the Rust side.

Cross-checks performed here (a failure aborts generation):
  - isqrt agrees with floor(mpmath.sqrt) at 300 digits for every SQRT entry
  - every exact rational pivot agrees with an mpmath Cholesky at 300 digits
    to better than 1e-200 relative
  - every predicate sign agrees with the sign of the mpmath determinant of
    the same configuration evaluated at 300 digits (values are exact
    integers, so this only checks the two determinant expansions agree)

Run from the repo root:
    python3 scripts/generate_certified_geometry_refs.py
"""

from fractions import Fraction
from math import isqrt
import sys

from mpmath import mp, mpf, sqrt as mpsqrt, matrix as mpmatrix

mp.dps = 300

PROFILES = [
    # name,      F,   W
    ("q16_16",   16,  32),
    ("q32_32",   32,  64),
    ("q64_64",   64,  128),
    ("q128_128", 128, 256),
    ("q256_256", 256, 512),
]

OUT = "tests/data/certified_geometry_refs.rs"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def le_bytes(v: int, width_bits: int) -> str:
    """Rust literal for the exact two's-complement LE bytes of v at width."""
    n = width_bits // 8
    lo, hi = -(1 << (width_bits - 1)), (1 << (width_bits - 1)) - 1
    assert lo <= v <= hi, f"{v} does not fit {width_bits} bits"
    b = (v & ((1 << width_bits) - 1)).to_bytes(n, "little")
    return "&[" + ", ".join(str(x) for x in b) + "]"


def floor_div(n: int, d: int) -> int:
    return n // d  # Python floors for positive and negative operands alike


def ceil_div(n: int, d: int) -> int:
    return -((-n) // d)


def sign(x: int) -> int:
    return (x > 0) - (x < 0)


class Rng:
    """Bit-exact replica of the test suites' LCG."""

    def __init__(self, seed: int):
        self.s = seed & 0xFFFFFFFFFFFFFFFF

    def next(self) -> int:
        self.s = (self.s * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        x = self.s
        return ((x >> 33) ^ x) & 0xFFFFFFFFFFFFFFFF

    def dyadic(self) -> Fraction:
        k = (self.next() % 17) - 8
        return Fraction(k, 16)


def dyadic_spd(rng: Rng, n: int):
    a = [[rng.dyadic() for _ in range(n)] for _ in range(n)]
    m = [[Fraction(0)] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            s = sum(a[k][i] * a[k][j] for k in range(n))
            if i == j:
                s += 1
            m[i][j] = s
    return m


def exact_cholesky_pivots(m):
    """Exact rational pivots d_i = m_ii - sum_k l_ik^2 (l_ik^2 is rational)."""
    n = len(m)
    # l2[i][k] = l_ik^2 as Fraction; l_ik * l_jk = (m_ik-ish) products need l itself,
    # so carry l_ik as (numerator_k, pivot_k): l_ik = c_ik / sqrt(d_k) with c_ik rational.
    c = [[Fraction(0)] * n for _ in range(n)]
    d = [Fraction(0)] * n
    for i in range(n):
        # pivot: m_ii - sum_k c_ik^2 / d_k
        d[i] = m[i][i] - sum(c[i][k] * c[i][k] / d[k] for k in range(i))
        assert d[i] > 0, "matrix is not PD"
        for j in range(i + 1, n):
            # l_ji = (m_ji - sum_k l_jk l_ik) / l_ii = (m_ji - sum_k c_jk c_ik / d_k) / sqrt(d_i)
            c[j][i] = m[j][i] - sum(c[j][k] * c[i][k] / d[k] for k in range(i))
    return d


def mp_cholesky_last_pivot(m):
    n = len(m)
    A = mpmatrix(n, n)
    for i in range(n):
        for j in range(n):
            A[i, j] = mpf(m[i][j].numerator) / mpf(m[i][j].denominator)
    L = mpmatrix(n, n)
    for i in range(n):
        s = A[i, i] - sum(L[i, k] ** 2 for k in range(i))
        last = s
        L[i, i] = mpsqrt(s)
        for j in range(i + 1, n):
            L[j, i] = (A[j, i] - sum(L[j, k] * L[i, k] for k in range(i))) / L[i, i]
    return last


def det2(r0, r1):
    return r0[0] * r1[1] - r0[1] * r1[0]


def det3(r0, r1, r2):
    return (r0[0] * (r1[1] * r2[2] - r1[2] * r2[1])
            - r0[1] * (r1[0] * r2[2] - r1[2] * r2[0])
            + r0[2] * (r1[0] * r2[1] - r1[1] * r2[0]))


def orient2d(a, b, c):
    return sign(det2([a[0] - c[0], a[1] - c[1]], [b[0] - c[0], b[1] - c[1]]))


def orient3d(a, b, c, d):
    sub = lambda p, q: [p[i] - q[i] for i in range(3)]
    return sign(det3(sub(a, d), sub(b, d), sub(c, d)))


def incircle(a, b, c, d):
    sub = lambda p, q: [p[i] - q[i] for i in range(2)]
    ad, bd, cd = sub(a, d), sub(b, d), sub(c, d)
    lift = lambda v: v[0] * v[0] + v[1] * v[1]
    return sign(lift(ad) * det2(bd, cd) - lift(bd) * det2(ad, cd) + lift(cd) * det2(ad, bd))


def insphere(a, b, c, d, e):
    sub = lambda p, q: [p[i] - q[i] for i in range(3)]
    ae, be, ce, de = sub(a, e), sub(b, e), sub(c, e), sub(d, e)
    lift = lambda v: v[0] * v[0] + v[1] * v[1] + v[2] * v[2]
    return sign(-lift(ae) * det3(be, ce, de) + lift(be) * det3(ae, ce, de)
                - lift(ce) * det3(ae, be, de) + lift(de) * det3(ae, be, ce))


def mp_det(rows):
    n = len(rows)
    M = mpmatrix(n, n)
    for i in range(n):
        for j in range(n):
            M[i, j] = mpf(rows[i][j])
    from mpmath import det
    return det(M)


# ---------------------------------------------------------------------------
# per-profile content
# ---------------------------------------------------------------------------

def sqrt_entries(F, W):
    one = 1 << F
    xs = [
        0, 1, 2, 3, one, 2 * one, 4 * one, one // 2, one // 4 + 1,
        (1 << (W - 2)) + 12345,          # near the storage maximum
        (1 << (W - 1)) - 1,              # the storage maximum itself
        3 * one + one // 7,
    ]
    out = []
    for x in xs:
        n = x << F
        k = isqrt(n)
        c = k if k * k == n else k + 1
        # mpmath cross-check
        mp_k = int(mp.floor(mpsqrt(mpf(n))))
        assert mp_k == k, f"isqrt disagrees with mpmath for x_raw={x}, F={F}"
        assert c < (1 << (W - 1)), "sqrt ceil must fit storage"
        out.append((x, k, c))
    return out


def mul_div_entries(F, W):
    one = 1 << F
    big = (1 << (W - 2)) + 977
    vals = [3, -3, one // 3, -(one // 3), one + 1, -(one + 1), 5 * one + one // 2, -(7 * one) - 1, big, -big]
    mul, div = [], []
    for a in vals:
        for b in vals:
            p = a * b
            f, c = floor_div(p, one), ceil_div(p, one)
            if -(1 << (W - 1)) <= f and c <= (1 << (W - 1)) - 1:
                mul.append((a, b, f, c))
            if b != 0:
                n = a * one
                f, c = floor_div(n, b), ceil_div(n, b)
                if -(1 << (W - 1)) <= f and c <= (1 << (W - 1)) - 1:
                    div.append((a, b, f, c))
    return mul, div


def pivot_entries(F):
    rng = Rng(0x1D7)
    out = []
    for n in (23, 50):
        m = dyadic_spd(rng, n)
        d = exact_cholesky_pivots(m)
        last = d[-1]
        mp_last = mp_cholesky_last_pivot(m)
        exact_mp = mpf(last.numerator) / mpf(last.denominator)
        assert abs(mp_last - exact_mp) / exact_mp < mpf(10) ** -200, "mpmath pivot disagrees with the exact rational"
        scaled_num = last.numerator << F
        f = floor_div(scaled_num, last.denominator)
        c = ceil_div(scaled_num, last.denominator)
        out.append((n, f, c))
    return out


def predicate_entries(W, with_circle):
    S = 1 << (W - 6)      # scale
    T = 1 << (W - 3)      # translation
    sc = lambda p: [x * S + T for x in p]
    entries = {"ORIENT2D": [], "ORIENT3D": [], "INCIRCLE": [], "INSPHERE": []}

    tri = [([0, 0], [4, 0], [0, 4]), ([0, 0], [0, 4], [4, 0]), ([0, 0], [4, 4], [8, 8]), ([-3, 7], [-3, 7], [5, 5])]
    for a, b, c in tri:
        A, B, C = sc(a), sc(b), sc(c)
        entries["ORIENT2D"].append(([A, B, C], orient2d(A, B, C)))
    # one unit off a collinear line at full scale, both sides
    A, B, C = sc([0, 0]), sc([4, 4]), sc([8, 8])
    entries["ORIENT2D"].append(([A, B, [C[0], C[1] + 1]], orient2d(A, B, [C[0], C[1] + 1])))
    entries["ORIENT2D"].append(([A, B, [C[0], C[1] - 1]], orient2d(A, B, [C[0], C[1] - 1])))

    tets = [([0, 0, 0], [4, 0, 0], [0, 4, 0], [0, 0, 4]), ([0, 0, 0], [0, 4, 0], [4, 0, 0], [0, 0, 4]),
            ([0, 0, 0], [4, 0, 0], [0, 4, 0], [8, 12, 0]), ([1, 2, 3], [2, 4, 6], [3, 6, 9], [4, 8, 12])]
    for a, b, c, d in tets:
        P = [sc(p) for p in (a, b, c, d)]
        entries["ORIENT3D"].append((P, orient3d(*P)))
    base = [sc(p) for p in ([0, 0, 0], [4, 0, 0], [0, 4, 0], [8, 12, 0])]
    for delta in (1, -1):
        P = [list(p) for p in base]  # fresh copy per entry: the lists are stored
        P[3][2] += delta
        entries["ORIENT3D"].append((P, orient3d(*P)))

    if with_circle:
        circ = [([0, 0], [4, 0], [4, 4], [2, 2]), ([0, 0], [4, 0], [4, 4], [8, 8]), ([0, 0], [4, 0], [4, 4], [0, 4]),
                ([0, 0], [4, 4], [4, 0], [2, 2])]
        for a, b, c, d in circ:
            P = [sc(p) for p in (a, b, c, d)]
            entries["INCIRCLE"].append((P, incircle(*P)))
        base = [sc(p) for p in ([0, 0], [4, 0], [4, 4], [0, 4])]
        for delta in (1, -1):
            P = [list(p) for p in base]
            P[3][0] += delta
            entries["INCIRCLE"].append((P, incircle(*P)))

        sph = [([0, 0, 0], [0, 4, 0], [4, 0, 0], [0, 0, 4], [1, 1, 1]), ([0, 0, 0], [0, 4, 0], [4, 0, 0], [0, 0, 4], [8, 8, 8]),
               ([0, 0, 0], [0, 4, 0], [4, 0, 0], [0, 0, 4], [4, 4, 0]), ([0, 0, 0], [4, 0, 0], [0, 4, 0], [0, 0, 4], [1, 1, 1])]
        for a, b, c, d, e in sph:
            P = [sc(p) for p in (a, b, c, d, e)]
            entries["INSPHERE"].append((P, insphere(*P)))
        base = [sc(p) for p in ([0, 0, 0], [0, 4, 0], [4, 0, 0], [0, 0, 4], [4, 4, 0])]
        for delta in (-1, 1):
            P = [list(p) for p in base]
            P[4][0] += delta
            entries["INSPHERE"].append((P, insphere(*P)))

    # mpmath cross-check of each sign through the plain determinant. The
    # determinant is an exact integer; mpmath at 300 digits reproduces it
    # exactly while it has fewer than ~990 bits, which every profile's
    # configurations satisfy (insphere at W = 512 reaches about 2^2600 and is
    # not generated there). A failure here means the two determinant
    # expansions disagree, and generation must stop.
    def check(key, rows_of):
        for idx, (pts, s) in enumerate(entries[key]):
            rows = rows_of(pts)
            got = sign(int(mp.nint(mp_det(rows))))
            assert got == s, f"{key}[{idx}] at W={W}: expansion says {s}, mpmath determinant says {got}"
    check("ORIENT2D", lambda pts: [[pts[0][0] - pts[2][0], pts[0][1] - pts[2][1]], [pts[1][0] - pts[2][0], pts[1][1] - pts[2][1]]])
    check("ORIENT3D", lambda pts: [[p[i] - pts[3][i] for i in range(3)] for p in pts[:3]])
    check("INCIRCLE", lambda pts: [[p[0] - pts[3][0], p[1] - pts[3][1], (p[0] - pts[3][0]) ** 2 + (p[1] - pts[3][1]) ** 2] for p in pts[:3]])
    check("INSPHERE", lambda pts: [[p[0] - pts[4][0], p[1] - pts[4][1], p[2] - pts[4][2], sum((p[i] - pts[4][i]) ** 2 for i in range(3))] for p in pts[:4]])
    return entries


def qf_entries(F, W):
    """Fused quadratic form references: (n, v raws, m raws row-major, floor,
    ceil, nearest) at the storage scale. The reference is computed on VALUES
    with Fractions (v_i m_ij v_j as rationals, then floor/ceil/nearest of
    q * 2^F), not on raws with integer shifts, so it shares no formulation
    with the Rust kernel; mpmath at 700 digits (3W bits fit) cross-checks it.
    """
    one = 1 << F
    rng = Rng(0x0F0_12A)

    def signed(bits):
        mag = rng.next() & ((1 << bits) - 1)
        return -mag if rng.next() & 1 else mag

    cases = [
        # constructed ties: exact value = half an ulp (and its negative twin)
        ([one // 2], [[2]]),
        ([one // 2], [[-2]]),
        ([one // 2, one // 2], [[0, 2], [2, 0]]),      # representable: point
        ([one // 2, one // 2], [[0, 2], [0, 0]]),      # one-sided: the tie
        # dyadic 3 x 3
        ([one // 2, -3 * one // 4, 5 * one // 8],
         [[one, one // 4, -one // 8], [one // 4, 2 * one, one // 16], [-one // 8, one // 16, one // 2]]),
    ]
    # random raws at a quarter of the storage width, dims 2..7
    for n in (2, 3, 5, 7):
        for _ in range(3):
            v = [signed(W // 4) for _ in range(n)]
            m = [[signed(W // 4) for _ in range(n)] for _ in range(n)]
            cases.append((v, m))
    # operands near the storage maximum with a result that still fits:
    # raws of 2^k with k = (W - 1 + 2F) // 3 - 2, n = 2, four terms
    k = (W - 1 + 2 * F) // 3 - 2
    big = (1 << k) - 977
    for sv, sm in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        v = [sv * big, sv * (big - 12345)]
        m = [[sm * big, sm * (big // 3)], [sm * (big // 5), sm * (big - 1)]]
        cases.append((v, m))

    out = []
    for v, m in cases:
        n = len(v)
        q = sum(Fraction(v[i], one) * Fraction(m[i][j], one) * Fraction(v[j], one)
                for i in range(n) for j in range(n))
        scaled = q * one
        f = scaled.numerator // scaled.denominator
        c = f if scaled.denominator == 1 else f + 1
        half = scaled + Fraction(1, 2)
        near = half.numerator // half.denominator
        assert f <= near <= c
        assert -(1 << (W - 1)) <= f and c <= (1 << (W - 1)) - 1, "qf reference must fit storage"
        with mp.workdps(700):
            qm = mpf(0)
            for i in range(n):
                for j in range(n):
                    qm += mpf(v[i]) * mpf(m[i][j]) * mpf(v[j])
            qm = qm / mpf(one) / mpf(one)
            assert int(mp.floor(qm)) == f, "mpmath disagrees with the Fraction floor"
            assert int(mp.ceil(qm)) == c, "mpmath disagrees with the Fraction ceil"
        out.append((n, v, [x for row in m for x in row], f, c, near))
    return out


def decimal_entries():
    out = {"DSQRT": [], "DMUL": [], "DDIV": []}
    for D in (2, 9):
        scale = 10 ** D
        xs = [0, 1, 2, 3, scale, 2 * scale, 4 * scale, scale // 4, 10 ** 25 * scale if D == 9 else 10 ** 25, (1 << 100), (1 << 126) + 7]
        for x in xs:
            if x >= (1 << 127):
                continue
            n = x * scale
            k = isqrt(n)
            c = k if k * k == n else k + 1
            assert int(mp.floor(mpsqrt(mpf(n)))) == k
            out["DSQRT"].append((D, x, k, c))
        vals = [3, -3, scale // 3, -(scale // 3), scale + 1, -(scale + 1), 10 ** 30, -(10 ** 30), 5 * scale + scale // 2]
        for a in vals:
            for b in vals:
                p = a * b
                f, c = floor_div(p, scale), ceil_div(p, scale)
                if -(1 << 127) <= f and c < (1 << 127):
                    out["DMUL"].append((D, a, b, f, c))
                if b != 0:
                    n = a * scale
                    f, c = floor_div(n, b), ceil_div(n, b)
                    if -(1 << 127) <= f and c < (1 << 127):
                        out["DDIV"].append((D, a, b, f, c))
    return out


# ---------------------------------------------------------------------------
# emit
# ---------------------------------------------------------------------------

def main():
    lines = []
    w = lines.append
    w("// GENERATED by scripts/generate_certified_geometry_refs.py. Do not edit.")
    w("// Independent references: Python exact integers and fractions, cross-checked")
    w("// against mpmath at 300 digits inside the generator. Raw values are exact")
    w("// little-endian two's-complement bytes at the profile's storage width.")
    w("")
    for name, F, W in PROFILES:
        w(f'#[cfg(table_format = "{name}")]')
        w("pub mod refs {")
        w(f"    pub const FRAC_BITS: u32 = {F};")
        w(f"    pub const STORAGE_BITS: u32 = {W};")
        w("    /// (x_raw, floor_raw, ceil_raw) of sqrt at the storage scale.")
        w("    pub const SQRT: &[(&[u8], &[u8], &[u8])] = &[")
        for x, k, c in sqrt_entries(F, W):
            w(f"        ({le_bytes(x, W)}, {le_bytes(k, W)}, {le_bytes(c, W)}),")
        w("    ];")
        mul, div = mul_div_entries(F, W)
        w("    /// (a_raw, b_raw, floor_raw, ceil_raw) of a * b at the storage scale.")
        w("    pub const MUL: &[(&[u8], &[u8], &[u8], &[u8])] = &[")
        for a, b, f, c in mul:
            w(f"        ({le_bytes(a, W)}, {le_bytes(b, W)}, {le_bytes(f, W)}, {le_bytes(c, W)}),")
        w("    ];")
        w("    /// (a_raw, b_raw, floor_raw, ceil_raw) of a / b at the storage scale.")
        w("    pub const DIV: &[(&[u8], &[u8], &[u8], &[u8])] = &[")
        for a, b, f, c in div:
            w(f"        ({le_bytes(a, W)}, {le_bytes(b, W)}, {le_bytes(f, W)}, {le_bytes(c, W)}),")
        w("    ];")
        w("    /// (n, floor_raw, ceil_raw) of the exact rational last Cholesky pivot of the")
        w("    /// dyadic A^T A + I matrices from tests/pd_verdict_validation.rs (seed 0x1D7).")
        w("    pub const PIVOT: &[(usize, &[u8], &[u8])] = &[")
        for n, f, c in pivot_entries(F):
            w(f"        ({n}, {le_bytes(f, W)}, {le_bytes(c, W)}),")
        w("    ];")
        w("    /// (n, v raws, m raws row-major, floor_raw, ceil_raw, nearest_raw) of v^T M v.")
        w("    pub const QF: &[(usize, &[&[u8]], &[&[u8]], &[u8], &[u8], &[u8])] = &[")
        for n, v, m, f, c, near in qf_entries(F, W):
            vs = ", ".join(le_bytes(x, W) for x in v)
            ms = ", ".join(le_bytes(x, W) for x in m)
            w(f"        ({n}, &[{vs}], &[{ms}], {le_bytes(f, W)}, {le_bytes(c, W)}, {le_bytes(near, W)}),")
        w("    ];")
        ents = predicate_entries(W, with_circle=(name != "q256_256"))
        for key, arity, dim in (("ORIENT2D", 3, 2), ("ORIENT3D", 4, 3), ("INCIRCLE", 4, 2), ("INSPHERE", 5, 3)):
            if name == "q256_256" and key in ("INCIRCLE", "INSPHERE"):
                continue
            w(f"    /// ({arity} points of {dim} raw coordinates, expected sign).")
            w(f"    pub const {key}: &[(&[&[&[u8]]], i8)] = &[")
            for pts, s in ents[key]:
                pts_s = ", ".join("&[" + ", ".join(le_bytes(x, W) for x in p) + "]" for p in pts)
                w(f"        (&[{pts_s}], {s}),")
            w("    ];")
        w("}")
        w("")
    dec = decimal_entries()
    w("/// Decimal references, profile-independent (DecimalFixed is i128 everywhere).")
    w("pub mod decimal_refs {")
    w("    /// (decimals, x_raw, floor_raw, ceil_raw) of sqrt at 10^decimals.")
    w("    pub const DSQRT: &[(u8, i128, i128, i128)] = &[")
    for D, x, k, c in dec["DSQRT"]:
        w(f"        ({D}, {x}, {k}, {c}),")
    w("    ];")
    w("    /// (decimals, a_raw, b_raw, floor_raw, ceil_raw) of a * b.")
    w("    pub const DMUL: &[(u8, i128, i128, i128, i128)] = &[")
    for D, a, b, f, c in dec["DMUL"]:
        w(f"        ({D}, {a}, {b}, {f}, {c}),")
    w("    ];")
    w("    /// (decimals, a_raw, b_raw, floor_raw, ceil_raw) of a / b.")
    w("    pub const DDIV: &[(u8, i128, i128, i128, i128)] = &[")
    for D, a, b, f, c in dec["DDIV"]:
        w(f"        ({D}, {a}, {b}, {f}, {c}),")
    w("    ];")
    w("}")
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"wrote {OUT}: {len(lines)} lines", file=sys.stderr)


if __name__ == "__main__":
    main()
