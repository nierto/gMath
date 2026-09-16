#!/usr/bin/env python3
"""Generate independent references for the iterative-decomposition gate
(tests/decomposition_convergence_validation.rs).

Two corpora, one format:

  fixed   tests/data/decomposition_refs.rs: named cases, the inputs that
          exposed silent failures in g_math 0.6.1.
  random  tests/data/decomposition_random_refs.rs: a seeded draw of matrices
          from the same failure classes, CASES per decomposition. CI runs the
          committed seed on every push and a fresh seed on a schedule.

Every matrix has integer entries, or integers over a power of two, so it is
stored exactly on every profile and its references do not depend on the
profile. Each reference is computed with mpmath at 120 digits and cross-checked
against the roots of the exact characteristic polynomial (Faddeev-LeVerrier on
Python fractions, roots by mpmath.polyroots), code that shares nothing with the
Rust side. No floating point is used anywhere. Values are emitted as decimal
strings with 90 fractional digits; the gate rounds them to the profile with at
most one ulp of error.

  SVD    singular values, descending (sqrt of the eigenvalues of A^T A)
  EIGEN  eigenvalues of symmetric matrices, ascending
  SCHUR  eigenvalues as (re, im), sorted by (re, im)

Random cases are seeded per (seed, kind, index), so the draw does not depend on
the number of worker processes.

Run from the repository root:
  python3 scripts/generate_decomposition_refs.py
  python3 scripts/generate_decomposition_refs.py --random --seed 20260916 --cases 64
"""
import argparse
import random
from fractions import Fraction
from multiprocessing import Pool

from mpmath import mp, matrix, svd_r, eigsy, eig, polyroots, mpf, mpc, nint

DPS = 120
DIGITS = 90
CROSS_CHECK = mpf(10) ** -40  # multiple roots converge to about half the digits
OUT = "tests/data/decomposition_refs.rs"
RANDOM_OUT = "tests/data/decomposition_random_refs.rs"

mp.dps = DPS


def dec(x):
    q = int(nint(abs(x) * 10 ** DIGITS))
    ip, fp = divmod(q, 10 ** DIGITS)
    sign = "-" if (x < 0 and q > 0) else ""
    return f"{sign}{ip}.{fp:0{DIGITS}d}"


def charpoly(a):
    """Exact characteristic polynomial coefficients, highest degree first."""
    n = len(a)
    ident = [[Fraction(int(i == j)) for j in range(n)] for i in range(n)]
    m = [[Fraction(0)] * n for _ in range(n)]
    coeffs = [Fraction(1)]
    for k in range(1, n + 1):
        am = [[sum(a[i][t] * m[t][j] for t in range(n)) for j in range(n)] for i in range(n)]
        m = [[am[i][j] + coeffs[-1] * ident[i][j] for j in range(n)] for i in range(n)]
        am = [[sum(a[i][t] * m[t][j] for t in range(n)) for j in range(n)] for i in range(n)]
        coeffs.append(-sum(am[i][i] for i in range(n)) / k)
    return coeffs


def p_trim(p):
    i = 0
    while i < len(p) - 1 and p[i] == 0:
        i += 1
    return p[i:]


def p_is_zero(p):
    return all(c == 0 for c in p)


def p_deriv(p):
    n = len(p) - 1
    return p_trim([c * (n - i) for i, c in enumerate(p[:-1])]) or [Fraction(0)]


def p_sub(a, b):
    n = max(len(a), len(b))
    a = [Fraction(0)] * (n - len(a)) + a
    b = [Fraction(0)] * (n - len(b)) + b
    return p_trim([x - y for x, y in zip(a, b)])


def p_divmod(a, b):
    a, b = list(p_trim(a)), p_trim(b)
    if len(a) < len(b):
        return [Fraction(0)], a
    q = [Fraction(0)] * (len(a) - len(b) + 1)
    for i in range(len(q)):
        q[i] = a[i] / b[0]
        for j in range(len(b)):
            a[i + j] -= q[i] * b[j]
    rest = a[len(q):]
    return q, (p_trim(rest) if rest else [Fraction(0)])


def p_gcd(a, b):
    a, b = p_trim(a), p_trim(b)
    while not p_is_zero(b):
        a, b = b, p_divmod(a, b)[1]
    return [c / a[0] for c in a]


def squarefree_decomposition(f):
    """Yun's algorithm over the rationals: [(a_i, i)] with f = prod a_i^i, each
    a_i square-free, so every root of a_i is simple and has multiplicity i."""
    f = [c / p_trim(f)[0] for c in p_trim(f)]
    fp = p_deriv(f)
    g = p_gcd(f, fp)
    b = p_divmod(f, g)[0]
    d = p_sub(p_divmod(fp, g)[0], p_deriv(b))
    out, i = [], 1
    while len(p_trim(b)) > 1:
        a = p_gcd(b, d)
        if len(a) > 1:
            out.append((a, i))
        b = p_divmod(b, a)[0]
        d = p_sub(p_divmod(d, a)[0], p_deriv(b))
        i += 1
    return out


def poly_roots(coeffs):
    """Roots with multiplicity. Numeric root finding only ever sees square-free
    factors (simple roots converge); linear factors give their root exactly."""
    roots = []
    for factor, mult in squarefree_decomposition(coeffs):
        if len(factor) == 2:
            r = -factor[1] / factor[0]
            found = [mpc(mpf(r.numerator) / r.denominator)]
        else:
            found = [mpc(x) for x in polyroots([mpf(c.numerator) / c.denominator for c in factor],
                                                maxsteps=2000, extraprec=4 * DPS)]
        roots += found * mult
    return roots


def squarefree(coeffs):
    return all(mult == 1 for _, mult in squarefree_decomposition(coeffs))


def matched(values, roots):
    """Every value has a distinct root within CROSS_CHECK."""
    pool = list(roots)
    for v in values:
        best = min(range(len(pool)), key=lambda i: abs(pool[i] - v))
        if abs(pool[best] - v) > CROSS_CHECK:
            return False
        pool.pop(best)
    return not pool


def stored(case):
    rows, cols, entries, shift = case
    return [[Fraction(entries[i * cols + j], 2 ** shift) for j in range(cols)] for i in range(rows)]


def to_mp(a):
    m = matrix(len(a), len(a[0]))
    for i, row in enumerate(a):
        for j, v in enumerate(row):
            m[i, j] = mpf(v.numerator) / v.denominator
    return m


def svd_refs(case):
    a = stored(case)
    s = svd_r(to_mp(a), compute_uv=False)
    sig = sorted([s[i] for i in range(min(len(a), len(a[0])))], reverse=True)
    rows, cols = len(a), len(a[0])
    if rows >= cols:
        gram = [[sum(a[k][i] * a[k][j] for k in range(rows)) for j in range(cols)] for i in range(cols)]
    else:
        gram = [[sum(a[i][k] * a[j][k] for k in range(cols)) for j in range(rows)] for i in range(rows)]
    lam = poly_roots(charpoly(gram))
    assert matched([mpc(v * v) for v in sig], lam), "SVD cross-check failed"
    return [dec(v) for v in sig]


def eigen_refs(case):
    a = stored(case)
    e, _ = eigsy(to_mp(a))
    vals = sorted([e[i] for i in range(len(a))])
    assert matched([mpc(v) for v in vals], poly_roots(charpoly(a))), "EIGEN cross-check failed"
    return [dec(v) for v in vals]


def is_normal(a):
    at = [list(col) for col in zip(*a)]
    return mul(a, at) == mul(at, a)


def schur_refs_random(case):
    """As `schur_refs`, but a non-normal matrix with a repeated eigenvalue gets no
    spectrum references: it can be defective, where any finite precision resolves
    the eigenvalue only to about the square root of its rounding. Its structure,
    reconstruction and orthogonality are still checked by the gate."""
    a = stored(case)
    if not is_normal(a) and not squarefree(charpoly(a)):
        return []
    return schur_refs(case)


def schur_refs(case):
    a = stored(case)
    ev = eig(to_mp(a), left=False, right=False)
    vals = [mpc(v) for v in ev]
    assert matched(vals, poly_roots(charpoly(a))), "SCHUR cross-check failed"
    # Sort on the emitted (rounded) values: a conjugate pair's real parts can
    # differ in the far digits, which would otherwise order +im before -im.
    scale = 10 ** DIGITS
    rounded = sorted((int(nint(v.real * scale)), int(nint(v.imag * scale))) for v in vals)
    return [(dec(mpf(re) / scale), dec(mpf(im) / scale)) for re, im in rounded]


def square(rows):
    return (len(rows), len(rows[0]), [v for row in rows for v in row], 0)


def rect(rows, shift=0):
    return (len(rows), len(rows[0]), [v for row in rows for v in row], shift)


def case_literal(name, case):
    rows, cols, entries, shift = case
    body = ", ".join(str(v) for v in entries)
    return (f'Case {{ name: "{name}", rows: {rows}, cols: {cols}, '
            f'entries: &[{body}], shift: {shift} }}')


def emit_tables(out, svd, eigen, schur):
    """svd/eigen/schur: lists of (name, case, refs)."""
    out += ["/// Singular values, descending.", "pub const SVD: &[(Case, &[&str])] = &["]
    for name, case, refs in svd:
        out.append(f"    ({case_literal(name, case)}, &[{', '.join(f'{chr(34)}{s}{chr(34)}' for s in refs)}]),")
    out += ["];", "", "/// Eigenvalues of symmetric matrices, ascending.", "pub const EIGEN: &[(Case, &[&str])] = &["]
    for name, case, refs in eigen:
        out.append(f"    ({case_literal(name, case)}, &[{', '.join(f'{chr(34)}{s}{chr(34)}' for s in refs)}]),")
    out += ["];", "", "/// Eigenvalues as (re, im), sorted by (re, im).", "pub const SCHUR: &[(Case, &[(&str, &str)])] = &["]
    for name, case, refs in schur:
        pairs = ", ".join(f'("{re}", "{im}")' for re, im in refs)
        out.append(f"    ({case_literal(name, case)}, &[{pairs}]),")
    out += ["];", ""]


# ============================================================================
# Fixed corpus
# ============================================================================

def fixed_main():
    rng = random.Random(20260915)
    r8 = [[((i * 3 + j * 7 + 5) % 20) - 10 for j in range(8)] for i in range(8)]
    r16 = [[((i * 3 + j * 7 + 5) % 20) - 10 for j in range(16)] for i in range(16)]
    rnd8 = [[[rng.randint(-10, 9) for _ in range(8)] for _ in range(8)] for _ in range(3)]
    fa = [[rng.randint(-3, 3) for _ in range(3)] for _ in range(8)]
    fb = [[rng.randint(-3, 3) for _ in range(8)] for _ in range(3)]
    lr8 = [[sum(fa[i][k] * fb[k][j] for k in range(3)) for j in range(8)] for i in range(8)]
    fa = [[rng.randint(-3, 3) for _ in range(5)] for _ in range(12)]
    fb = [[rng.randint(-3, 3) for _ in range(12)] for _ in range(5)]
    lr12 = [[sum(fa[i][k] * fb[k][j] for k in range(5)) for j in range(12)] for i in range(12)]
    tall = [[rng.randint(-10, 9) for _ in range(4)] for _ in range(6)]
    wide = [[rng.randint(-10, 9) for _ in range(6)] for _ in range(4)]
    sym8 = [[0] * 8 for _ in range(8)]
    for i in range(8):
        for j in range(i, 8):
            sym8[i][j] = sym8[j][i] = rng.randint(-10, 9)
    nonsym6 = [[rng.randint(-5, 5) for _ in range(6)] for _ in range(6)]

    svd_cases = [
        ("rank6_8x8_consumer_report", square(r8)),
        ("rank6_16x16", square(r16)),
        ("rank3_8x8", square(lr8)),
        ("rank5_12x12", square(lr12)),
        ("full_3x3", square([[1, 2, 3], [4, 5, 6], [7, 8, 10]])),
        ("rank1_3x3", square([[1, 2, 3], [2, 4, 6], [3, 6, 9]])),
        ("near_singular_2x2", rect([[1024, 1023], [1023, 1024]], 10)),
        ("random_8x8_a", square(rnd8[0])),
        ("random_8x8_b", square(rnd8[1])),
        ("random_8x8_c", square(rnd8[2])),
        ("tall_6x4", rect(tall)),
        ("wide_4x6", rect(wide)),
        ("bidiagonal_3x3_interior_zero", square([[3, 1, 0], [0, 0, 1], [0, 0, 2]])),
        ("bidiagonal_4x4_interior_zero", square([[4, 1, 0, 0], [0, 0, 2, 0], [0, 0, 3, 1], [0, 0, 0, 2]])),
        ("bidiagonal_5x5_interior_zero", square([[2, 1, 0, 0, 0], [0, 5, 1, 0, 0], [0, 0, 0, 2, 0], [0, 0, 0, 3, 1], [0, 0, 0, 0, 1]])),
        ("rank6_8x8_over_64", rect(r8, 6)),
        ("rank6_8x8_times_60", square([[60 * v for v in row] for row in r8])),
    ]
    eigen_cases = [
        ("spd_3x3", square([[4, 1, 2], [1, 3, 1], [2, 1, 5]])),
        ("swap_2x2", square([[0, 1], [1, 0]])),
        ("zero_diagonal_4x4", square([[0, 2, 0, 1], [2, 0, 3, 0], [0, 3, 0, 4], [1, 0, 4, 0]])),
        ("rank1_ones_4x4", square([[1] * 4 for _ in range(4)])),
        ("repeated_4x4", square([[2, 1, 0, 0], [1, 2, 0, 0], [0, 0, 2, 1], [0, 0, 1, 2]])),
        ("symmetrised_rank6_8x8", square([[r8[i][j] + r8[j][i] for j in range(8)] for i in range(8)])),
        ("gram_rank3_8x8", square([[sum(lr8[i][k] * lr8[j][k] for k in range(8)) for j in range(8)] for i in range(8)])),
        ("random_symmetric_8x8", square(sym8)),
    ]
    schur_cases = [
        ("symmetric_3x3", square([[4, 1, 2], [1, 3, 1], [2, 1, 5]])),
        ("real_pair_2x2", square([[3, 1], [1, 2]])),
        ("full_3x3", square([[1, 2, 3], [4, 5, 6], [7, 8, 10]])),
        ("cyclic_3x3", square([[0, 0, 1], [1, 0, 0], [0, 1, 0]])),
        ("cyclic_4x4", square([[0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])),
        ("rotation_plus_2", square([[0, -1, 0], [1, 0, 0], [0, 0, 2]])),
        ("jordan_2x2", square([[2, 1], [0, 2]])),
        ("upper_3x3", square([[1, 2, 3], [0, 4, 5], [0, 0, 6]])),
        ("companion_4x4", square([[10, -35, 50, -24], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])),
        ("rank6_8x8_consumer_report", square(r8)),
        ("rank3_8x8", square(lr8)),
        ("random_6x6", square(nonsym6)),
    ]

    out = [
        "// @generated by scripts/generate_decomposition_refs.py: do not edit by hand.",
        f"// mpmath {mp.dps} digits, cross-checked against exact characteristic-polynomial roots.",
        "",
        "pub struct Case {",
        "    pub name: &'static str,",
        "    pub rows: usize,",
        "    pub cols: usize,",
        "    /// Row-major entries; the stored value is `entry / 2^shift`.",
        "    pub entries: &'static [i32],",
        "    pub shift: u32,",
        "}",
        "",
    ]
    emit_tables(
        out,
        [(name, case, svd_refs(case)) for name, case in svd_cases],
        [(name, case, eigen_refs(case)) for name, case in eigen_cases],
        [(name, case, schur_refs(case)) for name, case in schur_cases],
    )
    with open(OUT, "w") as f:
        f.write("\n".join(out))
    print(f"wrote {OUT}: {len(svd_cases)} SVD, {len(eigen_cases)} EIGEN, {len(schur_cases)} SCHUR cases")


# ============================================================================
# Random corpus
# ============================================================================
#
# Value ranges keep every case inside the narrowest profile (Q16.16, range
# +-32768): singular values up to about 6000, eigenvalues up to about 30000.

def ints(rng, rows, cols, lo, hi):
    return [[rng.randint(lo, hi) for _ in range(cols)] for _ in range(rows)]


def mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def symmetric(rng, n, lo, hi, zero_diagonal=False):
    m = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            m[i][j] = m[j][i] = 0 if (zero_diagonal and i == j) else rng.randint(lo, hi)
    return m


def permuted(rng, m):
    """P^T M P for a random permutation P: an exact orthogonal similarity."""
    p = list(range(len(m)))
    rng.shuffle(p)
    return [[m[p[i]][p[j]] for j in range(len(m))] for i in range(len(m))]


def random_svd_case(rng):
    cls = rng.choice(["full", "rank_deficient", "near_rank_deficient", "bidiagonal_interior_zero",
                      "rectangular", "scaled", "dyadic"])
    if cls == "full":
        n = rng.randint(2, 12)
        return cls, rect(ints(rng, n, n, -10, 10))
    if cls in ("rank_deficient", "near_rank_deficient"):
        n = rng.randint(3, 12)
        r = rng.randint(1, n - 1)
        a = mul(ints(rng, n, r, -3, 3), ints(rng, r, n, -3, 3))
        if cls == "near_rank_deficient":
            a[rng.randrange(n)][rng.randrange(n)] += rng.choice((-1, 1))
        return cls, rect(a)
    if cls == "bidiagonal_interior_zero":
        n = rng.randint(3, 12)
        a = [[0] * n for _ in range(n)]
        for i in range(n):
            a[i][i] = rng.choice((-1, 1)) * rng.randint(1, 9)
            if i + 1 < n:
                a[i][i + 1] = rng.choice((-1, 1)) * rng.randint(1, 5)
        for i in rng.sample(range(1, n - 1), rng.randint(1, max(1, (n - 2) // 2))):
            a[i][i] = 0
        return cls, rect(a)
    if cls == "rectangular":
        m = rng.randint(2, 10)
        n = rng.choice([k for k in range(2, 11) if k != m])
        return cls, rect(ints(rng, m, n, -10, 10))
    if cls == "scaled":
        n = rng.randint(2, 10)
        return cls, rect([[60 * v for v in row] for row in ints(rng, n, n, -10, 10)])
    n = rng.randint(2, 10)
    return cls, rect(ints(rng, n, n, -10, 10), 6)


def random_eigen_case(rng):
    cls = rng.choice(["symmetric", "gram_low_rank", "zero_diagonal", "repeated", "large_pattern", "scaled",
                      "dyadic"])
    if cls == "symmetric":
        n = rng.randint(2, 12)
        return cls, rect(symmetric(rng, n, -10, 10))
    if cls == "gram_low_rank":
        n = rng.randint(2, 12)
        r = rng.randint(1, max(1, n - 1))
        f = ints(rng, n, r, -3, 3)
        return cls, rect(mul(f, [list(col) for col in zip(*f)]))
    if cls == "zero_diagonal":
        n = rng.randint(2, 12)
        return cls, rect(symmetric(rng, n, -10, 10, zero_diagonal=True))
    if cls == "repeated":
        k = rng.randint(1, 5)
        a, b = rng.randint(-6, 6), rng.randint(1, 4)
        m = [[0] * (2 * k) for _ in range(2 * k)]
        for t in range(k):
            m[2 * t][2 * t] = m[2 * t + 1][2 * t + 1] = a
            m[2 * t][2 * t + 1] = m[2 * t + 1][2 * t] = b
        return cls, rect(permuted(rng, m))
    if cls == "large_pattern":
        # v (J - I): squares of the entries leave the Q16.16 range
        n = rng.randint(3, 12)
        v = rng.randint(300, 30000 // (n - 1))
        return cls, rect([[0 if i == j else v for j in range(n)] for i in range(n)])
    if cls == "scaled":
        n = rng.randint(2, 10)
        return cls, rect([[40 * x for x in row] for row in symmetric(rng, n, -10, 10)])
    # entries of at most 10/64: a few hundred quanta at Q22.10
    n = rng.randint(2, 12)
    return cls, rect(symmetric(rng, n, -10, 10), 6)


def random_schur_case(rng):
    cls = rng.choice(["full", "symmetric", "rank_deficient", "signed_permutation", "complex_blocks",
                      "companion", "scaled", "dyadic"])
    if cls == "full":
        n = rng.randint(2, 10)
        return cls, rect(ints(rng, n, n, -10, 10))
    if cls == "symmetric":
        n = rng.randint(2, 10)
        return cls, rect(symmetric(rng, n, -10, 10))
    if cls == "rank_deficient":
        n = rng.randint(3, 10)
        r = rng.randint(1, n - 1)
        return cls, rect(mul(ints(rng, n, r, -3, 3), ints(rng, r, n, -3, 3)))
    if cls == "signed_permutation":
        n = rng.randint(2, 10)
        p = list(range(n))
        rng.shuffle(p)
        return cls, rect([[rng.choice((-1, 1)) if j == p[i] else 0 for j in range(n)] for i in range(n)])
    if cls == "complex_blocks":
        # quasi-triangular with a-bi/a+bi blocks, hidden by a permutation similarity
        n = rng.randint(2, 10)
        a = [[0] * n for _ in range(n)]
        i = 0
        while i < n:
            if i + 1 < n and rng.randint(0, 1) == 1:
                re, im = rng.randint(-5, 5), rng.randint(1, 5)
                a[i][i] = a[i + 1][i + 1] = re
                a[i][i + 1], a[i + 1][i] = -im, im
                i += 2
            else:
                a[i][i] = rng.randint(-8, 8)
                i += 1
        for r in range(n):
            for c in range(r + 1, n):
                if a[r][c] == 0:
                    a[r][c] = rng.randint(-5, 5)
        return cls, rect(permuted(rng, a))
    if cls == "companion":
        k = rng.randint(2, 5)
        coeffs = [1]
        for root in rng.sample(range(-4, 5), k):
            coeffs = [c - root * p for c, p in zip(coeffs + [0], [0] + coeffs)]
        m = [[0] * k for _ in range(k)]
        m[0] = [-c for c in coeffs[1:]]
        for i in range(1, k):
            m[i][i - 1] = 1
        return cls, rect(m)
    if cls == "scaled":
        n = rng.randint(2, 8)
        return cls, rect([[20 * v for v in row] for row in ints(rng, n, n, -10, 10)])
    # entries of at most 10/64: a few hundred quanta at Q22.10
    n = rng.randint(2, 10)
    return cls, rect(ints(rng, n, n, -10, 10), 6)


GENERATORS = {
    "SVD": (random_svd_case, svd_refs),
    "EIGEN": (random_eigen_case, eigen_refs),
    "SCHUR": (random_schur_case, schur_refs_random),
}


def random_case(task):
    seed, kind, index = task
    rng = random.Random(f"{seed}/{kind}/{index}")
    make, refs = GENERATORS[kind]
    cls, case = make(rng)
    return kind, index, f"{cls}_{index}", case, refs(case)


def random_main(seed, cases, out_path, workers):
    tasks = [(seed, kind, i) for kind in GENERATORS for i in range(cases)]
    with Pool(workers) as pool:
        results = sorted(pool.map(random_case, tasks, chunksize=4), key=lambda r: (r[0], r[1]))
    by_kind = {kind: [(name, case, refs) for k, _, name, case, refs in results if k == kind] for kind in GENERATORS}
    out = [
        "// @generated by scripts/generate_decomposition_refs.py: do not edit by hand.",
        f"// Reproduce: python3 scripts/generate_decomposition_refs.py --random --seed {seed} --cases {cases}",
        f"// mpmath {mp.dps} digits, cross-checked against exact characteristic-polynomial roots.",
        "",
        "use super::data::Case;",
        "",
        "/// Seed of this draw.",
        f"pub const SEED: u64 = {seed};",
        "",
    ]
    emit_tables(out, by_kind["SVD"], by_kind["EIGEN"], by_kind["SCHUR"])
    with open(out_path, "w") as f:
        f.write("\n".join(out))
    print(f"wrote {out_path}: seed {seed}, {cases} SVD, {cases} EIGEN, {cases} SCHUR cases")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--random", action="store_true", help="write a seeded random corpus instead of the fixed one")
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--cases", type=int, default=64, help="cases per decomposition")
    parser.add_argument("--out", default=None)
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()
    if args.random:
        random_main(args.seed, args.cases, args.out or RANDOM_OUT, args.workers)
    else:
        fixed_main()
