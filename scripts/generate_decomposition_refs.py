#!/usr/bin/env python3
"""Generate tests/data/decomposition_refs.rs: independent references for the
iterative-decomposition gate (tests/decomposition_convergence_validation.rs).

Every matrix has integer entries, or integers over a power of two, so it is
stored exactly on every profile and its references do not depend on the
profile. Each reference is computed with mpmath at 120 digits and cross-checked
against the roots of the exact characteristic polynomial (Faddeev-LeVerrier on
Python fractions, roots by mpmath.polyroots), code that shares nothing with the
Rust side. Values are emitted as decimal strings with 90 fractional digits; the
gate rounds them to the profile with at most one ulp of error.

  SVD    singular values, descending (sqrt of the eigenvalues of A^T A)
  EIGEN  eigenvalues of symmetric matrices, ascending
  SCHUR  eigenvalues as (re, im), sorted by (re, im)

The cases are the inputs that exposed silent failures in g_math 0.6.1: exactly
rank-deficient matrices, bidiagonal matrices with an interior zero on the
diagonal, 2x2 blocks with real eigenvalues, permutation matrices, and entries
large enough that their squares leave the storage range.

Run from the repository root: python3 scripts/generate_decomposition_refs.py
"""
import random
from fractions import Fraction

from mpmath import mp, matrix, svd_r, eigsy, eig, polyroots, mpf, mpc, nint

DPS = 120
DIGITS = 90
CROSS_CHECK = mpf(10) ** -40  # multiple roots converge to about half the digits
OUT = "tests/data/decomposition_refs.rs"


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


def poly_roots(coeffs):
    zeros = 0
    while coeffs and coeffs[-1] == 0:
        coeffs = coeffs[:-1]
        zeros += 1
    roots = [mpc(0)] * zeros
    if len(coeffs) > 1:
        found = polyroots([mpf(c.numerator) / c.denominator for c in coeffs],
                          maxsteps=2000, extraprec=4 * DPS)
        roots += [mpc(r) for r in found]
    return roots


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
    ata = a if len(a) < len(a[0]) else a  # A^T A for m >= n, A A^T for m < n
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


def main():
    mp.dps = DPS
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

    def case_literal(name, case):
        rows, cols, entries, shift = case
        body = ", ".join(str(v) for v in entries)
        return (f'Case {{ name: "{name}", rows: {rows}, cols: {cols}, '
                f'entries: &[{body}], shift: {shift} }}')

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
        "/// Singular values, descending.",
        "pub const SVD: &[(Case, &[&str])] = &[",
    ]
    for name, case in svd_cases:
        refs = ", ".join(f'"{s}"' for s in svd_refs(case))
        out.append(f"    ({case_literal(name, case)}, &[{refs}]),")
    out += ["];", "", "/// Eigenvalues of symmetric matrices, ascending.", "pub const EIGEN: &[(Case, &[&str])] = &["]
    for name, case in eigen_cases:
        refs = ", ".join(f'"{s}"' for s in eigen_refs(case))
        out.append(f"    ({case_literal(name, case)}, &[{refs}]),")
    out += ["];", "", "/// Eigenvalues as (re, im), sorted by (re, im).", "pub const SCHUR: &[(Case, &[(&str, &str)])] = &["]
    for name, case in schur_cases:
        refs = ", ".join(f'("{re}", "{im}")' for re, im in schur_refs(case))
        out.append(f"    ({case_literal(name, case)}, &[{refs}]),")
    out += ["];", ""]
    with open(OUT, "w") as f:
        f.write("\n".join(out))
    print(f"wrote {OUT}: {len(svd_cases)} SVD, {len(eigen_cases)} EIGEN, {len(schur_cases)} SCHUR cases")


if __name__ == "__main__":
    main()
