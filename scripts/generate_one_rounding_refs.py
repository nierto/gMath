#!/usr/bin/env python3
"""mpmath references for the 0.6.4 one-rounding rework
(tests/data/one_rounding_refs.rs).

Operations whose intermediates moved from storage to the compute tier: the
vector dot product, Euclidean length and distance, the Frobenius norm, the
Minkowski norm of HyperbolicSpace (including near-cancelling vectors), the R
factor of qr_decompose (its Householder sign convention replicated exactly),
the full symmetrization of a rank-3 tensor, matrix_exp, and SO(3) exp.

Inputs are dyadic with at most 8 fraction bits and magnitude <= 8, so they are
stored exactly at every gated split (8 to 24 fraction bits) and on every wider
profile. References are the exact result (dyadic sums) or mpmath at 120
significant digits; the test parses each with the exact literal parser, which
rounds it to nearest at the build's split, so one table serves every build.

Usage: python3 scripts/generate_one_rounding_refs.py
"""
import random
from fractions import Fraction

from mpmath import mp, mpf, sqrt, matrix, expm, logm, sqrtm

mp.dps = 140
SEED = 20260925
DIGITS = 120


def dy(rng, lo=-8, hi=8, bits=8):
    """A dyadic k / 2^bits in [lo, hi)."""
    return Fraction(rng.randrange(lo << bits, hi << bits), 1 << bits)


def lit(v):
    """A literal for a Fraction (exact when dyadic) or an mpf (120 digits)."""
    if isinstance(v, Fraction):
        sign = "-" if v < 0 else ""
        v = abs(v)
        d = 0
        while (v * 10 ** d).denominator != 1:
            d += 1
            if d > 60:
                return lit(mpf(v.numerator) / v.denominator) if sign == "" else "-" + lit(mpf(v.numerator) / v.denominator)
        n = int(v * 10 ** d)
        if d == 0:
            return f"{sign}{n}"
        s = str(n).rjust(d + 1, "0")
        return f"{sign}{s[:-d]}.{s[-d:]}"
    return mp.nstr(v, DIGITS, strip_zeros=False, min_fixed=-10**9, max_fixed=10**9)


def m_of(fr):
    return mpf(fr.numerator) / fr.denominator


def vec_lit(v):
    return "&[" + ", ".join(f'"{lit(x)}"' for x in v) + "]"


def householder_qr(a):
    """Q and R of qr_decompose: R <- H_k R, Q <- Q H_k, same convention."""
    n = len(a)
    r = [[m_of(x) for x in row] for row in a]
    q = [[mpf(1) if i == j else mpf(0) for j in range(n)] for i in range(n)]
    for k in range(n):
        x = [r[i][k] for i in range(k, n)]
        nx = sqrt(sum(t * t for t in x))
        if nx == 0:
            continue
        alpha = nx if x[0] < 0 else -nx
        v = x[:]
        v[0] = x[0] - alpha
        vv = sum(t * t for t in v)
        if vv == 0:
            continue
        for j in range(k, n):
            col = [r[i][j] for i in range(k, n)]
            f = 2 * sum(v[i] * col[i] for i in range(len(v))) / vv
            for i in range(len(v)):
                r[k + i][j] = col[i] - f * v[i]
        for i in range(n):
            row = [q[i][j] for j in range(k, n)]
            f = 2 * sum(v[t] * row[t] for t in range(len(v))) / vv
            for t in range(len(v)):
                q[i][k + t] = row[t] - f * v[t]
    return q, r


def householder_r(a):
    """R of qr_decompose's Householder QR: alpha = -sign(x0) ||x|| with
    sign(0) taken as positive (x0 < 0 -> alpha = +||x||, else -||x||)."""
    n = len(a)
    r = [[m_of(x) for x in row] for row in a]
    for k in range(n):
        x = [r[i][k] for i in range(k, n)]
        nx = sqrt(sum(t * t for t in x))
        if nx == 0:
            continue
        alpha = nx if x[0] < 0 else -nx
        v = x[:]
        v[0] = x[0] - alpha
        vv = sum(t * t for t in v)
        if vv == 0:
            continue
        for j in range(k, n):
            col = [r[i][j] for i in range(k, n)]
            f = 2 * sum(v[i] * col[i] for i in range(len(v))) / vv
            for i in range(len(v)):
                r[k + i][j] = col[i] - f * v[i]
    return r


def doolittle(a):
    """Exact PA = LU with lu_decompose's pivoting: the row with the largest
    |candidate| wins, the first on a tie. None if a pivot is zero or two
    candidates are within 1/64 relative (a rounded candidate could flip it)."""
    n = len(a)
    pa = [row[:] for row in a]
    l = [[Fraction(0)] * n for _ in range(n)]
    u = [[Fraction(0)] * n for _ in range(n)]
    perm = list(range(n))
    for k in range(n):
        cand = [pa[i][k] - sum(l[i][m] * u[m][k] for m in range(k)) for i in range(k, n)]
        mags = sorted((abs(c) for c in cand), reverse=True)
        if mags[0] == 0 or (len(mags) > 1 and mags[0] - mags[1] < mags[0] / 64):
            return None
        best = max(range(len(cand)), key=lambda t: (abs(cand[t]), -t)) + k
        if best != k:
            pa[k], pa[best] = pa[best], pa[k]
            perm[k], perm[best] = perm[best], perm[k]
            for j in range(k):
                l[k][j], l[best][j] = l[best][j], l[k][j]
        for j in range(k, n):
            u[k][j] = pa[k][j] - sum(l[k][m] * u[m][j] for m in range(k))
        l[k][k] = Fraction(1)
        for i in range(k + 1, n):
            l[i][k] = (pa[i][k] - sum(l[i][m] * u[m][k] for m in range(k))) / u[k][k]
    return l, u, perm


def solve_exact(a, b):
    """x = A^-1 b by exact Gaussian elimination on Fractions."""
    n = len(a)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for k in range(n):
        p = next(i for i in range(k, n) if m[i][k] != 0)
        m[k], m[p] = m[p], m[k]
        for i in range(k + 1, n):
            f = m[i][k] / m[k][k]
            m[i] = [x - f * y for x, y in zip(m[i], m[k])]
    x = [Fraction(0)] * n
    for i in reversed(range(n)):
        x[i] = (m[i][n] - sum(m[i][j] * x[j] for j in range(i + 1, n))) / m[i][i]
    return x


def det_exact(a):
    n = len(a)
    m = [row[:] for row in a]
    det = Fraction(1)
    for k in range(n):
        p = next((i for i in range(k, n) if m[i][k] != 0), None)
        if p is None:
            return Fraction(0)
        if p != k:
            m[k], m[p] = m[p], m[k]
            det = -det
        det *= m[k][k]
        for i in range(k + 1, n):
            f = m[i][k] / m[k][k]
            m[i] = [x - f * y for x, y in zip(m[i], m[k])]
    return det


def cholesky_mp(a):
    n = len(a)
    l = [[mpf(0)] * n for _ in range(n)]
    for i in range(n):
        for j in range(i + 1):
            s = m_of(a[i][j]) - sum(l[i][k] * l[j][k] for k in range(j))
            l[i][j] = sqrt(s) if i == j else s / l[j][j]
    return l


def main():
    rng = random.Random(SEED)
    out = [
        "// GENERATED by scripts/generate_one_rounding_refs.py - do not edit.",
        "// Dyadic inputs (<= 8 fraction bits, |x| <= 8); references exact or mpmath",
        "// at 120 digits, parsed at the build's split (nearest, ties toward +inf).",
        "",
    ]

    # dot: exact dyadic sums (the rounding of the sum itself, ties included)
    out.append("/// (a, b, a . b)")
    out.append("pub const DOT: &[(&[&str], &[&str], &str)] = &[")
    for _ in range(60):
        n = rng.randint(2, 8)
        a = [dy(rng, -4, 4, rng.randint(4, 8)) for _ in range(n)]
        b = [dy(rng, -4, 4, rng.randint(4, 8)) for _ in range(n)]
        out.append(f'    ({vec_lit(a)}, {vec_lit(b)}, "{lit(sum(x * y for x, y in zip(a, b)))}"),')
    out.append("];\n")

    # length and distance
    out.append("/// (v, |v|)")
    out.append("pub const LENGTH: &[(&[&str], &str)] = &[")
    for i in range(60):
        n = rng.randint(1, 8)
        scale = 8 if i % 3 else rng.randint(1, 8)   # include short vectors (a few units)
        v = [dy(rng, -scale, scale, 8) / (1 if i % 3 else 64) for _ in range(n)]
        v = [Fraction(int(x * 256), 256) for x in v]
        out.append(f'    ({vec_lit(v)}, "{lit(sqrt(sum(m_of(x) ** 2 for x in v)))}"),')
    out.append("];\n")
    out.append("/// (a, b, |a - b|)")
    out.append("pub const DISTANCE: &[(&[&str], &[&str], &str)] = &[")
    for _ in range(40):
        n = rng.randint(1, 6)
        a = [dy(rng, -2, 2) for _ in range(n)]
        b = [dy(rng, -2, 2) for _ in range(n)]
        out.append(f'    ({vec_lit(a)}, {vec_lit(b)}, "{lit(sqrt(sum(m_of(x - y) ** 2 for x, y in zip(a, b))))}"),')
    out.append("];\n")

    # Frobenius norm (3 x 3, row-major)
    out.append("/// (A 3x3 row-major, ||A||_F)")
    out.append("pub const FROBENIUS: &[(&[&str], &str)] = &[")
    for _ in range(30):
        a = [dy(rng, -2, 2) for _ in range(9)]
        out.append(f'    ({vec_lit(a)}, "{lit(sqrt(sum(m_of(x) ** 2 for x in a)))}"),')
    out.append("];\n")

    # Minkowski norm of spacelike vectors, some nearly lightlike
    out.append("/// (v with v0 the timelike coordinate, sqrt(-v0^2 + sum v_i^2))")
    out.append("pub const MINKOWSKI: &[(&[&str], &str)] = &[")
    for i in range(40):
        n = 3
        s = [dy(rng, -3, 3) for _ in range(n - 1)]
        sp = sum(x * x for x in s)
        if i % 2:
            # near cancellation: v0^2 just below the spatial sum
            v0 = Fraction(int(sqrt(m_of(sp)) * 256), 256) if sp > 0 else Fraction(0)
            while v0 * v0 >= sp and v0 > 0:
                v0 -= Fraction(1, 256)
        else:
            v0 = dy(rng, -1, 1)
            if v0 * v0 >= sp:
                v0 = Fraction(0)
        v = [v0] + s
        q = -v0 * v0 + sp
        out.append(f'    ({vec_lit(v)}, "{lit(sqrt(m_of(q)))}"),')
    out.append("];\n")

    # QR: R factor of dyadic 3 x 3
    out.append("/// (A 3x3 row-major, R 3x3, Q 3x3 row-major; qr_decompose's sign convention)")
    out.append("pub const QR_R: &[(&[&str], &[&str], &[&str])] = &[")
    for _ in range(25):
        a = [[dy(rng, -2, 2) for _ in range(3)] for _ in range(3)]
        q, r = householder_qr(a)
        flat_a = [x for row in a for x in row]
        flat_r = [r[i][j] if j >= i else mpf(0) for i in range(3) for j in range(3)]
        flat_q = [q[i][j] for i in range(3) for j in range(3)]
        quoted = lambda xs: "&[" + ", ".join(chr(34) + lit(x) + chr(34) for x in xs) + "]"
        out.append(f'    ({vec_lit(flat_a)}, {quoted(flat_r)}, {quoted(flat_q)}),')
    out.append("];\n")

    # symmetrize rank-3 2x2x2 over all indices: mean of 6 permutations
    import itertools
    out.append("/// (T 2x2x2 row-major, symmetrized T over indices [0, 1, 2])")
    out.append("pub const SYMMETRIZE3: &[(&[&str], &[&str])] = &[")
    for _ in range(20):
        t = [dy(rng, -2, 2) for _ in range(8)]
        idx = lambda i, j, k: 4 * i + 2 * j + k
        sym = []
        for i in range(2):
            for j in range(2):
                for k in range(2):
                    tot = sum(t[idx(*p)] for p in itertools.permutations((i, j, k)))
                    sym.append(mpf(tot.numerator) / tot.denominator / 6)
        out.append(f'    ({vec_lit(t)}, &[{", ".join(chr(34) + lit(x) + chr(34) for x in sym)}]),')
    out.append("];\n")

    # matrix_exp of small dyadic 2 x 2
    out.append("/// (A 2x2 row-major, expm(A) row-major)")
    out.append("pub const EXPM: &[(&[&str], &[&str])] = &[")
    for _ in range(20):
        a = [dy(rng, -1, 1) for _ in range(4)]
        e = expm(matrix([[m_of(a[0]), m_of(a[1])], [m_of(a[2]), m_of(a[3])]]))
        out.append(f'    ({vec_lit(a)}, &[{", ".join(chr(34) + lit(e[i, j]) + chr(34) for i in range(2) for j in range(2))}]),')
    out.append("];\n")

    # matrix_log of a dyadic 2 x 2 near the identity (positive spectrum)
    out.append("/// (A 2x2 row-major, logm(A) row-major)")
    out.append("pub const LOGM: &[(&[&str], &[&str])] = &[")
    for _ in range(20):
        a = [Fraction(1) + dy(rng, -1, 1) / 4 for _ in range(1)]
        m = [Fraction(1) + dy(rng, -1, 1) / 4, dy(rng, -1, 1) / 4, dy(rng, -1, 1) / 4, Fraction(1) + dy(rng, -1, 1) / 4]
        m = [Fraction(int(x * 256), 256) for x in m]
        l = logm(matrix([[m_of(m[0]), m_of(m[1])], [m_of(m[2]), m_of(m[3])]]))
        out.append(f'    ({vec_lit(m)}, &[{", ".join(chr(34) + lit(l[i, j].real) + chr(34) for i in range(2) for j in range(2))}]),')
    out.append("];\n")

    # matrix_sqrt of a dyadic SPD 2 x 2 (A^T A + I)
    out.append("/// (A 2x2 row-major SPD, sqrtm(A) row-major)")
    out.append("pub const SQRTM: &[(&[&str], &[&str])] = &[")
    for _ in range(20):
        b = [dy(rng, -2, 2, 4) for _ in range(4)]
        m = [b[0] * b[0] + b[2] * b[2] + 1, b[0] * b[1] + b[2] * b[3], b[0] * b[1] + b[2] * b[3], b[1] * b[1] + b[3] * b[3] + 1]
        r = sqrtm(matrix([[m_of(m[0]), m_of(m[1])], [m_of(m[2]), m_of(m[3])]]))
        out.append(f'    ({vec_lit(m)}, &[{", ".join(chr(34) + lit(r[i, j].real if hasattr(r[i, j], "real") else r[i, j]) + chr(34) for i in range(2) for j in range(2))}]),')
    out.append("];\n")

    # SO(3) exp of a dyadic rotation vector
    out.append("/// (omega, expm(hat(omega)) row-major)")
    out.append("pub const SO3_EXP: &[(&[&str], &[&str])] = &[")
    for _ in range(20):
        w = [dy(rng, -2, 2) for _ in range(3)]
        x, y, z = (m_of(c) for c in w)
        e = expm(matrix([[0, -z, y], [z, 0, -x], [-y, x, 0]]))
        out.append(f'    ({vec_lit(w)}, &[{", ".join(chr(34) + lit(e[i, j]) + chr(34) for i in range(3) for j in range(3))}]),')
    out.append("];\n")

    # solvers: their own stream, so the tables above are unchanged
    rng = random.Random(SEED + 1)
    quoted = lambda xs: "&[" + ", ".join(chr(34) + lit(x) + chr(34) for x in xs) + "]"

    # LU with partial pivoting (first row wins a tie; candidates kept well
    # apart so rounding cannot flip a pivot), solve, inverse, determinant
    out.append("/// (n, A row-major, b, L row-major, U row-major, x = A^-1 b, A^-1 row-major, det A)")
    out.append("pub const LU: &[(usize, &[&str], &[&str], &[&str], &[&str], &[&str], &[&str], &str)] = &[")
    made = 0
    while made < 30:
        n = rng.randint(2, 4)
        a = [[dy(rng, -2, 2) + (2 if i == j else 0) * rng.choice([1, -1]) for j in range(n)] for i in range(n)]
        lu = doolittle(a)
        if lu is None:
            continue
        l, u, perm = lu
        b = [dy(rng, -2, 2) for _ in range(n)]
        x = solve_exact(a, b)
        inv = [solve_exact(a, [Fraction(int(i == j)) for i in range(n)]) for j in range(n)]
        det = det_exact(a)
        if abs(det) < Fraction(1, 4):
            continue
        made += 1
        flat = lambda m: [m[i][j] for i in range(n) for j in range(n)]
        inv_rows = [[inv[j][i] for j in range(n)] for i in range(n)]
        out.append(f'    ({n}, {vec_lit(flat(a))}, {vec_lit(b)}, {quoted(flat(l))}, {quoted(flat(u))}, {quoted(x)}, {quoted(flat(inv_rows))}, "{lit(det)}"),')
    out.append("];\n")

    # Cholesky of dyadic SPD B^T B + I: L (mpmath), solve, determinant (exact)
    out.append("/// (n, A row-major SPD, b, L row-major, x = A^-1 b, det A)")
    out.append("pub const CHOLESKY: &[(usize, &[&str], &[&str], &[&str], &[&str], &str)] = &[")
    for _ in range(30):
        n = rng.randint(2, 4)
        bm = [[dy(rng, -1, 1, 4) for _ in range(n)] for _ in range(n)]
        a = [[sum(bm[k][i] * bm[k][j] for k in range(n)) + (1 if i == j else 0) for j in range(n)] for i in range(n)]
        l = cholesky_mp(a)
        b = [dy(rng, -2, 2) for _ in range(n)]
        x = solve_exact(a, b)
        out.append(f'    ({n}, {vec_lit([a[i][j] for i in range(n) for j in range(n)])}, {vec_lit(b)}, {quoted([l[i][j] for i in range(n) for j in range(n)])}, {quoted(x)}, "{lit(det_exact(a))}"),')
    out.append("];\n")

    # QR solve of square dyadic systems (same matrices as the LU kind)
    out.append("/// (n, A row-major, b, x = A^-1 b)")
    out.append("pub const QR_SOLVE: &[(usize, &[&str], &[&str], &[&str])] = &[")
    made = 0
    while made < 30:
        n = rng.randint(2, 4)
        a = [[dy(rng, -2, 2) + (2 if i == j else 0) for j in range(n)] for i in range(n)]
        if abs(det_exact(a)) < Fraction(1, 4):
            continue
        b = [dy(rng, -2, 2) for _ in range(n)]
        made += 1
        out.append(f'    ({n}, {vec_lit([a[i][j] for i in range(n) for j in range(n)])}, {vec_lit(b)}, {quoted(solve_exact(a, b))}),')
    out.append("];\n")

    # iterative decompositions: their own stream
    rng = random.Random(SEED + 2)

    def normalized_columns(vecs):
        """Each column's sign fixed so its largest-magnitude entry is positive."""
        out = []
        for col in vecs:
            k = max(range(len(col)), key=lambda t: abs(col[t]))
            out.append([-x for x in col] if col[k] < 0 else col)
        return out

    # symmetric eigenproblem: eigenvalues >= 1/8 apart, sorted by |lambda| desc
    out.append("/// (n, A row-major symmetric, eigenvalues by |lambda| desc, eigenvectors as columns row-major;")
    out.append("/// each column's largest-magnitude entry positive)")
    out.append("pub const EIGEN_SYM: &[(usize, &[&str], &[&str], &[&str])] = &[")
    made = 0
    while made < 25:
        n = rng.randint(2, 4)
        a = [[Fraction(0)] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                a[i][j] = a[j][i] = dy(rng, -2, 2)
        ev, evec = mp.eigsy(matrix([[m_of(x) for x in row] for row in a]))
        pairs = sorted(((ev[i], [evec[r, i] for r in range(n)]) for i in range(n)), key=lambda t: -abs(t[0]))
        vals = [t[0] for t in pairs]
        if any(abs(abs(vals[i]) - abs(vals[i + 1])) < mpf(1) / 8 for i in range(n - 1)):
            continue
        if any(abs(vals[i] - vals[j]) < mpf(1) / 8 for i in range(n) for j in range(i + 1, n)):
            continue
        cols = normalized_columns([t[1] for t in pairs])
        made += 1
        out.append(f'    ({n}, {vec_lit([a[i][j] for i in range(n) for j in range(n)])}, {quoted(vals)}, {quoted([cols[j][i] for i in range(n) for j in range(n)])}),')
    out.append("];\n")

    # SVD of dyadic m x n (m >= n): singular values >= 1/8 apart and >= 1/8
    out.append("/// (m, n, A row-major, sigma desc, first n columns of U row-major (m x n), V columns row-major (n x n);")
    out.append("/// each V column's largest-magnitude entry positive, U columns flipped with it)")
    out.append("pub const SVD: &[(usize, usize, &[&str], &[&str], &[&str], &[&str])] = &[")
    made = 0
    while made < 25:
        n = rng.randint(2, 3)
        m = rng.randint(n, 4)
        a = [[dy(rng, -2, 2) for _ in range(n)] for _ in range(m)]
        u, sv, vt = mp.svd_r(matrix([[m_of(x) for x in row] for row in a]))
        sig = [sv[i] for i in range(n)]
        if any(sig[i] - sig[i + 1] < mpf(1) / 8 for i in range(n - 1)) or sig[-1] < mpf(1) / 8:
            continue
        vcols, ucols = [], []
        for i in range(n):
            vc = [vt[i, r] for r in range(n)]
            uc = [u[r, i] for r in range(m)]
            k = max(range(n), key=lambda t: abs(vc[t]))
            if vc[k] < 0:
                vc = [-x for x in vc]
                uc = [-x for x in uc]
            vcols.append(vc)
            ucols.append(uc)
        made += 1
        out.append(f'    ({m}, {n}, {vec_lit([a[i][j] for i in range(m) for j in range(n)])}, {quoted(sig)}, '
                   f'{quoted([ucols[j][i] for i in range(m) for j in range(n)])}, {quoted([vcols[j][i] for i in range(n) for j in range(n)])}),')
    out.append("];\n")

    # Schur of S T S^-1: S unimodular integer, T upper triangular dyadic with
    # eigenvalues >= 1/8 apart; A dyadic exactly, eigenvalues exact
    out.append("/// (n, A row-major, eigenvalues ascending (exact))")
    out.append("pub const SCHUR: &[(usize, &[&str], &[&str])] = &[")
    made = 0
    while made < 25:
        n = rng.randint(2, 4)
        t = [[dy(rng, -2, 2, 4) if j >= i else Fraction(0) for j in range(n)] for i in range(n)]
        diag = [t[i][i] for i in range(n)]
        if any(abs(diag[i] - diag[j]) < Fraction(1, 8) for i in range(n) for j in range(i + 1, n)):
            continue
        # S = unit lower times unit upper, small integers: det 1
        lo = [[Fraction(rng.randint(-1, 1)) if j < i else Fraction(int(i == j)) for j in range(n)] for i in range(n)]
        up = [[Fraction(rng.randint(-1, 1)) if j > i else Fraction(int(i == j)) for j in range(n)] for i in range(n)]
        mul = lambda x, y: [[sum(x[i][k] * y[k][j] for k in range(n)) for j in range(n)] for i in range(n)]
        sm = mul(lo, up)
        sinv = [solve_exact(sm, [Fraction(int(i == j)) for i in range(n)]) for j in range(n)]
        sinv = [[sinv[j][i] for j in range(n)] for i in range(n)]
        a = mul(mul(sm, t), sinv)
        if max(abs(x) for row in a for x in row) > 6:
            continue
        made += 1
        out.append(f'    ({n}, {vec_lit([a[i][j] for i in range(n) for j in range(n)])}, {vec_lit(sorted(diag))}),')
    out.append("];\n")

    # matrix functions beyond the small-norm cases: expm of norms up to 7 and
    # logm / sqrtm of SPD matrices with spectra from 1/8 to 60 (own stream).
    # A case whose input or result leaves a build's storage range is skipped
    # there by the test.
    rng = random.Random(SEED + 3)
    out.append("/// (n, A row-major, expm(A) row-major): norms up to 7")
    out.append("pub const EXPM_WIDE: &[(usize, &[&str], &[&str])] = &[")
    for i in range(24):
        n = 2 if i < 12 else 3
        scale = [1, 2, 3, 4][i % 4]
        a = [dy(rng, -scale, scale, 6) for _ in range(n * n)]
        e = expm(matrix([[m_of(a[r * n + c]) for c in range(n)] for r in range(n)]))
        out.append(f'    ({n}, {vec_lit(a)}, {quoted([e[r, c] for r in range(n) for c in range(n)])}),')
    out.append("];\n")
    spd = []
    for i in range(24):
        n = 2 if i < 12 else 3
        b = [[dy(rng, -2, 2, 4) for _ in range(n)] for _ in range(n)]
        shift = Fraction([1, 1, 8, 1][i % 4], [8, 1, 1, 64][i % 4])
        m = [[sum(b[k][r] * b[k][c] for k in range(n)) + (shift if r == c else 0) for c in range(n)] for r in range(n)]
        spd.append((n, m))
    out.append("/// (n, A row-major SPD, logm(A) row-major)")
    out.append("pub const LOGM_WIDE: &[(usize, &[&str], &[&str])] = &[")
    for n, m in spd:
        l = logm(matrix([[m_of(x) for x in row] for row in m]))
        out.append(f'    ({n}, {vec_lit([x for row in m for x in row])}, {quoted([l[r, c].real for r in range(n) for c in range(n)])}),')
    out.append("];\n")
    out.append("/// (n, A row-major SPD, sqrtm(A) row-major)")
    out.append("pub const SQRTM_WIDE: &[(usize, &[&str], &[&str])] = &[")
    for n, m in spd:
        r_ = sqrtm(matrix([[m_of(x) for x in row] for row in m]))
        vals = [r_[r, c].real if hasattr(r_[r, c], "real") else r_[r, c] for r in range(n) for c in range(n)]
        out.append(f'    ({n}, {vec_lit([x for row in m for x in row])}, {quoted(vals)}),')
    out.append("];\n")

    # rms_norm: x_i w_i / sqrt(mean(x^2) + eps), eps a Q64.64 raw (own
    # stream). The second table reaches large and outlier magnitudes; a case
    # that leaves a build's storage range is skipped there by the test.
    rng = random.Random(SEED + 4)
    eps_raws = [0, 1 << 44, round(Fraction(1, 10 ** 5) * 2 ** 64), round(Fraction(1, 10 ** 6) * 2 ** 64), 1 << 56, 1 << 62]

    def rms_case(x, w, eps_raw):
        radicand = m_of(sum(t * t for t in x) / len(x) + Fraction(eps_raw, 2 ** 64))
        root = sqrt(radicand)
        return f'    ({vec_lit(x)}, {vec_lit(w)}, {eps_raw}, {quoted([m_of(a * b) / root for a, b in zip(x, w)])}),'

    out.append("/// (x, weight, eps as a Q64.64 raw, x_i w_i / sqrt(mean(x^2) + eps))")
    out.append("pub const RMS_NORM: &[(&[&str], &[&str], i128, &[&str])] = &[")
    for i in range(60):
        n = rng.randint(1, 12)
        if i % 4 == 3:
            # a few storage units at 8 fraction bits: the bottom of the range
            x = [Fraction(rng.randint(-3, 3), 256) for _ in range(n)]
        else:
            x = [dy(rng, -8, 8) for _ in range(n)]
        w = [dy(rng, -4, 4) for _ in range(n)]
        eps_raw = eps_raws[rng.randrange(len(eps_raws))]
        if eps_raw == 0 and all(t == 0 for t in x):
            x[0] = Fraction(1, 256)
        out.append(rms_case(x, w, eps_raw))
    out.append("];\n")
    out.append("/// (x, weight, eps as a Q64.64 raw, outputs): magnitudes to 2000, outliers")
    out.append("pub const RMS_NORM_WIDE: &[(&[&str], &[&str], i128, &[&str])] = &[")
    for i in range(40):
        n = rng.randint(2, 64)
        scale = [20, 100, 500, 2000][i % 4]
        if i % 3 == 0:
            # one outlier among small values
            x = [dy(rng, -1, 1) for _ in range(n)]
            x[rng.randrange(n)] = dy(rng, -scale, scale, 4)
        else:
            x = [dy(rng, -scale, scale, 4) for _ in range(n)]
        w = [dy(rng, -16, 16, 6) for _ in range(n)]
        out.append(rms_case(x, w, eps_raws[rng.randrange(len(eps_raws))]))
    out.append("];\n")

    # rotate_pairs: one pair, exact dyadic results (16 fraction bits)
    out.append("/// (x0, x1, sin, cos, x0 cos - x1 sin, x0 sin + x1 cos)")
    out.append("pub const ROTATE_PAIRS: &[(&str, &str, &str, &str, &str, &str)] = &[")
    for _ in range(80):
        x0, x1, sn, cs = dy(rng, -8, 8), dy(rng, -8, 8), dy(rng, -1, 1), dy(rng, -1, 1)
        out.append(f'    ("{lit(x0)}", "{lit(x1)}", "{lit(sn)}", "{lit(cs)}", "{lit(x0 * cs - x1 * sn)}", "{lit(x0 * sn + x1 * cs)}"),')
    out.append("];\n")

    with open("tests/data/one_rounding_refs.rs", "w") as fh:
        fh.write("\n".join(out))


if __name__ == "__main__":
    main()
