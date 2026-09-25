#!/usr/bin/env python3
"""mpmath references for the Lie group and fiber bundle compute-tier rework
(tests/data/lie_fiber_compute_tier_refs.rs).

Covers SO(3) and SE(3) exp (including small angles, where theta^2 used to be
rounded to storage and could reach 0 raw), log round trips out to near pi,
the Manifold chains (exp_map, log_map, distance, SO(3) parallel transport),
SE(3) inverse, adjoint and bracket, SO(n) / GL(n) / SL(n) chains, adjoints,
brackets and SL(n) project_traceless, and the vector bundle's horizontal lift,
discrete parallel transport, curvature and a principal bundle transition
inverse.

Inputs are dyadic. Each table row carries the fewest fraction bits at which
its inputs are exact (8 for most rows); the test skips rows whose inputs are
not exact at the build's split. References are exact rationals (printed
exactly when dyadic) or mpmath at 120 significant digits; the test parses each
with the exact literal parser, which rounds it to nearest at the build's
split, so one table serves every build and every error is in storage units.

Usage: python3 scripts/generate_lie_fiber_compute_tier_refs.py
"""
import random
from fractions import Fraction

from mpmath import mp, mpf, sqrt, sin, cos, atan2, matrix, expm, logm, pi

mp.dps = 140
SEED = 20260925
DIGITS = 120


def lit(v):
    """A literal for a Fraction (exact when dyadic) or an mpf (120 digits)."""
    if isinstance(v, Fraction):
        if v.denominator & (v.denominator - 1) == 0:
            sign = "-" if v < 0 else ""
            v = abs(v)
            d = 0
            while (v * 10 ** d).denominator != 1:
                d += 1
            n = int(v * 10 ** d)
            if d == 0:
                return f"{sign}{n}"
            s = str(n).rjust(d + 1, "0")
            return f"{sign}{s[:-d]}.{s[-d:]}"
        return lit(mpf(v.numerator) / v.denominator)
    if v == 0:
        return "0"
    return mp.nstr(v, DIGITS, strip_zeros=False, min_fixed=-10**9, max_fixed=10**9)


def m_of(x):
    return mpf(x.numerator) / x.denominator if isinstance(x, Fraction) else mpf(x)


def arr(v):
    return "&[" + ", ".join(f'"{lit(x)}"' for x in v) + "]"


def bits_of(values):
    b = 0
    for x in values:
        d = x.denominator
        b = max(b, d.bit_length() - 1)
    return max(b, 8)


def dy(rng, lo, hi, bits=8):
    """A dyadic k / 2^bits in [lo, hi) (bounds exact: ints or decimal strings)."""
    lo, hi = Fraction(lo), Fraction(hi)
    return Fraction(rng.randrange(int(lo * (1 << bits)), int(hi * (1 << bits))), 1 << bits)


# ---------------------------------------------------------------- SO(3) / SE(3)

def hat3(w):
    return matrix([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]])


def so3_exp(w):
    w = [m_of(x) for x in w]
    th = sqrt(sum(x * x for x in w))
    k = hat3(w)
    if th == 0:
        return mp.eye(3)
    return mp.eye(3) + (sin(th) / th) * k + ((1 - cos(th)) / th ** 2) * (k * k)


def se3_v(w):
    w = [m_of(x) for x in w]
    th = sqrt(sum(x * x for x in w))
    k = hat3(w)
    if th == 0:
        return mp.eye(3)
    return mp.eye(3) + ((1 - cos(th)) / th ** 2) * k + ((th - sin(th)) / th ** 3) * (k * k)


def se3_exp(xi):
    r = so3_exp(xi[:3])
    t = se3_v(xi[:3]) * matrix([m_of(x) for x in xi[3:]])
    g = mp.eye(4)
    for i in range(3):
        for j in range(3):
            g[i, j] = r[i, j]
        g[i, 3] = t[i]
    return g


def so3_log(r):
    d = [r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]]
    nd = sqrt(sum(x * x for x in d))
    tr = r[0, 0] + r[1, 1] + r[2, 2]
    th = atan2(nd, tr - 1)
    if nd == 0:
        return [mpf(0)] * 3
    return [th * x / nd for x in d]


def se3_log(g):
    r = g[0:3, 0:3]
    t = matrix([g[0, 3], g[1, 3], g[2, 3]])
    w = so3_log(r)
    th = sqrt(sum(x * x for x in w))
    k = hat3(w)
    if th == 0:
        vinv = mp.eye(3)
    else:
        alpha = th * sin(th) / (2 * (1 - cos(th)))
        vinv = mp.eye(3) - k / 2 + ((1 - alpha) / th ** 2) * (k * k)
    v = vinv * t
    return w + [v[0], v[1], v[2]]


def se3_inv(g):
    r = g[0:3, 0:3]
    t = matrix([g[0, 3], g[1, 3], g[2, 3]])
    h = mp.eye(4)
    rt = r.T
    nt = -(rt * t)
    for i in range(3):
        for j in range(3):
            h[i, j] = rt[i, j]
        h[i, 3] = nt[i]
    return h


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def mat_rows(m, rows, cols):
    return [m[i, j] for i in range(rows) for j in range(cols)]


# ---------------------------------------------------------------- SO(n) etc.

def hat_son(xi, n):
    m = mp.zeros(n, n)
    k = 0
    for i in range(n):
        for j in range(i + 1, n):
            m[i, j] = m_of(xi[k])
            m[j, i] = -m_of(xi[k])
            k += 1
    return m


def vee_son(m, n):
    return [m[i, j] for i in range(n) for j in range(i + 1, n)]


def hat_gln(xi, n):
    return matrix([[m_of(xi[i * n + j]) for j in range(n)] for i in range(n)])


def vee_gln(m, n):
    return [m[i, j] for i in range(n) for j in range(n)]


def hat_sln(xi, n):
    m = mp.zeros(n, n)
    s = mpf(0)
    for i in range(n - 1):
        m[i, i] = m_of(xi[i])
        s += m_of(xi[i])
    m[n - 1, n - 1] = -s
    k = n - 1
    for i in range(n):
        for j in range(n):
            if i != j:
                m[i, j] = m_of(xi[k])
                k += 1
    return m


def vee_sln(m, n):
    v = [m[i, i] for i in range(n - 1)]
    for i in range(n):
        for j in range(n):
            if i != j:
                v.append(m[i, j])
    return v


def traceless(m, n):
    tr = sum(m[i, i] for i in range(n))
    r = m.copy()
    for i in range(n):
        r[i, i] = m[i, i] - tr / n
    return r


def frac_mat(rows):
    return [[Fraction(x) for x in r] for r in rows]


def frac_inverse(a):
    n = len(a)
    m = [row[:] + [Fraction(int(i == j)) for j in range(n)] for i, row in enumerate(a)]
    for c in range(n):
        p = next(r for r in range(c, n) if m[r][c] != 0)
        m[c], m[p] = m[p], m[c]
        pv = m[c][c]
        m[c] = [x / pv for x in m[c]]
        for r in range(n):
            if r != c and m[r][c] != 0:
                f = m[r][c]
                m[r] = [x - f * y for x, y in zip(m[r], m[c])]
    return [row[n:] for row in m]


def frac_mul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def main():
    rng = random.Random(SEED)
    out = [
        "// GENERATED by scripts/generate_lie_fiber_compute_tier_refs.py - do not edit.",
        "// Dyadic inputs; each row's first field is the fewest fraction bits at which",
        "// its inputs are exact (rows are skipped below that). References exact or",
        "// mpmath at 120 digits, parsed at the build's split (nearest, ties toward +inf).",
        "",
    ]
    F = Fraction

    # small angles: exact at the given bits, spanning every profile's threshold
    small = [
        [F(3, 1 << 10), F(0), F(0)],                  # Q22.10: theta 3 raw, theta^2 0 raw
        [F(1, 1 << 9), F(1, 1 << 10), F(-1, 1 << 10)],
        [F(-5, 1 << 12), F(3, 1 << 12), F(1, 1 << 11)],
        [F(1, 1 << 8), F(1, 1 << 8), F(0)],
        [F(0), F(3, 1 << 16), F(-5, 1 << 16)],        # Q16.16: theta ~ 89 raw
        [F(7, 1 << 20), F(-3, 1 << 19), F(1, 1 << 20)],  # |w| = 8.8e-6, below 1e-5 (Q64.64)
        [F(3, 1 << 17), F(0), F(-1, 1 << 17)],        # |w| = 2.4e-5
        [F(1, 1 << 34), F(-1, 1 << 35), F(0)],        # |w| = 6.5e-11, below 1e-10 (Q128.128 threshold)
        [F(3, 1 << 69), F(1, 1 << 68), F(0)],         # |w| = 6.1e-21, below 1e-20 (Q256.256)
        [F(0), F(0), F(0)],
    ]
    moderate = []
    for _ in range(8):
        moderate.append([dy(rng, "-1.5", "1.5") for _ in range(3)])
    near_pi = [
        [F(804, 256), F(0), F(0)],                    # 3.140625, pi - 0.00097
        [F(0), F(-800, 256), F(0)],                   # 3.125
        [F(568, 256), F(568, 256), F(0)],             # |w| = 3.1378
        [F(464, 256), F(464, 256), F(464, 256)],      # |w| = 3.1393
        [F(-2, 1), F(608, 256), F(-1, 4)],            # |w| = 3.1147
        [F(3, 1), F(0), F(-1, 4)],                    # 3.0104
        [F(2, 1), F(-2, 1), F(1, 1)],                 # 3.0
    ]

    out.append("/// (bits, omega, exp(omega) row-major 3x3)")
    out.append("pub const SO3_EXP: &[(u32, &[&str], &[&str])] = &[")
    for w in small + moderate + near_pi:
        out.append(f"    ({bits_of(w)}, {arr(w)}, {arr(mat_rows(so3_exp(w), 3, 3))}),")
    out.append("];\n")

    vs = [[F(3, 2), F(-9, 4), F(3, 4)], [F(-1, 2), F(5, 8), F(2, 1)], [F(1, 1), F(1, 1), F(-3, 2)]]
    out.append("/// (bits, xi = [omega, v], exp(xi) top three rows of the 4x4)")
    out.append("pub const SE3_EXP: &[(u32, &[&str], &[&str])] = &[")
    for idx, w in enumerate(small + moderate + near_pi):
        xi = w + vs[idx % 3]
        out.append(f"    ({bits_of(xi)}, {arr(xi)}, {arr(mat_rows(se3_exp(xi), 3, 4))}),")
    out.append("];\n")

    out.append("/// (bits, omega): log(exp(omega)) should give omega back (|omega| < pi)")
    out.append("pub const SO3_ROUND_TRIP: &[(u32, &[&str])] = &[")
    for w in small + moderate + near_pi:
        out.append(f"    ({bits_of(w)}, {arr(w)}),")
    out.append("];\n")

    out.append("/// (bits, xi): se3_log(se3_exp(xi)) should give xi back")
    out.append("pub const SE3_ROUND_TRIP: &[(u32, &[&str])] = &[")
    for idx, w in enumerate(small + moderate + near_pi):
        xi = w + vs[idx % 3]
        out.append(f"    ({bits_of(xi)}, {arr(xi)}),")
    out.append("];\n")

    # Manifold chains
    pairs = []
    for _ in range(6):
        pairs.append(([dy(rng, "-1.25", "1.25") for _ in range(3)], [dy(rng, "-1.25", "1.25") for _ in range(3)]))
    pairs.append(([F(3, 2), F(0), F(0)], [F(-13, 8), F(0), F(0)]))              # relative angle 3.125
    pairs.append(([F(1, 4), F(1, 2), F(0)], [F(-3, 2), F(-9, 8), F(3, 4)]))
    pairs.append(([F(1, 256), F(0), F(0)], [F(3, 256), F(-1, 256), F(0)]))       # small
    tangents = [dy(rng, -2, 2) for _ in range(3)]

    out.append("/// (base, tangent, exp_map = log(exp(base) exp(tangent)))")
    out.append("pub const SO3_EXP_MAP: &[(&[&str], &[&str], &[&str])] = &[")
    for a, b in pairs:
        res = so3_log(so3_exp(a) * so3_exp(b))
        out.append(f"    ({arr(a)}, {arr(b)}, {arr(res)}),")
    out.append("];\n")

    out.append("/// (base, target, log_map = log(exp(base)^T exp(target)), distance, parallel transport of tangent)")
    out.append("pub const SO3_LOG_MAP: &[(&[&str], &[&str], &[&str], &str, &[&str], &[&str])] = &[")
    for a, b in pairs:
        res = so3_log(so3_exp(a).T * so3_exp(b))
        dist = sqrt(sum(x * x for x in res))
        rh = so3_exp([x / 2 for x in res])
        pt = rh * matrix([m_of(x) for x in tangents])
        out.append(f"    ({arr(a)}, {arr(b)}, {arr(res)}, \"{lit(dist)}\", {arr(tangents)}, {arr([pt[0], pt[1], pt[2]])}),")
    out.append("];\n")

    se3_pairs = []
    for _ in range(5):
        se3_pairs.append(([dy(rng, -1, 1) for _ in range(3)] + [dy(rng, -2, 2) for _ in range(3)],
                          [dy(rng, -1, 1) for _ in range(3)] + [dy(rng, -2, 2) for _ in range(3)]))
    se3_pairs.append(([F(3, 2), F(0), F(0), F(1), F(-1, 2), F(1, 4)], [F(-13, 8), F(0), F(0), F(-3, 4), F(1, 2), F(2)]))
    se3_pairs.append(([F(1, 256), F(0), F(0), F(1), F(1), F(1)], [F(3, 256), F(-1, 256), F(0), F(-1, 2), F(1, 4), F(0)]))
    out.append("/// (base, tangent, exp_map, target, log_map, distance) with target = the tangent row")
    out.append("pub const SE3_MAPS: &[(&[&str], &[&str], &[&str], &[&str], &str)] = &[")
    for a, b in se3_pairs:
        em = se3_log(se3_exp(a) * se3_exp(b))
        lm = se3_log(se3_inv(se3_exp(a)) * se3_exp(b))
        dist = sqrt(sum(x * x for x in lm))
        out.append(f"    ({arr(a)}, {arr(b)}, {arr(em)}, {arr(lm)}, \"{lit(dist)}\"),")
    out.append("];\n")

    # SE(3) exact algebra: inverse, adjoint, brackets on dyadic R (not orthogonal), t
    out.append("/// (R row-major, t, xi, -R^T t, Ad_g xi, [xi, eta] se3 with eta, eta, [w1, w2] so3)")
    out.append("pub const SE3_ALGEBRA: &[(&[&str], &[&str], &[&str], &[&str], &[&str], &[&str], &[&str], &[&str])] = &[")
    for _ in range(6):
        r = [[dy(rng, -1, 1) for _ in range(3)] for _ in range(3)]
        t = [dy(rng, -3, 3) for _ in range(3)]
        xi = [dy(rng, -2, 2) for _ in range(6)]
        eta = [dy(rng, -2, 2) for _ in range(6)]
        rt_t = [-sum(r[k][i] * t[k] for k in range(3)) for i in range(3)]
        rw = [sum(r[i][k] * xi[k] for k in range(3)) for i in range(3)]
        rv = [sum(r[i][k] * xi[3 + k] for k in range(3)) for i in range(3)]
        tx = cross(t, rw)
        adj = rw + [rv[i] + tx[i] for i in range(3)]
        w = cross(xi[:3], eta[:3])
        v1 = cross(xi[:3], eta[3:])
        v2 = cross(eta[:3], xi[3:])
        br = w + [v1[i] - v2[i] for i in range(3)]
        rflat = [x for row in r for x in row]
        out.append(f"    ({arr(rflat)}, {arr(t)}, {arr(xi)}, {arr(rt_t)}, {arr(adj)}, {arr(br)}, {arr(eta)}, {arr(w)}),")
    out.append("];\n")

    # SO(4): chains, log of a dyadic near-rotation, adjoint, bracket
    n = 4
    out.append("/// SO(4): (base, tangent, exp_map, log_map, distance)")
    out.append("pub const SO4_MAPS: &[(&[&str], &[&str], &[&str], &[&str], &str)] = &[")
    for _ in range(4):
        a = [dy(rng, "-0.75", "0.75") for _ in range(6)]
        b = [dy(rng, "-0.75", "0.75") for _ in range(6)]
        ga, gb = expm(hat_son(a, n)), expm(hat_son(b, n))
        em = logm(ga * gb)
        lm = logm(ga.T * gb)
        em = vee_son((em - em.T) / 2, n)
        lm = vee_son((lm - lm.T) / 2, n)
        dist = sqrt(sum(x * x for x in lm))
        out.append(f"    ({arr(a)}, {arr(b)}, {arr(em)}, {arr(lm)}, \"{lit(dist)}\"),")
    out.append("];\n")

    out.append("/// SO(4): (g dyadic row-major, lie_log(g) = vee(skew(logm g)), xi, Ad_g xi = vee(g xi^ g^T), eta, [xi, eta])")
    out.append("pub const SO4_ALGEBRA: &[(&[&str], &[&str], &[&str], &[&str], &[&str], &[&str])] = &[")
    for _ in range(4):
        a = [dy(rng, "-0.75", "0.75") for _ in range(6)]
        g = expm(hat_son(a, n))
        gd = [[Fraction(int(mp.nint(g[i, j] * 256)), 256) for j in range(n)] for i in range(n)]
        gm = matrix([[m_of(x) for x in row] for row in gd])
        lg = logm(gm)
        lg = vee_son((lg - lg.T) / 2, n)
        xi = [dy(rng, -2, 2) for _ in range(6)]
        eta = [dy(rng, -2, 2) for _ in range(6)]
        # exact rationals
        def hat_f(x):
            m = [[Fraction(0)] * n for _ in range(n)]
            k = 0
            for i in range(n):
                for j in range(i + 1, n):
                    m[i][j] = x[k]
                    m[j][i] = -x[k]
                    k += 1
            return m
        gt = [[gd[j][i] for j in range(n)] for i in range(n)]
        adj = frac_mul(frac_mul(gd, hat_f(xi)), gt)
        adj = [adj[i][j] for i in range(n) for j in range(i + 1, n)]
        ha, hb = hat_f(xi), hat_f(eta)
        ab, ba = frac_mul(ha, hb), frac_mul(hb, ha)
        br = [ab[i][j] - ba[i][j] for i in range(n) for j in range(i + 1, n)]
        gflat = [x for row in gd for x in row]
        out.append(f"    ({arr(gflat)}, {arr(lg)}, {arr(xi)}, {arr(adj)}, {arr(eta)}, {arr(br)}),")
    out.append("];\n")

    # GL(2) and SL(2), SL(3)
    for name, dim, hat, vee, proj in (("GL2", 2, hat_gln, vee_gln, False), ("SL2", 2, hat_sln, vee_sln, True), ("SL3", 3, hat_sln, vee_sln, True)):
        adim = dim * dim - (1 if proj else 0)
        out.append(f"/// {name}: (base, tangent, exp_map, log_map, distance)")
        out.append(f"pub const {name}_MAPS: &[(&[&str], &[&str], &[&str], &[&str], &str)] = &[")
        for _ in range(4):
            a = [dy(rng, "-0.5", "0.5") for _ in range(adim)]
            b = [dy(rng, "-0.5", "0.5") for _ in range(adim)]
            ga, gb = expm(hat(a, dim)), expm(hat(b, dim))
            em = logm(ga * gb)
            lm = logm(ga ** -1 * gb)
            if proj:
                em, lm = traceless(em, dim), traceless(lm, dim)
            dist = sqrt(sum(x * x for x in vee(lm, dim)))
            out.append(f"    ({arr(a)}, {arr(b)}, {arr(vee(em, dim))}, {arr(vee(lm, dim))}, \"{lit(dist)}\"),")
        out.append("];\n")

        out.append(f"/// {name}: (g dyadic row-major, xi, Ad_g xi = vee(g xi^ g^-1), eta, [xi, eta], g^-1)")
        out.append(f"pub const {name}_ALGEBRA: &[(&[&str], &[&str], &[&str], &[&str], &[&str], &[&str])] = &[")
        for _ in range(5):
            while True:
                g = [[dy(rng, -2, 2) for _ in range(dim)] for _ in range(dim)]
                if frac_det(g) != 0 and abs(frac_det(g)) >= Fraction(1, 4):
                    break
            xi = [dy(rng, -2, 2) for _ in range(adim)]
            eta = [dy(rng, -2, 2) for _ in range(adim)]

            def hat_f(x):
                if not proj:
                    return [[x[i * dim + j] for j in range(dim)] for i in range(dim)]
                m = [[Fraction(0)] * dim for _ in range(dim)]
                s = Fraction(0)
                for i in range(dim - 1):
                    m[i][i] = x[i]
                    s += x[i]
                m[dim - 1][dim - 1] = -s
                k = dim - 1
                for i in range(dim):
                    for j in range(dim):
                        if i != j:
                            m[i][j] = x[k]
                            k += 1
                return m

            def vee_f(m):
                if not proj:
                    return [m[i][j] for i in range(dim) for j in range(dim)]
                v = [m[i][i] for i in range(dim - 1)]
                for i in range(dim):
                    for j in range(dim):
                        if i != j:
                            v.append(m[i][j])
                return v
            adj = frac_mul(frac_mul(g, hat_f(xi)), frac_inverse(g))
            ha, hb = hat_f(xi), hat_f(eta)
            ab, ba = frac_mul(ha, hb), frac_mul(hb, ha)
            br = [[ab[i][j] - ba[i][j] for j in range(dim)] for i in range(dim)]
            gflat = [x for row in g for x in row]
            ginv = [x for row in frac_inverse(g) for x in row]
            out.append(f"    ({arr(gflat)}, {arr(xi)}, {arr(vee_f(adj))}, {arr(eta)}, {arr(vee_f(br))}, {arr(ginv)}),")
        out.append("];\n")

    out.append("/// (n, m row-major, m - tr(m)/n I)")
    out.append("pub const PROJECT_TRACELESS: &[(usize, &[&str], &[&str])] = &[")
    for dim in (2, 2, 2, 3, 3, 4):
        m = [[dy(rng, -4, 4) for _ in range(dim)] for _ in range(dim)]
        tr = sum(m[i][i] for i in range(dim))
        r = [m[i][j] - (tr / dim if i == j else 0) for i in range(dim) for j in range(dim)]
        out.append(f"    ({dim}, {arr([x for row in m for x in row])}, {arr(r)}),")
    out.append("];\n")

    # Vector bundle: exact rationals
    out.append("/// (n, k, coeffs [a, b, i], total point (base then fiber), base tangent, horizontal lift)")
    out.append("pub const HORIZONTAL_LIFT: &[(usize, usize, &[&str], &[&str], &[&str], &[&str])] = &[")
    bundles = []
    for (nb, kf) in ((2, 2), (3, 2), (2, 3), (3, 3)):
        coeffs = [dy(rng, -2, 2) for _ in range(kf * kf * nb)]
        bundles.append((nb, kf, coeffs))
        for _ in range(2):
            tp = [dy(rng, -3, 3) for _ in range(nb + kf)]
            v = [dy(rng, -2, 2) for _ in range(nb)]
            fib = tp[nb:]
            lift = v + [-sum(coeffs[a * kf * nb + b * nb + i] * fib[b] * v[i] for b in range(kf) for i in range(nb)) for a in range(kf)]
            out.append(f"    ({nb}, {kf}, {arr(coeffs)}, {arr(tp)}, {arr(v)}, {arr(lift)}),")
    out.append("];\n")

    out.append("/// (n, k, coeffs, path points flattened, initial fiber, transported fiber)")
    out.append("pub const TRANSPORT_ALONG: &[(usize, usize, &[&str], &[&str], &[&str], &[&str])] = &[")
    for nb, kf, coeffs in bundles:
        for steps in (4, 12):
            pts = [[dy(rng, -1, 1) for _ in range(nb)]]
            for _ in range(steps):
                pts.append([pts[-1][i] + dy(rng, "-0.25", "0.25") for i in range(nb)])
            xi0 = [dy(rng, -2, 2) for _ in range(kf)]
            xi = xi0[:]
            for s in range(steps):
                dx = [pts[s + 1][i] - pts[s][i] for i in range(nb)]
                xi = [xi[a] - sum(coeffs[a * kf * nb + b * nb + i] * xi[b] * dx[i] for b in range(kf) for i in range(nb)) for a in range(kf)]
            flat = [x for p in pts for x in p]
            out.append(f"    ({nb}, {kf}, {arr(coeffs)}, {arr(flat)}, {arr(xi0)}, {arr(xi)}),")
    out.append("];\n")

    out.append("/// (n, k, coeffs, curvature [a, b, i, j] flattened)")
    out.append("pub const CURVATURE: &[(usize, usize, &[&str], &[&str])] = &[")
    for nb, kf, coeffs in bundles:
        c = lambda a, b, i: coeffs[a * kf * nb + b * nb + i]
        curv = []
        for a in range(kf):
            for b in range(kf):
                for i in range(nb):
                    for j in range(nb):
                        curv.append(sum(c(a, q, i) * c(q, b, j) - c(a, q, j) * c(q, b, i) for q in range(kf)))
        out.append(f"    ({nb}, {kf}, {arr(coeffs)}, {arr(curv)}),")
    out.append("];\n")

    out.append("/// (g 3x3 row-major, g^-1)")
    out.append("pub const TRANSITION_INVERSE: &[(&[&str], &[&str])] = &[")
    for _ in range(5):
        while True:
            g = [[dy(rng, -2, 2) for _ in range(3)] for _ in range(3)]
            if abs(frac_det(g)) >= Fraction(1, 2):
                break
        inv = frac_inverse(g)
        out.append(f"    ({arr([x for row in g for x in row])}, {arr([x for row in inv for x in row])}),")
    out.append("];\n")

    with open("tests/data/lie_fiber_compute_tier_refs.rs", "w") as fh:
        fh.write("\n".join(out))


def frac_det(a):
    n = len(a)
    if n == 1:
        return a[0][0]
    if n == 2:
        return a[0][0] * a[1][1] - a[0][1] * a[1][0]
    return sum((-1) ** j * a[0][j] * frac_det([row[:j] + row[j + 1:] for row in a[1:]]) for j in range(n))


if __name__ == "__main__":
    main()
