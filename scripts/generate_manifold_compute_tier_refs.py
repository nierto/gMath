#!/usr/bin/env python3
"""mpmath references for the manifold compute-tier rework
(tests/data/manifold_compute_tier_refs.rs).

Two kinds of table:

* Dyadic tables (SPD, Stiefel, norms, product inner product): inputs have at
  most 8 fraction bits and |x| <= 8, so they are stored exactly on every gated
  build; references are exact rationals or mpmath at 140 digits, written with
  enough digits for Q256.256 and parsed at the build's split.

* Per-split tables (Sphere, HyperbolicSpace, Grassmannian, the product
  distance): points on a curved manifold have no dyadic coordinates, so the
  inputs are ideal points written with 100 decimals and the test parses them
  at the build's split (nearest, ties toward +infinity). This generator
  rounds the same literals the same way for each split F and evaluates the
  references on the STORED values, so the stored points are off the manifold
  by at most 2^-F and every reference is the exact result for exactly the
  inputs the build sees. The formulas are the scale-invariant ones the
  implementation uses (sphere angle atan2(|p x q|, p.q), hyperbolic distance
  acosh(a / sqrt(PP QQ)), Grassmann log U atan(S) V^T of
  (I - Q1 Q1^T) Q2 (Q1^T Q2)^-1); on points exactly on the manifold they are
  the textbook formulas.

Usage: python3 scripts/generate_manifold_compute_tier_refs.py
"""
import random
from fractions import Fraction

from mpmath import mp, mpf, sqrt, matrix, expm, logm, sqrtm, cos, sin, cosh, sinh, atan, atan2, log, acosh

mp.dps = 160
SEED = 20260925
SPLITS = [8, 10, 12, 16, 20, 24, 32, 64, 128, 256]
IN_DECIMALS = 100
DYADIC_DIGITS = 90


def fixed(x, dp):
    """x rounded to dp decimals, as a literal (x an mpf or Fraction)."""
    if isinstance(x, Fraction):
        n = x * 10 ** dp
        k = (n.numerator * 2 + n.denominator) // (2 * n.denominator)
    else:
        k = int(mp.floor(x * mpf(10) ** dp + mpf(1) / 2))
    sign = "-" if k < 0 else ""
    s = str(abs(k)).rjust(dp + 1, "0")
    return f"{sign}{s[:-dp]}.{s[-dp:]}"


def dyadic_lit(v):
    """Exact decimal literal of a dyadic Fraction."""
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


def split_dp(f):
    """Decimals that pin a reference well below one unit at F bits."""
    return int(f * 0.30103) + 12


def stored(lit_str, f):
    """The value FixedPoint::from_str stores: nearest at F bits, ties +inf."""
    x = Fraction(lit_str)
    n = x * (1 << f)
    k = (2 * n.numerator + n.denominator) // (2 * n.denominator)

    return mpf(k) / (1 << f)


def vl(v, dp):
    return "&[" + ", ".join(f'"{fixed(x, dp)}"' for x in v) + "]"


def ideal(v):
    return [fixed(x, IN_DECIMALS) for x in v]


def slist(v):
    return "&[" + ", ".join(f'"{s}"' for s in v) + "]"


# ----------------------------------------------------------------------------
# Sphere S^2 in R^3
# ----------------------------------------------------------------------------

def unit(v):
    n = sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def gram_tangent(p, rng):
    """A unit vector orthogonal to the unit vector p."""
    r = [mpf(rng.uniform(-1, 1)) for _ in p]
    c = sum(a * b for a, b in zip(r, p))
    return unit([a - c * b for a, b in zip(r, p)])


def sphere_ref(p, q, v):
    pp = sum(x * x for x in p)
    qq = sum(x * x for x in q)
    c = sum(a * b for a, b in zip(p, q))
    s = sqrt(max(pp * qq - c * c, mpf(0)))
    theta = atan2(s, c)
    w = [b - (c / pp) * a for a, b in zip(p, q)]
    nw = sqrt(sum(x * x for x in w))
    log_v = [theta * x / nw for x in w] if nw != 0 else [mpf(0)] * len(p)
    s_pq = [a + b for a, b in zip(p, q)]
    coeff = sum(a * b for a, b in zip(v, s_pq)) / (1 + c)
    pt = [a - coeff * b for a, b in zip(v, s_pq)]
    return theta, log_v, pt


def sphere_exp_ref(p, v):
    th = sqrt(sum(x * x for x in v))
    if th == 0:
        return list(p)
    return [cos(th) * a + sin(th) * b / th for a, b in zip(p, v)]


# ----------------------------------------------------------------------------
# Hyperbolic H^2 (hyperboloid in R^{2,1})
# ----------------------------------------------------------------------------

def mdot(a, b):
    return -a[0] * b[0] + sum(x * y for x, y in zip(a[1:], b[1:]))


def hyp_point(a, phi):
    return [cosh(a), sinh(a) * cos(phi), sinh(a) * sin(phi)]


def hyp_frame(a, phi):
    """Unit radial and angular tangents at hyp_point(a, phi)."""
    r = [sinh(a), cosh(a) * cos(phi), cosh(a) * sin(phi)]
    e = [mpf(0), -sin(phi), cos(phi)]
    return r, e


def hyp_ref(p, q, v):
    al = -mdot(p, q)
    pp = -mdot(p, p)
    qq = -mdot(q, q)
    s = sqrt(max(al * al - pp * qq, mpf(0)))
    d = log((al + s) / sqrt(pp * qq))
    w = [b - (al / pp) * a for a, b in zip(p, q)]
    nw2 = mdot(w, w)
    nw = sqrt(nw2) if nw2 > 0 else mpf(0)
    if nw == 0:
        return d, [mpf(0)] * 3, list(v)
    u = [x / nw for x in w]
    log_v = [d * x for x in u]
    a = mdot(v, u)
    pt = [vi + a * (sinh(d) * pi + (cosh(d) - 1) * ui) for vi, pi, ui in zip(v, p, u)]
    return d, log_v, pt


def hyp_exp_ref(p, v):
    th2 = mdot(v, v)
    if th2 <= 0:
        return list(p)
    th = sqrt(th2)
    return [cosh(th) * a + sinh(th) * b / th for a, b in zip(p, v)]


# ----------------------------------------------------------------------------
# Grassmannian Gr(k, n); frames stored column-major (as the manifold packs them)
# ----------------------------------------------------------------------------

def random_orthogonal(n, rng):
    a = matrix(n, n)
    for i in range(n):
        for j in range(n):
            a[i, j] = mpf(rng.uniform(-1, 1))
    q, _ = mp.qr(a)
    return q


def col_major(m):
    return [m[r, c] for c in range(m.cols) for r in range(m.rows)]


def from_col_major(v, n, k):
    m = matrix(n, k)
    idx = 0
    for c in range(k):
        for r in range(n):
            m[r, c] = v[idx]
            idx += 1
    return m


def thin_svd(m):
    u, s, vt = mp.svd_r(m, full_matrices=False)
    return u, [s[i] for i in range(len(s))], vt


def diag(vals):
    d = matrix(len(vals), len(vals))
    for i, x in enumerate(vals):
        d[i, i] = x
    return d


def grass_log(q1, q2):
    n, k = q1.rows, q1.cols
    m = (mp.eye(n) - q1 * q1.T) * q2 * mp.inverse(q1.T * q2)
    u, s, vt = thin_svd(m)
    th = [atan(x) for x in s]
    return u * diag(th) * vt, th


def grass_pt(q1, h, delta):
    n = q1.rows
    u, s, vt = thin_svd(h)
    v = vt.T
    return (-q1 * v * diag([sin(x) for x in s]) * u.T + u * diag([cos(x) for x in s]) * u.T
            + mp.eye(n) - u * u.T) * delta


def grass_exp(q, delta):
    u, s, vt = thin_svd(delta)
    v = vt.T
    return q * v * diag([cos(x) for x in s]) * vt + u * diag([sin(x) for x in s]) * vt


def grass_geodesic(q1, comp, thetas, v):
    """Q1 V cos(T) V^T + U sin(T) V^T with U = comp (n x k, orthonormal, _|_ Q1)."""
    return q1 * v * diag([cos(t) for t in thetas]) * v.T + comp * diag([sin(t) for t in thetas]) * v.T


# ----------------------------------------------------------------------------
# Dyadic helpers
# ----------------------------------------------------------------------------

def dy(rng, lo=-2, hi=2, bits=2):
    return Fraction(rng.randrange(lo << bits, hi << bits), 1 << bits)


def fr_matrix(rows):
    return [[Fraction(x) for x in r] for r in rows]


def to_mp(m):
    out = matrix(len(m), len(m[0]))
    for i, r in enumerate(m):
        for j, x in enumerate(r):
            out[i, j] = mpf(x.numerator) / x.denominator
    return out


def sym_pack(m):
    n = m.rows
    return [m[i, j] for i in range(n) for j in range(i, n)]


def fr_sym_pack(m):
    n = len(m)
    return [m[i][j] for i in range(n) for j in range(i, n)]


def fr_mul(a, b):
    return [[sum(a[i][t] * b[t][j] for t in range(len(b))) for j in range(len(b[0]))] for i in range(len(a))]


def fr_inv(a):
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


def fr_trace(a):
    return sum(a[i][i] for i in range(len(a)))


def dlit(x):
    if isinstance(x, Fraction):
        if (x.denominator & (x.denominator - 1)) == 0:
            return dyadic_lit(x)
        return fixed(x, DYADIC_DIGITS)
    return fixed(x, DYADIC_DIGITS)


def dvl(v):
    return "&[" + ", ".join(f'"{dlit(x)}"' for x in v) + "]"


def main():
    rng = random.Random(SEED)
    out = []
    w = out.append
    w("// GENERATED by scripts/generate_manifold_compute_tier_refs.py - do not edit.")
    w("// Dyadic tables: exact inputs, references exact or mpmath at 160 digits.")
    w("// Per-split tables: ideal inputs (100 decimals) that the test rounds at the")
    w("// build's split; references evaluated on the stored values for each F.")
    w("")
    w(f"pub const SPLITS: &[u32] = &{SPLITS};")
    w("")

    # ---------------- Sphere ----------------
    sphere_pairs = []  # (p, q, v, close)
    for theta, close in [(mpf("0.001"), True), (mpf("0.01"), True), (mpf("0.03125"), True),
                         (mpf("0.3"), False), (mpf("1.2"), False), (mpf("2.5"), False), (mpf("3.1"), False)]:
        p = unit([mpf(rng.uniform(-1, 1)) for _ in range(3)])
        t = gram_tangent(p, rng)
        q = [cos(theta) * a + sin(theta) * b for a, b in zip(p, t)]
        v = [mpf(rng.uniform(-1, 1)) * x for x in gram_tangent(p, rng)]
        sphere_pairs.append((ideal(p), ideal(q), ideal(v), close))
    sphere_exps = []
    for norm in [mpf("0.001"), mpf("0.01"), mpf("0.5"), mpf("2"), mpf("4")]:
        p = unit([mpf(rng.uniform(-1, 1)) for _ in range(3)])
        v = [norm * x for x in gram_tangent(p, rng)]
        sphere_exps.append((ideal(p), ideal(v)))

    w("/// Sphere S^2: (p, q, v, close) ideal inputs, parsed at the build's split.")
    w("pub const SPHERE_PAIRS: &[(&[&str], &[&str], &[&str], bool)] = &[")
    for p, q, v, c in sphere_pairs:
        w(f"    ({slist(p)}, {slist(q)}, {slist(v)}, {str(c).lower()}),")
    w("];")
    w("/// Sphere exp_map inputs (p, v).")
    w("pub const SPHERE_EXPS: &[(&[&str], &[&str])] = &[")
    for p, v in sphere_exps:
        w(f"    ({slist(p)}, {slist(v)}),")
    w("];")

    # ---------------- Hyperbolic ----------------
    hyp_pairs = []
    for d, close in [(mpf("0.001"), True), (mpf("0.01"), True), (mpf("0.03125"), True),
                     (mpf("0.3"), False), (mpf("1"), False), (mpf("2.5"), False)]:
        a = mpf(rng.uniform(0, 1))
        phi = mpf(rng.uniform(0, 6.28))
        psi = mpf(rng.uniform(0, 6.28))
        p = hyp_point(a, phi)
        r, e = hyp_frame(a, phi)
        u = [cos(psi) * x + sin(psi) * y for x, y in zip(r, e)]
        q = [cosh(d) * x + sinh(d) * y for x, y in zip(p, u)]
        psi2 = mpf(rng.uniform(0, 6.28))
        v = [mpf(rng.uniform(-1, 1)) * (cos(psi2) * x + sin(psi2) * y) for x, y in zip(r, e)]
        hyp_pairs.append((ideal(p), ideal(q), ideal(v), close))
    hyp_exps = []
    for norm in [mpf("0.001"), mpf("0.01"), mpf("0.5"), mpf("1.5"), mpf("2.5")]:
        a = mpf(rng.uniform(0, 1))
        phi = mpf(rng.uniform(0, 6.28))
        psi = mpf(rng.uniform(0, 6.28))
        p = hyp_point(a, phi)
        r, e = hyp_frame(a, phi)
        v = [norm * (cos(psi) * x + sin(psi) * y) for x, y in zip(r, e)]
        hyp_exps.append((ideal(p), ideal(v)))

    w("/// Hyperbolic H^2: (p, q, v, close) ideal inputs, parsed at the build's split.")
    w("pub const HYP_PAIRS: &[(&[&str], &[&str], &[&str], bool)] = &[")
    for p, q, v, c in hyp_pairs:
        w(f"    ({slist(p)}, {slist(q)}, {slist(v)}, {str(c).lower()}),")
    w("];")
    w("/// Hyperbolic exp_map inputs (p, v).")
    w("pub const HYP_EXPS: &[(&[&str], &[&str])] = &[")
    for p, v in hyp_exps:
        w(f"    ({slist(p)}, {slist(v)}),")
    w("];")

    # ---------------- Grassmannian ----------------
    grass_cases = []  # (k, n, q1, q2, delta, close)
    specs = [
        (1, 3, [mpf("0.01")], False, True),
        (1, 3, [mpf("0.7")], True, False),      # negative representative
        (2, 4, [mpf("0.3"), mpf("0.05")], False, False),
        (2, 4, [mpf("1.0"), mpf("0.4")], True, False),
        (2, 4, [mpf("0.02"), mpf("0.008")], False, True),
        (2, 5, [mpf("1.3"), mpf("0.6")], False, False),
    ]
    for k, n, thetas, flip, close in specs:
        o = random_orthogonal(n, rng)
        q1 = matrix(n, k)
        comp = matrix(n, k)
        for r in range(n):
            for c in range(k):
                q1[r, c] = o[r, c]
                comp[r, c] = o[r, k + c]
        vk = random_orthogonal(k, rng)
        q2 = grass_geodesic(q1, comp, thetas, vk)
        # another representative of the same subspace
        rot = random_orthogonal(k, rng)
        if flip:
            rot = -rot
        q2 = q2 * rot
        # tangent at q1 for transport
        y = matrix(n, k)
        for r in range(n):
            for c in range(k):
                y[r, c] = mpf(rng.uniform(-0.5, 0.5))
        delta = (mp.eye(n) - q1 * q1.T) * y
        grass_cases.append((k, n, ideal(col_major(q1)), ideal(col_major(q2)), ideal(col_major(delta)), close))
    grass_exps = []
    for k, n, scale in [(1, 3, mpf("0.01")), (2, 4, mpf("0.5")), (2, 4, mpf("1.4")), (2, 5, mpf("0.05"))]:
        o = random_orthogonal(n, rng)
        q = matrix(n, k)
        for r in range(n):
            for c in range(k):
                q[r, c] = o[r, c]
        y = matrix(n, k)
        for r in range(n):
            for c in range(k):
                y[r, c] = scale * mpf(rng.uniform(-1, 1))
        delta = (mp.eye(n) - q * q.T) * y
        grass_exps.append((k, n, ideal(col_major(q)), ideal(col_major(delta))))

    w("/// Grassmannian: (k, n, q1, q2, tangent at q1, close) ideal inputs, column-major.")
    w("pub const GRASS_CASES: &[(usize, usize, &[&str], &[&str], &[&str], bool)] = &[")
    for k, n, q1, q2, de, c in grass_cases:
        w(f"    ({k}, {n}, {slist(q1)}, {slist(q2)}, {slist(de)}, {str(c).lower()}),")
    w("];")
    w("/// Grassmannian exp_map inputs (k, n, q, tangent).")
    w("pub const GRASS_EXPS: &[(usize, usize, &[&str], &[&str])] = &[")
    for k, n, q, de in grass_exps:
        w(f"    ({k}, {n}, {slist(q)}, {slist(de)}),")
    w("];")
    w("")

    # ---------------- per-split references ----------------
    w("/// Per split F: (F, case, distance, log_map, parallel_transport) of SPHERE_PAIRS.")
    w("pub const SPHERE: &[(u32, usize, &str, &[&str], &[&str])] = &[")
    for f in SPLITS:
        dp = split_dp(f)
        for i, (p, q, v, _) in enumerate(sphere_pairs):
            ps, qs, vs = [stored(x, f) for x in p], [stored(x, f) for x in q], [stored(x, f) for x in v]
            th, lg, pt = sphere_ref(ps, qs, vs)
            w(f'    ({f}, {i}, "{fixed(th, dp)}", {vl(lg, dp)}, {vl(pt, dp)}),')
    w("];")
    w("/// Per split F: (F, case, exp_map) of SPHERE_EXPS.")
    w("pub const SPHERE_EXP: &[(u32, usize, &[&str])] = &[")
    for f in SPLITS:
        dp = split_dp(f)
        for i, (p, v) in enumerate(sphere_exps):
            ps, vs = [stored(x, f) for x in p], [stored(x, f) for x in v]
            w(f"    ({f}, {i}, {vl(sphere_exp_ref(ps, vs), dp)}),")
    w("];")
    w("/// Per split F: (F, case, distance, log_map, parallel_transport) of HYP_PAIRS.")
    w("pub const HYP: &[(u32, usize, &str, &[&str], &[&str])] = &[")
    for f in SPLITS:
        dp = split_dp(f)
        for i, (p, q, v, _) in enumerate(hyp_pairs):
            ps, qs, vs = [stored(x, f) for x in p], [stored(x, f) for x in q], [stored(x, f) for x in v]
            d, lg, pt = hyp_ref(ps, qs, vs)
            w(f'    ({f}, {i}, "{fixed(d, dp)}", {vl(lg, dp)}, {vl(pt, dp)}),')
    w("];")
    w("/// Per split F: (F, case, exp_map) of HYP_EXPS.")
    w("pub const HYP_EXP: &[(u32, usize, &[&str])] = &[")
    for f in SPLITS:
        dp = split_dp(f)
        for i, (p, v) in enumerate(hyp_exps):
            ps, vs = [stored(x, f) for x in p], [stored(x, f) for x in v]
            w(f"    ({f}, {i}, {vl(hyp_exp_ref(ps, vs), dp)}),")
    w("];")
    w("/// Per split F: (F, case i, sqrt(d_S(i)^2 + d_H(i)^2)) for the product Sphere x Hyperbolic")
    w("/// on SPHERE_PAIRS[i] and HYP_PAIRS[i].")
    w("pub const PRODUCT_DISTANCE: &[(u32, usize, &str)] = &[")
    for f in SPLITS:
        dp = split_dp(f)
        for i in range(min(len(sphere_pairs), len(hyp_pairs))):
            sp, hp = sphere_pairs[i], hyp_pairs[i]
            ds = sphere_ref([stored(x, f) for x in sp[0]], [stored(x, f) for x in sp[1]], [stored(x, f) for x in sp[2]])[0]
            dh = hyp_ref([stored(x, f) for x in hp[0]], [stored(x, f) for x in hp[1]], [stored(x, f) for x in hp[2]])[0]
            w(f'    ({f}, {i}, "{fixed(sqrt(ds * ds + dh * dh), dp)}"),')
    w("];")
    w("/// Per split F: (F, case, distance, log_map, parallel_transport) of GRASS_CASES.")
    w("pub const GRASS: &[(u32, usize, &str, &[&str], &[&str])] = &[")
    for f in SPLITS:
        dp = split_dp(f)
        for i, (k, n, q1, q2, de, _) in enumerate(grass_cases):
            m1 = from_col_major([stored(x, f) for x in q1], n, k)
            m2 = from_col_major([stored(x, f) for x in q2], n, k)
            md = from_col_major([stored(x, f) for x in de], n, k)
            h, th = grass_log(m1, m2)
            dist = sqrt(sum(t * t for t in th))
            pt = grass_pt(m1, h, md)
            w(f'    ({f}, {i}, "{fixed(dist, dp)}", {vl(col_major(h), dp)}, {vl(col_major(pt), dp)}),')
    w("];")
    w("/// Per split F: (F, case, exp_map) of GRASS_EXPS.")
    w("pub const GRASS_EXP: &[(u32, usize, &[&str])] = &[")
    for f in SPLITS:
        dp = split_dp(f)
        for i, (k, n, q, de) in enumerate(grass_exps):
            mq = from_col_major([stored(x, f) for x in q], n, k)
            md = from_col_major([stored(x, f) for x in de], n, k)
            w(f"    ({f}, {i}, {vl(col_major(grass_exp(mq, md)), dp)}),")
    w("];")
    w("")

    # ---------------- SPD (dyadic) ----------------
    spd_cases = []
    for n in [2, 2, 2, 3, 3]:
        while True:
            b = [[dy(rng) for _ in range(n)] for _ in range(n)]
            bt = [list(r) for r in zip(*b)]
            p = fr_mul(b, bt)
            for i in range(n):
                p[i][i] += Fraction(1, 2)
            b2 = [[dy(rng) for _ in range(n)] for _ in range(n)]
            q = fr_mul(b2, [list(r) for r in zip(*b2)])
            for i in range(n):
                q[i][i] += Fraction(1, 2)
            if max(abs(x) for r in p + q for x in r) <= 6:
                break
        def rsym():
            m = [[Fraction(0)] * n for _ in range(n)]
            for i in range(n):
                for j in range(i, n):
                    m[i][j] = m[j][i] = dy(rng, -1, 1, 3)
            return m
        u, v = rsym(), rsym()
        pinv = fr_inv(p)
        inner = fr_trace(fr_mul(fr_mul(pinv, u), fr_mul(pinv, v)))
        norm_sq = fr_trace(fr_mul(fr_mul(pinv, v), fr_mul(pinv, v)))
        pm, qm, vm = to_mp(p), to_mp(q), to_mp(v)
        s = sqrtm(pm)
        si = mp.inverse(s)
        ex = s * expm(si * vm * si) * s
        lg = s * logm(si * qm * si) * s
        lin = logm(si * qm * si)
        dist = sqrt(sum(lin[i, j] ** 2 for i in range(n) for j in range(n)))
        e = sqrtm(qm * mp.inverse(pm))
        pt = e * vm * e.T
        spd_cases.append((n, fr_sym_pack(p), fr_sym_pack(q), fr_sym_pack(u), fr_sym_pack(v),
                          inner, sym_pack(ex), sym_pack(lg), dist, sym_pack(pt), sqrt(mpf(norm_sq.numerator) / norm_sq.denominator)))
    w("/// SPD (dyadic): (n, P, Q, U, V, <U,V>_P, exp_P(V), log_P(Q), d(P,Q), PT_{P->Q}(V), ||V||_P),")
    w("/// matrices packed as the upper triangle, row-major.")
    w("pub const SPD: &[(usize, &[&str], &[&str], &[&str], &[&str], &str, &[&str], &[&str], &str, &[&str], &str)] = &[")
    for n, p, q, u, v, inner, ex, lg, dist, pt, nrm in spd_cases:
        w(f'    ({n}, {dvl(p)}, {dvl(q)}, {dvl(u)}, {dvl(v)}, "{dlit(inner)}", {dvl(ex)}, {dvl(lg)}, "{dlit(dist)}", {dvl(pt)}, "{dlit(nrm)}"),')
    w("];")

    # ---------------- Stiefel (dyadic) ----------------
    stiefel_cases = []
    for k, n, q, q2 in [
        (1, 3, [[1], [0], [0]], [["0.96875"], ["0.25"], ["0.0625"]]),
        (2, 3, [[1, 0], [0, 1], [0, 0]], [["0.9375", "-0.25"], ["0.25", "0.9375"], ["0.125", "0.25"]]),
        (2, 4, [["0.5", "0.5"], ["0.5", "-0.5"], ["0.5", "0.5"], ["0.5", "-0.5"]],
               [["0.59375", "0.40625"], ["0.40625", "-0.5625"], ["0.5", "0.5"], ["0.46875", "-0.5"]]),
        (1, 2, [["0.6"], ["0.8"]], [["0.609375"], ["0.79296875"]]),
    ]:
        qf = fr_matrix([[Fraction(x) for x in r] for r in q])
        q2f = fr_matrix([[Fraction(x) for x in r] for r in q2])
        # 0.6 / 0.8 are not dyadic: replace by their 8-bit dyadic neighbours
        qf = [[Fraction(round(x * 256), 256) for x in r] for r in qf]
        d = [[b - a for a, b in zip(r1, r2)] for r1, r2 in zip(qf, q2f)]
        qtd = fr_mul([list(r) for r in zip(*qf)], d)
        sym = [[(qtd[i][j] + qtd[j][i]) / 2 for j in range(k)] for i in range(k)]
        lg = [[d[r][c] - sum(qf[r][t] * sym[t][c] for t in range(k)) for c in range(k)] for r in range(n)]
        fro = sum(x * x for r in lg for x in r)
        cm = lambda m: [m[r][c] for c in range(k) for r in range(n)]
        stiefel_cases.append((k, n, cm(qf), cm(q2f), cm(lg), sqrt(mpf(fro.numerator) / fro.denominator)))
    w("/// Stiefel (dyadic): (k, n, Q, Q', log_Q(Q'), d(Q, Q')), column-major.")
    w("pub const STIEFEL: &[(usize, usize, &[&str], &[&str], &[&str], &str)] = &[")
    for k, n, q, q2, lg, dist in stiefel_cases:
        w(f'    ({k}, {n}, {dvl(q)}, {dvl(q2)}, {dvl(lg)}, "{dlit(dist)}"),')
    w("];")

    # ---------------- norms (dyadic) ----------------
    norms = []
    for n in [3, 4, 6]:
        v = [dy(rng, -4, 4, 8) for _ in range(n)]
        ss = sum(x * x for x in v)
        norms.append((v, sqrt(mpf(ss.numerator) / ss.denominator)))
    small = [Fraction(1, 256), Fraction(-3, 256), Fraction(2, 256)]
    ss = sum(x * x for x in small)
    norms.append((small, sqrt(mpf(ss.numerator) / ss.denominator)))
    w("/// Euclidean norms (dyadic): (v, ||v||), for Sphere, Euclidean, Grassmannian and Stiefel norm.")
    w("pub const NORMS: &[(&[&str], &str)] = &[")
    for v, r in norms:
        w(f'    ({dvl(v)}, "{dlit(r)}"),')
    w("];")

    # ---------------- product inner product (dyadic) ----------------
    prods = []
    for _ in range(4):
        u = [dy(rng, -2, 2, 8) for _ in range(6)]
        v = [dy(rng, -2, 2, 8) for _ in range(6)]
        r = sum(a * b for a, b in zip(u[:3], v[:3])) - u[3] * v[3] + sum(a * b for a, b in zip(u[4:], v[4:]))
        prods.append((u, v, r))
    w("/// Product Sphere(2) x Hyperbolic(2) inner product (dyadic): (u, v, u1.v1 + <u2, v2>_M).")
    w("pub const PRODUCT_INNER: &[(&[&str], &[&str], &str)] = &[")
    for u, v, r in prods:
        w(f'    ({dvl(u)}, {dvl(v)}, "{dlit(r)}"),')
    w("];")

    path = "tests/data/manifold_compute_tier_refs.rs"
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
