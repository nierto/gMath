#!/usr/bin/env python3
"""References for the curvature compute-tier gate
(tests/data/curvature_compute_tier_refs.rs).

Each reference is the SAME scheme the library runs, evaluated exactly:
central differences with the build's step h = 2^-k for the metric partials
and for the Christoffel derivatives, the exact inverse metric, and the exact
contractions. The difference the test measures is rounding alone, not the
O(h^2) discretization error.

* POLY: the metric g = [[1 + x^2, x y / 2], [x y / 2, 1 + y^2]] (no closed
  forms, so Christoffel symbols come from differences of g and the default
  LU inverse). At the gated points every metric entry the scheme evaluates
  is a dyadic with at most 2k + 1 fraction bits, stored exactly on every
  gated build, so the reference is exact rational arithmetic (Fractions).
* SPHERE: SphereMetric { radius: 1.5 } at theta = 0.75, whose Christoffel
  symbols are closed forms; Riemann, Ricci and sectional curvature are
  central differences of the exact closed forms, in mpmath at 300 digits.
* CLOSED: the closed forms themselves (sphere metric, Christoffel symbols
  and scalar curvature; hyperbolic metric), mpmath at 300 digits.

k is FRAC_BITS / 3 rounded (realtime (F + 1) / 3), 11 on compact, 21 on
embedded, 43 on balanced and 85 on scientific. Tables are emitted for the
realtime splits 8 to 30 (k = 3..10) and the wide profiles. The test parses
each literal with the exact literal parser, which rounds it to nearest at the
build's split.

Usage: python3 scripts/generate_curvature_compute_tier_refs.py
"""
from fractions import Fraction as Q

from mpmath import mp, mpf, sin, cos

mp.dps = 300
DIGITS = 120
OUT = "tests/data/curvature_compute_tier_refs.rs"
KS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 43, 85]

POINTS = [(Q(1, 2), Q(1, 4)), (Q(-3, 4), Q(1, 2))]
PAIRS = [((Q(1), Q(0)), (Q(0), Q(1))), ((Q(1), Q(1, 2)), (Q(1), Q(9, 16)))]
RADIUS = Q(3, 2)
THETA = Q(3, 4)
PHI = Q(1, 4)
HYP_Y = Q(3, 4)


def lit(v):
    if isinstance(v, Q):
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
        v = mpf(v.numerator) / v.denominator
    return mp.nstr(v, DIGITS, strip_zeros=False, min_fixed=-10**9, max_fixed=10**9)


def inv2(g):
    det = g[0][0] * g[1][1] - g[0][1] * g[1][0]
    return [[g[1][1] / det, -g[0][1] / det], [-g[1][0] / det, g[0][0] / det]]


def poly_metric(p):
    x, y = p
    return [[1 + x * x, x * y / 2], [x * y / 2, 1 + y * y]]


def shifted(p, k, d):
    q = list(p)
    q[k] = q[k] + d
    return q


def christoffel_fd(metric, p, h):
    n = 2
    ginv = inv2(metric(p))
    dg = []
    for m in range(n):
        gp, gm = metric(shifted(p, m, h)), metric(shifted(p, m, -h))
        dg.append([[(gp[i][j] - gm[i][j]) / (2 * h) for j in range(n)] for i in range(n)])
    return [[[sum(ginv[k][l] * (dg[i][j][l] + dg[j][l][i] - dg[l][i][j]) for l in range(n)) / 2
              for j in range(n)] for i in range(n)] for k in range(n)]


def riemann_fd(gamma, p, h):
    n = 2
    g0 = gamma(p)
    dgam = []
    for j in range(n):
        gp, gm = gamma(shifted(p, j, h)), gamma(shifted(p, j, -h))
        dgam.append([[[(gp[l][i][k] - gm[l][i][k]) / (2 * h) for k in range(n)] for i in range(n)] for l in range(n)])
    return [[[[dgam[j][l][i][k] - dgam[k][l][i][j]
               + sum(g0[l][j][m] * g0[m][i][k] - g0[l][k][m] * g0[m][i][j] for m in range(n))
               for k in range(n)] for j in range(n)] for i in range(n)] for l in range(n)]


def ricci(r):
    return [[sum(r[k][i][k][j] for k in range(2)) for j in range(2)] for i in range(2)]


def sectional(g, r, u, v):
    n = 2
    w = [sum(g[l][s] * u[l] for l in range(n)) for s in range(n)]
    # <R(u, v) v, u> = g_ls R^l_ijk v^i u^j v^k u^s (R^l_ijk is antisymmetric
    # in j, k: contracting v^j v^k, as the library did before 0.6.4, is zero)
    num = sum(r[s][i][j][k] * v[i] * u[j] * v[k] * w[s]
              for s in range(n) for i in range(n) for j in range(n) for k in range(n))
    ip = lambda a, b: sum(a[i] * g[i][j] * b[j] for i in range(n) for j in range(n))
    return num / (ip(u, u) * ip(v, v) - ip(u, v) ** 2)


def flat3(t):
    return [t[a][b][c] for a in range(2) for b in range(2) for c in range(2)]


def flat4(t):
    return [t[a][b][c][d] for a in range(2) for b in range(2) for c in range(2) for d in range(2)]


def m(fr):
    return mpf(fr.numerator) / fr.denominator


def sphere_gamma(p):
    s, c = sin(p[0]), cos(p[0])
    z = mpf(0)
    return [[[z, z], [z, -s * c]], [[z, c / s], [c / s, z]]]


def sphere_metric(p):
    r = m(RADIUS)
    return [[r * r, mpf(0)], [mpf(0), r * r * sin(p[0]) ** 2]]


def arr(vals):
    return "[" + ", ".join(f'"{lit(v)}"' for v in vals) + "]"


def main():
    out = ["// Generated by scripts/generate_curvature_compute_tier_refs.py. Do not edit.\n",
           "/// The polynomial metric's curvature at one point, for one step 2^-k.",
           "pub struct PolyRefs {",
           "    pub k: u32,",
           "    pub point: usize,",
           "    pub christoffel: [&'static str; 8],",
           "    pub riemann: [&'static str; 16],",
           "    pub ricci: [&'static str; 4],",
           "    pub scalar: &'static str,",
           "    pub sectional: [&'static str; 2],",
           "}\n",
           "/// SphereMetric curvature from differences of the closed-form symbols.",
           "pub struct SphereRefs {",
           "    pub k: u32,",
           "    pub riemann: [&'static str; 16],",
           "    pub ricci: [&'static str; 4],",
           "    pub sectional: &'static str,",
           "}\n"]
    out.append("/// Gated points (dyadic).")
    out.append("pub const POINTS: [[&str; 2]; 2] = [" + ", ".join(arr(p) for p in POINTS) + "];")
    out.append("/// Sectional pairs (u, v): orthogonal, and near-parallel.")
    out.append("pub const PAIRS: [[[&str; 2]; 2]; 2] = [" + ", ".join(
        "[" + arr(u) + ", " + arr(v) + "]" for u, v in PAIRS) + "];\n")

    out.append("pub const POLY: &[PolyRefs] = &[")
    for k in KS:
        h = Q(1, 2 ** k)
        for pi, p in enumerate(POINTS):
            gam = lambda q: christoffel_fd(poly_metric, q, h)
            g0 = gam(list(p))
            r = riemann_fd(gam, list(p), h)
            ric = ricci(r)
            ginv = inv2(poly_metric(p))
            scal = sum(ginv[i][j] * ric[i][j] for i in range(2) for j in range(2))
            sec = [sectional(poly_metric(p), r, u, v) for u, v in PAIRS]
            out.append(f"    PolyRefs {{ k: {k}, point: {pi}, christoffel: {arr(flat3(g0))}, "
                       f"riemann: {arr(flat4(r))}, ricci: {arr([ric[i][j] for i in range(2) for j in range(2)])}, "
                       f"scalar: \"{lit(scal)}\", sectional: {arr(sec)} }},")
    out.append("];\n")

    out.append("pub const SPHERE: &[SphereRefs] = &[")
    p = [m(THETA), m(PHI)]
    for k in KS:
        h = mpf(1) / mpf(2) ** k
        r = riemann_fd(sphere_gamma, p, h)
        ric = ricci(r)
        sec = sectional(sphere_metric(p), r, [mpf(1), mpf(0)], [mpf(0), mpf(1)])
        out.append(f"    SphereRefs {{ k: {k}, riemann: {arr(flat4(r))}, "
                   f"ricci: {arr([ric[i][j] for i in range(2) for j in range(2)])}, sectional: \"{lit(sec)}\" }},")
    out.append("];\n")

    s, c = sin(m(THETA)), cos(m(THETA))
    rr = m(RADIUS) ** 2
    out.append("/// Sphere radius 1.5 at (theta, phi) = (0.75, 0.25): g_11 = r^2 sin^2, "
               "Gamma^0_11 = -sin cos, Gamma^1_01 = cot, R = 2 / r^2; hyperbolic at y = 0.75: g_00 = 1 / y^2.")
    out.append(f"pub const SPHERE_POINT: [&str; 2] = {arr([THETA, PHI])};")
    out.append(f"pub const SPHERE_RADIUS: &str = \"{lit(RADIUS)}\";")
    out.append(f"pub const SPHERE_G11: &str = \"{lit(rr * s * s)}\";")
    out.append(f"pub const SPHERE_GAMMA_011: &str = \"{lit(-s * c)}\";")
    out.append(f"pub const SPHERE_GAMMA_101: &str = \"{lit(c / s)}\";")
    out.append(f"pub const SPHERE_SCALAR: &str = \"{lit(2 / rr)}\";")
    out.append(f"pub const HYP_POINT: [&str; 2] = {arr([Q(1, 4), HYP_Y])};")
    out.append(f"pub const HYP_G00: &str = \"{lit(1 / m(HYP_Y) ** 2)}\";")
    out.append(f"pub const HYP_GAMMA_100: &str = \"{lit(1 / m(HYP_Y))}\";")

    with open(OUT, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
