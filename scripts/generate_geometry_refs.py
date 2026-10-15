#!/usr/bin/env python3
"""mpmath references for the closed-form manifold maps on the binary formats:
Sphere and HyperbolicSpace distance, log_map, exp_map and parallel_transport
on stored inputs (random raws; points need not lie exactly on the manifold,
the formulas define the exact value for any stored input). Each output
component is the exact value rounded once to storage (nearest, ties toward
+infinity). Writes tests/data/geometry_refs.rs: per format a table of
(op, inputs, outputs) with raws as decimal strings, `;` between vectors.
"""
import os, random
from mpmath import mp, mpf, sqrt, atan2, sin, cos, sinh, cosh, log, floor

mp.dps = 400
SEED = int(os.environ.get("SEED", "20261016"))
PER_OP = int(os.environ.get("PER_OP", "60"))
FORMATS = [(32, 10), (32, 16), (32, 24), (64, 32), (128, 64), (256, 128), (512, 256)]
DIM = 3   # ambient dimension
EDGE = mpf(10) ** -300


def nearest_raw(v, f):
    s = v * (mpf(2) ** f)
    fl = floor(s)
    n = int(fl)
    if s - fl >= mpf("0.5") - EDGE:
        n += 1
    return n


def fits(r, w):
    return -(1 << (w - 1)) <= r <= (1 << (w - 1)) - 1


def val(r, f):
    return mpf(r) / mpf(2) ** f


def rnd(rng, f, lo, hi):
    return rng.randint(int(lo * 2 ** f), int(hi * 2 ** f))


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def mdot(a, b):
    return -a[0] * b[0] + dot(a[1:], b[1:])


def sphere(op, P, Q, V):
    pp, qq, c = dot(P, P), dot(Q, Q), dot(P, Q)
    if op == "s_exp":
        th = sqrt(dot(V, V))
        if th == 0:
            return None
        return [cos(th) * p + sin(th) / th * v for p, v in zip(P, V)]
    if op == "s_tr":
        den = 1 + c
        if den == 0:
            return None
        k = (dot(V, P) + dot(V, Q)) / den
        return [v - k * (p + q) for p, q, v in zip(P, Q, V)]
    cross2 = pp * qq - c * c
    if pp == 0 or qq == 0 or cross2 <= 0:
        return None
    th = atan2(sqrt(cross2), c)
    if op == "s_dist":
        return [th]
    w = [q - c / pp * p for p, q in zip(P, Q)]
    wl = sqrt(dot(w, w))
    return [th * x / wl for x in w]


def hyper(op, P, Q, V):
    if op == "h_exp":
        t2 = mdot(V, V)
        if t2 == 0:
            return None
        th = sqrt(abs(t2))
        return [cosh(th) * p + sinh(th) / th * v for p, v in zip(P, V)]
    a, pp, qq = -mdot(P, Q), -mdot(P, P), -mdot(Q, Q)
    if not (a > 0 and pp > 0 and qq > 0) or a * a - pp * qq <= EDGE:
        return None
    d = log((a + sqrt(a * a - pp * qq)) / sqrt(pp * qq))
    if op == "h_dist":
        return [d]
    w = [q - a / pp * p for p, q in zip(P, Q)]
    wl = sqrt(mdot(w, w))
    if op == "h_log":
        return [d * x / wl for x in w]
    u = [x / wl for x in w]
    k = mdot(V, u)
    return [v + k * (sinh(d) * p + (cosh(d) - 1) * ui) for p, v, ui in zip(P, V, u)]


# ---- SO(3) and SE(3) -------------------------------------------------------
from mpmath import tan, matrix

def hat3(w):
    return [[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]]

def mm(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(3)) for j in range(3)] for i in range(3)]

def mv(a, v):
    return [sum(a[i][k] * v[k] for k in range(3)) for i in range(3)]

def tr3(a):
    return [[a[j][i] for j in range(3)] for i in range(3)]

def poly(k, a, b):
    k2 = mm(k, k)
    return [[(1 if i == j else 0) + a * k[i][j] + b * k2[i][j] for j in range(3)] for i in range(3)]

def rod_exp(w):
    t2 = sum(x * x for x in w)
    if t2 == 0:
        return poly(hat3(w), 1, mpf(1) / 2)
    th = sqrt(t2)
    return poly(hat3(w), sin(th) / th, (1 - cos(th)) / t2)

BRANCH = mpf(10) ** -50

def rod_log(r):
    d = [r[2][1] - r[1][2], r[0][2] - r[2][0], r[1][0] - r[0][1]]
    trm1 = r[0][0] + r[1][1] + r[2][2] - 1
    if abs(trm1) < BRANCH and trm1 != 0:
        return None
    dn2 = sum(x * x for x in d)
    if trm1 >= 0:
        if dn2 == 0:
            return [mpf(0)] * 3, mpf(0)
        dn = sqrt(dn2)
        th = atan2(dn, trm1)
        return [th * x / dn for x in d], th
    th = atan2(sqrt(dn2), trm1)
    cos_t = trm1 / 2
    omc = 1 - cos_t
    i = 0
    for k in (1, 2):
        if r[i][i] < r[k][k]:
            i = k
    ui2 = (r[i][i] - cos_t) / omc
    if ui2 <= BRANCH:
        return None
    ui = sqrt(ui2)
    den = 2 * omc * ui
    u = [ui if k == i else (r[i][k] + r[k][i]) / den for k in range(3)]
    flip = False
    if dn2 != 0:
        dot = sum(a * b for a, b in zip(u, d))
        if abs(dot) < BRANCH:
            return None
        flip = dot < 0
    return [(-th if flip else th) * x for x in u], th

def se3_exp(xi):
    w = xi[:3]
    t2 = sum(x * x for x in w)
    k = hat3(w)
    if t2 == 0:
        c1, c2 = mpf(1) / 2, mpf(1) / 6
        r = poly(k, 1, c1)
    else:
        th = sqrt(t2)
        c1 = (1 - cos(th)) / t2
        c2 = (th - sin(th)) / (t2 * th)
        r = poly(k, sin(th) / th, c1)
    return r, mv(poly(k, c1, c2), xi[3:])

def se3_log(r, t):
    res = rod_log(r)
    if res is None:
        return None
    w, th = res
    c = mpf(1) / 12 if th == 0 else (1 - (th / 2) / tan(th / 2)) / (th * th)
    return w + mv(poly(hat3(w), -mpf(1) / 2, c), t)

def lie(op, A, B, C):
    if op == "so3_exp":
        return [x for row in rod_exp(A) for x in row]
    if op == "so3_log":
        r = [A[0:3], A[3:6], A[6:9]]
        res = rod_log(r)
        return None if res is None else res[0]
    if op.startswith("so3_"):
        ga, gb = rod_exp(A), rod_exp(B)
        g = mm(ga, gb) if op == "so3_expmap" else mm(tr3(ga), gb)
        res = rod_log(g)
        if res is None:
            return None
        w = res[0]
        if op == "so3_dist":
            return [sqrt(sum(x * x for x in w))]
        if op == "so3_tr":
            return mv(rod_exp([x / 2 for x in w]), C)
        return w
    if op == "se3_exp":
        r, t = se3_exp(A)
        return [r[0][0], r[0][1], r[0][2], t[0], r[1][0], r[1][1], r[1][2], t[1], r[2][0], r[2][1], r[2][2], t[2], 0, 0, 0, 1]
    if op == "se3_log":
        r = [A[0:3], A[4:7], A[8:11]]
        return se3_log(r, [A[3], A[7], A[11]])
    ra, ta = se3_exp(A)
    rb, tb = se3_exp(B)
    if op == "se3_expmap":
        r, t = mm(ra, rb), [x + y for x, y in zip(mv(ra, tb), ta)]
    else:
        rt = tr3(ra)
        ti = [-x for x in mv(rt, ta)]
        r, t = mm(rt, rb), [x + y for x, y in zip(mv(rt, tb), ti)]
    xi = se3_log(r, t)
    if xi is None:
        return None
    return [sqrt(sum(x * x for x in xi))] if op == "se3_dist" else xi

RAT_OPS = ["mob", "st_proj", "st_unproj", "cross"]

def rational_op(op, A):
    if op == "mob":
        a, b, c, d, x = A
        den = c * x + d
        return None if den == 0 else [(a * x + b) / den]
    if op == "st_proj":
        den = 1 - A[-1]
        return None if den == 0 else [t / den for t in A[:-1]]
    if op == "st_unproj":
        x2 = sum(t * t for t in A)
        return [2 * t / (1 + x2) for t in A] + [(x2 - 1) / (x2 + 1)]
    a, b, c, d = A
    den = (a - d) * (b - c)
    return None if den == 0 else [(a - c) * (b - d) / den]

LIE_OPS = ["so3_exp", "so3_log", "so3_expmap", "so3_logmap", "so3_dist", "so3_tr", "se3_exp", "se3_log", "se3_expmap", "se3_logmap", "se3_dist"]


def corpus(w, f, rng):
    rows = []
    ib = w - f - 1
    span = mpf(2) ** min(ib - 3, 3)          # coordinates up to 8, less on narrow formats
    for op in ["s_dist", "s_log", "s_exp", "s_tr", "h_dist", "h_log", "h_exp", "h_tr"] + LIE_OPS + RAT_OPS:
        made = 0
        tries = 0
        while made < PER_OP and tries < PER_OP * 20:
            tries += 1
            if op in RAT_OPS:
                n = {"mob": 5, "st_proj": 3, "st_unproj": 2, "cross": 4}[op]
                A = [rnd(rng, f, -span, span) for _ in range(n)]
                if op == "st_unproj":
                    A = [rnd(rng, f, -span / 4, span / 4) for _ in range(n)]
                P, Q, V = A, [0], [0]
                out = rational_op(op, [val(x, f) for x in A])
            elif op in LIE_OPS:
                lim = min(span, 3)
                A = [rnd(rng, f, -lim, lim) for _ in range(6)]
                B = [rnd(rng, f, -lim, lim) for _ in range(6)]
                C = [rnd(rng, f, -lim, lim) for _ in range(3)]
                if op.startswith("so3"):
                    A, B = A[:3], B[:3]
                if op == "so3_log":
                    A = [nearest_raw(x, f) for row in rod_exp([val(t, f) for t in A[:3]]) for x in row]
                if op == "se3_log":
                    r, t = se3_exp([val(x, f) for x in A])
                    A = [nearest_raw(x, f) for x in [r[0][0], r[0][1], r[0][2], t[0], r[1][0], r[1][1], r[1][2], t[1], r[2][0], r[2][1], r[2][2], t[2], 0, 0, 0, 1]]
                if any(not fits(t, w) for t in A):
                    continue
                P, Q, V = A, B, C
                out = lie(op, [val(x, f) for x in A], [val(x, f) for x in B], [val(x, f) for x in C])
            elif op.startswith("s_"):
                P = [rnd(rng, f, -span, span) for _ in range(DIM)]
                Q = [rnd(rng, f, -span, span) for _ in range(DIM)]
                V = [rnd(rng, f, -span / 2, span / 2) for _ in range(DIM)]
                out = sphere(op, [val(x, f) for x in P], [val(x, f) for x in Q], [val(x, f) for x in V])
            else:
                def point():
                    x = [rnd(rng, f, -span / 2, span / 2) for _ in range(DIM - 1)]
                    x0 = sqrt(1 + sum(val(t, f) ** 2 for t in x))
                    return [nearest_raw(x0, f)] + x
                P, Q = point(), point()
                V = [rnd(rng, f, -span / 4, span / 4) for _ in range(DIM)]
                if any(not fits(t, w) for t in P + Q):
                    continue
                out = hyper(op, [val(x, f) for x in P], [val(x, f) for x in Q], [val(x, f) for x in V])
            if out is None:
                continue
            raws = [nearest_raw(o, f) for o in out]
            if not all(fits(r, w) for r in raws):
                continue
            ins = ";".join(",".join(str(t) for t in vec) for vec in (P, Q, V))
            rows.append((op, ins, ",".join(str(r) for r in raws)))
            made += 1
    return rows


def main():
    rng = random.Random(SEED)
    out = ["// GENERATED by scripts/generate_geometry_refs.py (mpmath, 400 digits). Do not edit.",
           "// (op, \"p;q;v\" input raws, output raws): each output is the exact value of the",
           "// map on the stored inputs rounded once, nearest, ties toward +infinity.", ""]
    names = []
    for w, f in FORMATS:
        rows = corpus(w, f, rng)
        name = f"REFS_Q{w - f}_{f}"
        names.append((w, f, name))
        out.append(f"/// Q{w - f}.{f}: {len(rows)} cases.")
        out.append(f"pub const {name}: &[(&str, &str, &str)] = &[")
        out += [f'    ("{a}", "{b}", "{c}"),' for a, b, c in rows]
        out += ["];", ""]
    arms = " ".join(f"({w}, {f}) => {n}," for w, f, n in names)
    out.append("/// The table of the including module's format.")
    out.append(f"pub const REFS: &[(&str, &str, &str)] = match (BITS, FRAC_BITS) {{ {arms} _ => &[] }};")
    path = os.path.join(os.path.dirname(__file__), "..", "tests", "data", "geometry_refs.rs")
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print("wrote", os.path.normpath(path))


if __name__ == "__main__":
    main()
