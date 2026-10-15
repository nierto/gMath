#!/usr/bin/env python3
"""mpmath references for canonical expression chains on the binary formats:
random trees of depth 2 to 4 over the 17 functions and + - * /, with binary
leaves (raws), small integer literals and pi. The expected value is the exact
result of the exact leaves rounded once to the storage raw (nearest, ties
toward +infinity). Every intermediate is kept inside the format's storage
range and inside its function's domain, so the gate measures rounding, not
overflow handling.

Tree text (prefix): "(op a b)", "#<raw>" a binary leaf, "i<int>" an integer
literal, "pi". Writes tests/data/chain_refs.rs, one table per format and a
REFS facade selected by the including module's BITS and FRAC_BITS.
"""
import os, random
from mpmath import mp, mpf, exp, log, sqrt, sin, cos, tan, atan, atan2, asin, acos, sinh, cosh, tanh, asinh, acosh, atanh, floor, pi

mp.dps = 600
SEED = int(os.environ.get("SEED", "20261015"))
PER_FORMAT = int(os.environ.get("PER_FORMAT", "300"))
FORMATS = [(32, 10), (32, 16), (32, 24), (64, 32), (128, 64), (256, 128), (512, 256)]

UNARY = {
    "exp": exp, "ln": log, "sqrt": sqrt, "sin": sin, "cos": cos, "tan": tan, "atan": atan,
    "asin": asin, "acos": acos, "sinh": sinh, "cosh": cosh, "tanh": tanh, "asinh": asinh,
    "acosh": acosh, "atanh": atanh, "neg": lambda v: -v,
}
BINARY = {"add": lambda a, b: a + b, "sub": lambda a, b: a - b, "mul": lambda a, b: a * b,
          "div": lambda a, b: a / b, "pow": lambda a, b: a ** b, "atan2": atan2}


EDGE = mpf(10) ** -500   # below the reference's own precision (600 digits)


def nearest_raw(v, f):
    """Nearest raw, ties toward +infinity; a value within EDGE of a tie is
    the tie (an algebraic identity such as ln(sqrt(exp(x))) = x/2)."""
    s = v * (mpf(2) ** f)
    fl = floor(s)
    n = int(fl)
    if s - fl >= mpf("0.5") - EDGE:
        n += 1
    return n


class Gen:
    def __init__(self, rng, w, f):
        self.rng, self.w, self.f = rng, w, f
        self.vmax = mpf(2) ** (w - f - 1)
        self.ib = w - f - 1

    def ok(self, v):
        # inside storage with a factor 2 of headroom, and not below 2^-(f-2) unless zero
        return abs(v) < self.vmax / 2 and (v == 0 or abs(v) > mpf(2) ** (-(self.f - 2)))

    def leaf(self):
        r = self.rng.random()
        if r < 0.1:
            return "pi", +pi
        if r < 0.25:
            k = self.rng.randint(-9, 9)
            return f"i{k}", mpf(k)
        e = self.rng.uniform(-min(self.f - 2, 8), min(self.ib - 1, 12))
        raw = int(mpf(2) ** e * mpf(2) ** self.f) or 1
        if self.rng.random() < 0.4:
            raw = -raw
        return f"#{raw}", mpf(raw) / mpf(2) ** self.f

    def tree(self, depth):
        if depth == 0:
            return self.leaf()
        for _ in range(40):
            if self.rng.random() < 0.6:
                name = self.rng.choice(list(UNARY))
                a = self.tree(depth - 1)
                if a is None:
                    continue
                try:
                    v = UNARY[name](a[1])
                except Exception:
                    continue
                if hasattr(v, "imag") and v.imag != 0:
                    continue
                # domain edges closer than the reference can resolve are skipped
                if name in ("asin", "acos", "atanh") and abs(a[1]) >= 1 - EDGE:
                    continue
                if name == "acosh" and a[1] < 1 + EDGE:
                    continue
                if name in ("ln", "sqrt") and a[1] <= EDGE:
                    continue
                if name in ("sin", "cos", "tan") and abs(a[1]) > 2 ** 20:
                    continue
                if name == "tan" and abs(cos(a[1])) < mpf(2) ** -(self.f // 2):
                    continue
                v = mpf(v.real) if hasattr(v, "real") else v
                if self.ok(v):
                    return f"({name} {a[0]})", v
            else:
                name = self.rng.choice(list(BINARY))
                a = self.tree(depth - 1)
                b = self.tree(self.rng.randint(0, depth - 1))
                if a is None or b is None:
                    continue
                if name == "div" and abs(b[1]) < mpf(2) ** -(self.f // 2):
                    continue
                if name == "pow" and (a[1] <= EDGE or abs(b[1]) > 64):
                    continue
                if name == "atan2" and a[1] == 0 and b[1] == 0:
                    continue
                try:
                    v = BINARY[name](a[1], b[1])
                except Exception:
                    continue
                if hasattr(v, "imag") and v.imag != 0:
                    continue
                if self.ok(v):
                    return f"({name} {a[0]} {b[0]})", v
        return None


def corpus(w, f, rng):
    g = Gen(rng, w, f)
    rows, seen = [], set()
    while len(rows) < PER_FORMAT:
        t = g.tree(rng.randint(2, 4))
        if t is None or t[0] in seen or "(" not in t[0]:
            continue
        seen.add(t[0])
        rows.append((t[0], str(nearest_raw(t[1], f))))
    return rows


def main():
    rng = random.Random(SEED)
    out = ["// GENERATED by scripts/generate_chain_refs.py (mpmath, 600 digits). Do not edit.",
           "// (tree, expected_raw): the expected raw is the exact value of the tree on its exact",
           "// leaves rounded once to storage, nearest, ties toward +infinity.", ""]
    names = []
    for w, f in FORMATS:
        rows = corpus(w, f, rng)
        name = f"REFS_Q{w - f}_{f}"
        names.append((w, f, name))
        out.append(f"/// Q{w - f}.{f} on the {w}-bit word: {len(rows)} trees.")
        out.append(f"pub const {name}: &[(&str, &str)] = &[")
        out += [f'    ("{t}", "{r}"),' for t, r in rows]
        out += ["];", ""]
    arms = " ".join(f"({w}, {f}) => {n}," for w, f, n in names)
    out.append("/// The table of the including module's format (it defines BITS and FRAC_BITS).")
    out.append(f"pub const REFS: &[(&str, &str)] = match (BITS, FRAC_BITS) {{ {arms} _ => &[] }};")
    path = os.path.join(os.path.dirname(__file__), "..", "tests", "data", "chain_refs.rs")
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print("wrote", os.path.normpath(path))


if __name__ == "__main__":
    main()
