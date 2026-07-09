# Linear algebra

Decompositions, derived quantities, and matrix functions over `FixedMatrix`, each
chained at the wide compute tier.

## What it is

Three modules build on [`FixedMatrix`](README_IMPERATIVE.md): `decompose`
(factorizations and the solvers built on them), `derived` (norms, inverses,
condition numbers, least squares), and `matrix_functions` (`exp`, `log`, `sqrt`,
`pow` of a matrix). Operation chains run through a `ComputeMatrix` at tier N+1 so
the whole chain rounds once at the end rather than once per intermediate.

## Usage

```rust
use g_math::fixed_point::{FixedMatrix, FixedVector, FixedPoint};
use g_math::fixed_point::imperative::decompose;

let a = FixedMatrix::from_slice(2, 2, &[
    FixedPoint::from_int(4), FixedPoint::from_int(3),
    FixedPoint::from_int(6), FixedPoint::from_int(3),
]);
let lu = decompose::lu_decompose(&a).unwrap();   // Doolittle, partial pivoting
let b = FixedVector::from_slice(&[FixedPoint::from_int(1), FixedPoint::from_int(2)]);
let x = lu.solve(&b).unwrap();                    // solve A x = b
let det = lu.determinant();
```

## What's here

- **Decompositions** (`decompose`): LU (Doolittle, partial pivoting), QR
  (Householder), Cholesky, SVD (Golub-Kahan), symmetric eigenvalues (Jacobi),
  Schur (Francis QR). Each returns a struct exposing `solve` / `determinant` /
  `inverse` where applicable, plus iterative refinement on LU.
- **Derived** (`derived`): `frobenius_norm`, `norm_1`, `norm_inf`, `solve`,
  `solve_spd`, `determinant`, `inverse`, `inverse_spd`, `pseudoinverse`, `rank`,
  `nullspace`, `least_squares`, `condition_number_1` / `_2`.
- **Matrix functions** (`matrix_functions`): `matrix_exp` (Padé +
  scaling-and-squaring), `matrix_sqrt` (Denman-Beavers), `matrix_log` (inverse
  scaling-and-squaring), `matrix_pow` — all chained through `ComputeMatrix`.

## Public API

See **[PUBLIC_API.md → Linear algebra](../PUBLIC_API.md#linear-algebra)** and
[docs.rs](https://docs.rs/g_math).

## Behaviour & limits

- Solver and factor accuracy is validated against mpmath references combined with
  structural checks (PA=LU, QᵀQ=I, exp/log roundtrips).
- Error in a solved system scales with the condition number of the matrix. An
  ill-conditioned system (e.g. a Hilbert matrix) amplifies input error by orders
  of magnitude in any finite precision; iterative refinement recovers the residual
  but not the lost input information. See [the precision guide](README_PRECISION.md).

Determinism guarantees are in **[CONTRACT.md](../CONTRACT.md)**.

## Disclaimer

This software is provided **"as is"**, without warranty of any kind, express or
implied. Use of this software is entirely at your own risk. In no event shall the
author or contributors be held liable for any damages arising from the use or
inability to use this software.

---

Built by **Niels Erik Toren** — [support & donations](../README.md#author--support).
