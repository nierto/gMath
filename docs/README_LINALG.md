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
  scaling-and-squaring), `matrix_pow`: all chained through `ComputeMatrix`
  (on realtime through a Q64.64 matrix, since the realtime compute tier holds
  only `2F` bits), one rounding per output entry. Against mpmath they are
  correctly rounded on every profile and at realtime 8 to 24 fraction bits,
  including norms up to 7 and SPD spectra from 1/8 to 60.

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
- LU, Cholesky and QR factor at the compute tier (`2F` fractional bits) and keep
  those factors: `solve`, `inverse`, `determinant` and `refine` run on them,
  every sum exact, and round the result once. The public `l`, `u`, `q`, `r`
  fields are the same factors rounded once, for inspection. Against exact
  rationals and mpmath on well-conditioned 2x2 to 4x4 systems, factors,
  solutions, inverses and determinants are within one unit on every profile and
  at realtime 8 to 24 fraction bits (0.6.3 stored the factors at storage
  precision: up to 114 units in a solve, 70 in a determinant). On an
  ill-conditioned system the error before the final rounding grows as
  `kappa * 2^-2F`, so a solve stays within a unit while `kappa` is well below
  `2^F`.
- The iterative decompositions (`svd_decompose`, `eigen_symmetric`,
  `schur_decompose`) carry the matrix being reduced and the accumulated
  transforms at the compute tier and round them to storage once. They either
  converge or return an error, never a partially converged result:
  `Err(PrecisionLimit)` when the iteration budget runs out, `Err(TierOverflow)`
  when a norm or an entry leaves the range. An off-diagonal entry counts as
  zero within `2^-(3F/2)` of its diagonal neighbours, never below `2^-(3F/2)`
  absolute, so exactly rank-deficient matrices converge. Against mpmath on
  well-separated spectra, eigenvalues and eigenvectors, singular values and
  vectors, and Schur eigenvalues are within one unit on every profile and
  split (0.6.3 converged to `2^-(2F/3)`: up to 1483 units at Q16.16 and past
  `2^30` units on the wide profiles). A Schur eigenvalue's error is still the
  deflation bound times the eigenvalue's condition number, and a vector of a
  nearly repeated eigen- or singular value is only as well determined as the
  gap allows. `schur_decompose` returns a real Schur form: exact zeros below
  the subdiagonal, and 2×2 blocks
  only for complex pairs. Gate: `tests/decomposition_convergence_validation.rs`,
  37 fixed cases plus a seeded random corpus from the same failure classes
  (rank-deficient, interior zero diagonals, rectangular, scaled and small
  entries, repeated spectra, signed permutations, hidden complex pairs,
  companion matrices); CI draws a fresh seed weekly.

Determinism guarantees are in **[CONTRACT.md](../CONTRACT.md)**.

## Disclaimer

This software is provided **"as is"**, without warranty of any kind, express or
implied. Use of this software is entirely at your own risk. In no event shall the
author or contributors be held liable for any damages arising from the use or
inability to use this software.

---

Built by **Niels Erik Toren** · [support & donations](../README.md#author--support).
