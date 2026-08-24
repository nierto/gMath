# Geometry

Riemannian manifolds, Lie groups, differential geometry, projective geometry,
fiber bundles, ODE integrators, and tensors: all over the binary fixed-point
types.

## What it is

The geometric layer builds on [`FixedVector`/`FixedMatrix`](README_IMPERATIVE.md)
and the [linear algebra](README_LINALG.md) modules to provide coordinate-free
constructs: geodesics and parallel transport on manifolds, exp/log and the
adjoint on Lie groups, curvature tensors, projective transforms, connections on
bundles, ODE integration, and arbitrary-rank tensors. Closed-form algorithms are
used where they exist (Rodrigues for SO(3), the V-matrix for SE(3)); everything
else composes the wide-tier matrix functions.

## Usage

```rust
use g_math::fixed_point::{FixedVector, FixedPoint};
use g_math::fixed_point::imperative::manifold::{Manifold, Sphere};

let s = Sphere { dim: 2 };                    // unit 2-sphere (ambient = dim + 1)
let p = FixedVector::from_slice(&[FixedPoint::one(), FixedPoint::ZERO, FixedPoint::ZERO]);
let q = FixedVector::from_slice(&[FixedPoint::ZERO, FixedPoint::one(), FixedPoint::ZERO]);
let d = s.distance(&p, &q).unwrap();          // geodesic distance
let v = s.log_map(&p, &q).unwrap();           // tangent vector p → q
let back = s.exp_map(&p, &v).unwrap();        // ≈ q
```

## What's here

- **Manifolds** (`manifold`): trait `exp_map`, `log_map`, `distance`,
  `parallel_transport`, `inner_product`: Euclidean, Sphere, Hyperbolic
  (hyperboloid model), SPD, Grassmannian, Stiefel, products.
- **Lie groups** (`lie_group`): trait adds `lie_exp`, `lie_log`, `hat`/`vee`,
  `adjoint`, `bracket`, `act`: SO(3) via closed-form Rodrigues, SE(3) via
  closed-form V-matrix, plus SO(n), GL(n), O(n), SL(n) via matrix exp/log.
- **Differential geometry** (`curvature`): Christoffel symbols,
  Riemann/Ricci/scalar/sectional curvature, geodesic integration, parallel
  transport along curves.
- **Projective** (`projective`): homogeneous coordinates, projective transforms,
  cross-ratios, stereographic projection, Möbius transformations (real and
  complex).
- **Fiber bundles** (`fiber_bundle`): trivial, vector (connection coefficients,
  horizontal lift, parallel transport, curvature 2-form), principal (transition
  cocycles).
- **ODE solvers** (`ode`): RK4 (fixed step), Dormand-Prince RK45 (adaptive),
  symplectic Störmer-Verlet (energy-preserving). Weighted sums accumulate at the
  compute tier; step halving is an exact bit shift.
- **Tensors** (`tensor`, `tensor_decompose`): arbitrary-rank contraction, outer
  product, trace, index raising/lowering via a metric, (anti)symmetrization;
  decompositions `truncated_svd`, `tucker_decompose` (HOSVD), `cp_decompose` (ALS).

## Public API

See **[PUBLIC_API.md → Geometry](../PUBLIC_API.md#geometry)**,
**[→ ODE](../PUBLIC_API.md#ode)**, **[→ Tensors](../PUBLIC_API.md#tensors)**, and
[docs.rs](https://docs.rs/g_math).

## Behaviour & limits

- All operations are in the binary domain via `FixedMatrix`/`FixedVector`.
- Geodesic and Lie-group roundtrips are validated against mpmath references;
  accuracy of formula-sensitive constructs depends on input representability (see
  [the precision guide](README_PRECISION.md)).

Determinism guarantees are in **[CONTRACT.md](../CONTRACT.md)**.

## Disclaimer

This software is provided **"as is"**, without warranty of any kind, express or
implied. Use of this software is entirely at your own risk. In no event shall the
author or contributors be held liable for any damages arising from the use or
inability to use this software.

---

Built by **Niels Erik Toren** · [support & donations](../README.md#author--support).
