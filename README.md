# g_math | gMath

**Author**: Niels Erik Toren

Multi-domain fixed-point arithmetic for Rust.

`g_math` is a pure-Rust arithmetic crate built around a canonical expression pipeline:

`gmath(...) -> LazyExpr -> evaluate(...) -> StackValue`

Under that API, the crate routes values and operations across four numeric domains:

* Binary fixed-point
* Decimal fixed-point
* Balanced ternary fixed-point
* Symbolic rational

It also exposes imperative types such as `FixedPoint`, `FixedVector`, and `FixedMatrix`, but the canonical API is the primary public entry point.

## What g_math is trying to do

Most numeric systems are good at some things and bad at others.

* Binary fixed-point is fast and natural for many low-level operations.
* Decimal fixed-point is often preferable for decimal-facing arithmetic.
* Ternary is available as a first-class domain rather than a curiosity.
* Symbolic rational provides an exact fallback for values that should not be collapsed into an approximation too early.

`g_math` is an attempt to let those domains coexist in one library instead of forcing everything through a single representation.

## Current architecture

### 1. Canonical API

The primary API is the canonical expression pipeline:

```rust
use g_math::canonical::{gmath, evaluate};

let expr = gmath("1.5") + gmath("2.5");
let value = evaluate(&expr).unwrap();
println!("{}", value);
```

There is also `gmath_parse(...)` for runtime strings and `LazyExpr::from(...)` for feeding an evaluated value back into a new expression without reparsing.

### 2. FASC

FASC stands for **Fixed Allocation Stack Computation**.

In practical terms, it means:

* expressions are built as `LazyExpr` trees
* evaluation is deferred until `evaluate(...)`
* evaluation runs through a thread-local `StackEvaluator`
* the evaluator uses a fixed-size value stack and domain-aware dispatch

The important consequence is that the **evaluation engine** is stack-oriented and built around fixed workspace structures rather than a growable runtime evaluator.

This is the path the crate is organized around, and the one new users should start with.

### 3. UGOD — Universal Graceful Overflow Delegation

UGOD is the tiered overflow model.

Each major domain is aligned to a shared tier system. Operations are attempted at the current tier, and when a result cannot be represented there, the computation can promote upward. At the top end, symbolic rational is the exact fallback.

The current universal tier model is:

| Tier | Bits | Binary   | Decimal  | Ternary   | Symbolic  |
| ---- | ---- | -------- | -------- | --------- | --------- |
| 1    | 32   | Q16.16   | D16.16   | TQ8.8     | i16/u16   |
| 2    | 64   | Q32.32   | D32.32   | TQ16.16   | i32/u32   |
| 3    | 128  | Q64.64   | D64.64   | TQ32.32   | i64/u64   |
| 4    | 256  | Q128.128 | D128.128 | TQ64.64   | i128/u128 |
| 5    | 512  | Q256.256 | D256.256 | TQ128.128 | I256/U256 |
| 6    | 1024 | Q512.512 | D512.512 | TQ256.256 | I512/U512 |

At the architecture level:

* tiers 1-5 promote upward on overflow
* tier 6 overflows can fall back to rational arithmetic
* optional unbounded precision can extend symbolic arithmetic beyond the bounded native tiers

The goal is not to avoid overflow by pretending it never happens. The goal is to overflow **gracefully** into a larger or exact representation instead of failing silently.

### 4. Shadow system

`g_math` includes a compact shadow system for preserving exactness metadata alongside approximated values.

The public `CompactShadow` type can store:

* no shadow
* small rational shadows in progressively larger compact forms (2 to 32 bytes)
* a full rational shadow (i128/u128 numerator-denominator pair)
* references to known constants: pi, e, sqrt(2), phi, ln2, ln10, Euler's gamma

This lets an inexact domain value carry a compact rational companion when one exists.

Example idea:

* if a value is stored in a fixed-point domain as an approximation of `1/3`,
* a compact rational shadow can still preserve that exact fractional identity for later use.

In the current implementation, shadow arithmetic is propagated where possible. It is best understood as **exactness retention infrastructure**, not magical infinite memory.

### 5. Wider-tier transcendental computation

The crate implements 18 transcendental functions:

`exp`, `ln`, `sqrt`, `pow`, `sin`, `cos`, `tan`, `atan`, `atan2`, `asin`, `acos`, `sinh`, `cosh`, `tanh`, `asinh`, `acosh`, `atanh`

The current implementation computes each at a wider tier than the active storage tier and then rounds back down. Because the intermediate has more fractional bits than the storage format can represent, the final rounding step produces the nearest representable value at the storage tier.

The profile mapping is:

| Profile    | Storage   | Compute tier |
| ---------- | --------- | ------------ |
| Embedded   | Q64.64    | Q128.128     |
| Balanced   | Q128.128  | Q256.256     |
| Scientific | Q256.256  | Q512.512     |

That wider-tier strategy is one of the central design decisions in the crate.

## Profiles

Build profile selection is driven by `GMATH_PROFILE`. The default is **embedded** (Q64.64, 19 decimal digits).

| Profile      | Format    | Storage | Compute | Decimal digits |
| ------------ | --------- | ------- | ------- | -------------- |
| `embedded`   | Q64.64    | i128    | I256    | 19             |
| `balanced`   | Q128.128  | I256    | I512    | 38             |
| `scientific` | Q256.256  | I512    | I1024   | 77             |

```bash
cargo build                             # embedded (default)
GMATH_PROFILE=balanced cargo build      # 38-digit precision
GMATH_PROFILE=scientific cargo build    # 77-digit precision
```

**Important**: clear the incremental cache when switching profiles. Each profile compiles entirely different code paths via `cfg` flags. Stale artifacts cause build failures or runtime crashes.

```bash
rm -rf target/debug/incremental/        # Run BEFORE switching profiles
GMATH_PROFILE=scientific cargo build    # Now safe to build a different profile
```

Pre-built lookup tables are checked into the repository. A default build completes in about 2 seconds. To regenerate tables from scratch (about 20 minutes):

```bash
cargo build --features rebuild-tables
```

## Feature flags

| Flag | Default | Effect |
| ---- | ------- | ------ |
| `infinite-precision` | off | Adds BigInt tier 8 to the symbolic rational domain. Pulls in `num-bigint`, `num-traits`, `num-integer` as runtime dependencies. Without this flag, the rational domain caps at tier 7 (I512 numerator/denominator). |
| `rebuild-tables` | off | Regenerates all lookup tables (exp, ln, trig) from `build.rs` instead of using the checked-in pre-built tables. Takes about 20 minutes. |
| `legacy-tests` | off | Enables compilation of legacy test suites from earlier development phases. |
| `embedded` | off | Selects embedded profile via Cargo feature instead of environment variable. |
| `balanced` | off | Selects balanced profile via Cargo feature instead of environment variable. |
| `scientific` | off | Selects scientific profile via Cargo feature instead of environment variable. |

All other arithmetic — including transcendental functions, SIMD acceleration (AVX2 runtime-detected on x86_64), tiered overflow, and I256/I512/I1024 wide integer types — is always compiled. There are no feature gates around core functionality.

## Quick start

Add the crate:

```toml
[dependencies]
g_math = "0.1.1"
```

Basic use:

```rust
use g_math::canonical::{gmath, evaluate};

fn main() {
    let expr = (gmath("100") + gmath("50")) / gmath("3");
    let value = evaluate(&expr).unwrap();
    println!("{}", value);
}
```

Runtime parsing:

```rust
use g_math::canonical::{gmath_parse, evaluate};

fn main() {
    let input = "3.14159265358979323846";
    let parsed = gmath_parse(input).unwrap();
    let result = evaluate(&(parsed * gmath("2"))).unwrap();
    println!("{}", result);
}
```

Feeding values back into the expression system:

```rust
use g_math::canonical::{gmath, evaluate, LazyExpr};

fn main() {
    let year0 = evaluate(&gmath("1000")).unwrap();
    let year1 = evaluate(&(LazyExpr::from(year0) * gmath("1.05"))).unwrap();
    println!("{}", year1);
}
```

## Domain routing and mode control

The crate exposes a compute and output mode system.

You can set modes such as:

* `auto:auto` (default — routes each value to its natural domain)
* `binary:ternary` (compute in binary, output in ternary)
* `decimal:symbolic` (compute in decimal, output as symbolic rational)

Available domains: `auto`, `binary`, `decimal`, `symbolic`, `ternary` — any combination as `compute:output`.

Example:

```rust
use g_math::canonical::{set_gmath_mode, reset_gmath_mode, gmath, evaluate};

fn main() {
    set_gmath_mode("binary:ternary").unwrap();
    let value = evaluate(&(gmath("3") + gmath("7"))).unwrap();
    println!("{}", value);
    reset_gmath_mode();
}
```

## Canonical API surface

The primary public interface lives in `g_math::canonical`:

| Item | Purpose |
| ---- | ------- |
| `gmath("...")` | Build a `LazyExpr` from a string literal (deferred parsing) |
| `gmath_parse(&str)` | Build a `LazyExpr` from a runtime string (eager parsing, returns `Result`) |
| `evaluate(&LazyExpr)` | Evaluate an expression tree, returns `Result<StackValue, _>` |
| `LazyExpr` | Expression tree node — supports operator overloading and transcendental methods |
| `LazyExpr::from(StackValue)` | Feed a previous result back into a new expression |
| `StackValue` | Domain-tagged result — implements `Display`, carries shadow metadata |
| `set_gmath_mode("compute:output")` | Set compute and output domain routing |
| `reset_gmath_mode()` | Reset to `auto:auto` |

The imperative API (`FixedPoint`, `FixedVector`, `FixedMatrix`) is also available via `g_math::fixed_point` for mutable arithmetic workflows. Transcendentals on `FixedPoint` route through the FASC evaluator internally.

If you are new to the crate, start with `g_math::canonical`.

## Validation and tests

The published crate includes test suites for:

* arithmetic sweep validation (4 domains, 4 operations, 60k+ reference points)
* boundary stress testing
* compound operations (chained arithmetic, iterative accumulation)
* domain arithmetic validation
* error handling
* FASC ULP validation (18 transcendentals, validated against mpmath at 250+ digit precision)
* mode routing validation (12 modes x 24 test cases)
* transcendental ULP validation

Run the comprehensive suite:

```bash
cargo test --release --test comprehensive_benchmark -- --nocapture --test-threads=1
```

This README intentionally avoids broad numerical slogans. Stronger correctness claims belong in a dedicated validation document with exact definitions, scope, corpus size, and methodology.

## Geometric extension (L1–L5)

The crate includes a geometric mathematics extension built on top of the FASC canonical API. Every operation in this extension follows the **compute-tier principle**: all accumulations, dot products, and matrix chains operate at tier N+1 (double width), with a single downscale at the output boundary. This is the matrix-level analog of BinaryCompute chain persistence for scalars.

**797 tests, 0 failures, all 3 profiles.**

### L1A: Linear algebra

Imperative matrix and vector types. All dot products and matrix multiplications use `compute_tier_dot_raw` at tier N+1.

```rust
use g_math::fixed_point::{FixedPoint, FixedVector, FixedMatrix};

let fp = |s| FixedPoint::from_str(s);

// Vectors — dot product at compute tier (1 ULP)
let u = FixedVector::from_slice(&[fp("1"), fp("2"), fp("3")]);
let v = FixedVector::from_slice(&[fp("4"), fp("5"), fp("6")]);
let d = u.dot(&v);                    // compute-tier accumulation
let len = u.length();                  // via compute-tier dot → sqrt
let n = u.normalized();                // via compute-tier length
let dist = u.metric_distance_safe(&v); // compute-tier sum-of-squares → sqrt
let cross = u.cross(&v);              // 3D cross product
let outer = u.outer_product(&v);       // u ⊗ v → matrix

// Matrices — multiply at compute tier (1 ULP per output element)
let a = FixedMatrix::from_slice(2, 2, &[fp("4"), fp("2"), fp("2"), fp("3")]);
let b = FixedVector::from_slice(&[fp("1"), fp("2")]);
let c = &a * &a;                       // mat-mat multiply (compute-tier dots)
let x = a.mul_vector(&b);             // mat-vec multiply (compute-tier dots)
let tr = a.trace();                    // diagonal sum
let at = a.transpose();               // transpose
let id = FixedMatrix::identity(3);     // identity
let sub = a.submatrix(0, 0, 2, 2);    // extract submatrix
let kron = a.kronecker(&a);           // Kronecker product
```

### L1B: Matrix decompositions

Six decompositions, all at compute-tier precision internally. Every entry computed via `compute_tier_sub_dot_raw` — 0-1 ULP per element.

```rust
use g_math::fixed_point::imperative::decompose::*;

// LU decomposition (Doolittle, partial pivoting)
let lu = lu_decompose(&a).unwrap();
let x = lu.solve(&b).unwrap();         // Ax = b, 0-1 ULP
let det = lu.determinant();             // exact at compute tier
let a_inv = lu.inverse().unwrap();      // full inverse
lu.refine(&a, &b, &x);                 // iterative refinement

// QR decomposition (Householder reflections)
let qr = qr_decompose(&a).unwrap();

// Cholesky decomposition (for SPD matrices)
let chol = cholesky_decompose(&a).unwrap();

// Eigenvalues (Jacobi rotation, symmetric matrices)
let (eigenvalues, eigenvectors) = eigen_symmetric(&a).unwrap();

// SVD (Golub-Kahan-Reinsch)
let svd = svd_decompose(&a).unwrap();

// Schur decomposition (Francis QR)
let schur = schur_decompose(&a).unwrap();
```

### L1C: Derived operations

Norms, least-squares, condition numbers. Norms use compute-tier accumulation.

```rust
use g_math::fixed_point::imperative::derived::*;

let f_norm = frobenius_norm(&a);       // compute-tier sum-of-squares → sqrt
let n1 = norm_1(&a);                   // compute-tier column sums
let ni = norm_inf(&a);                 // compute-tier row sums
let x = solve(&a, &b).unwrap();       // via LU
let d = determinant(&a).unwrap();      // via LU
let a_inv = inverse(&a).unwrap();      // via LU
let cond = condition_number_1(&a).unwrap();
let x_ls = least_squares(&a, &b).unwrap();
let a_inv_spd = inverse_spd(&a).unwrap(); // via Cholesky
```

### L1D: Matrix functions

Matrix exp, log, sqrt, pow. All operations chain through `ComputeMatrix` at tier N+1 — zero mid-chain materializations.

```rust
use g_math::fixed_point::imperative::matrix_functions::*;

let exp_a = matrix_exp(&a).unwrap();      // Padé [6/6] + scaling-squaring
let sqrt_a = matrix_sqrt(&a).unwrap();    // Denman-Beavers iteration
let log_a = matrix_log(&a).unwrap();      // inverse scaling-squaring + Horner

// matrix_pow chains log → scalar_mul → exp entirely at compute tier
let a_half = matrix_pow(&a, fp("0.5")).unwrap();

// exp(log(A)) roundtrip: 2 ULP (was 301 trillion before ComputeMatrix)
```

The `matrix_log` sqrt loop now stays at compute tier (previously: N downscale-upscale cycles per sqrt iteration). The `matrix_pow` log→exp chain is a single compute-tier pipeline with one downscale at the end.

### L2A: ODE solvers

Three integrators. Weighted sums (k1..k6 combinations) are accumulated at compute tier via `compute_tier_dot_raw`. Step-size halving is exact bit-shift.

```rust
use g_math::fixed_point::imperative::ode::*;

// RK4 — classical 4th-order (fixed step)
let traj = rk4_integrate(&system, t0, &x0, t_end, h);

// Dormand-Prince 4(5) — adaptive step, discrete controller
let result = rk45_integrate(&system, t0, &x0, t_end, h0, tol).unwrap();

// Symplectic Störmer-Verlet — energy-preserving (Hamiltonian systems)
let traj = verlet_integrate(&ham_system, t0, &q0, &p0, t_end, h);

// Optional conserved-quantity monitoring with projection
let mut monitor = InvariantMonitor::new(invariant_fn, threshold);
```

### L2B: Tensors

Arbitrary-rank tensors with compute-tier contraction, trace, and symmetrization.

```rust
use g_math::fixed_point::imperative::tensor::Tensor;

let t = Tensor::from_matrix(&a);                    // rank-2 from matrix
let v = Tensor::from_vector(&u);                    // rank-1 from vector
let c = Tensor::contract(&t, &v, &[(1, 0)]);       // index contraction (compute-tier dots)
let tr = t.trace(0, 1);                             // trace (compute-tier accumulation)
let s = t.symmetrize(&[0, 1]);                      // symmetrize (compute-tier sums)
let a = t.antisymmetrize(&[0, 1]);                  // antisymmetrize (compute-tier sums)
let outer = Tensor::outer_product(&t, &v);           // outer product
let raised = t.raise_index(0, &metric_inv);          // index raising via metric
```

### L3A–L3C: Riemannian manifolds

Seven manifold implementations. All metric computations (inner products, distances, geodesics) route through compute-tier dot products or `ComputeMatrix` chains. SPD and Grassmannian operations use `ComputeMatrix` for all matrix multiplication chains with `trace_compute()` for metric traces.

```rust
use g_math::fixed_point::imperative::manifold::*;

// Euclidean R^n — flat space
let euclidean = EuclideanSpace { dim: 3 };
let d = euclidean.distance(&p, &q).unwrap();

// Sphere S^n — closed-form sin/cos/acos geodesics (0-2 ULP)
let sphere = Sphere { dim: 2 };
let d = sphere.distance(&p, &q).unwrap();
let transported = sphere.parallel_transport(&p, &q, &tangent).unwrap();

// Hyperbolic H^n — Minkowski inner product, sinh/cosh/acosh geodesics (0-1 ULP)
let hyp = HyperbolicSpace { dim: 3 };
let d = hyp.distance(&p, &q).unwrap();

// SPD manifold — symmetric positive definite matrices
// Inner product, exp/log maps, distance, transport all via ComputeMatrix chains
let spd = SPDManifold { n: 2 };
let d = spd.distance(&p_spd, &q_spd).unwrap();

// Grassmannian Gr(k, n) — k-dimensional subspaces of R^n
// exp/log/distance via SVD + ComputeMatrix chains
let gr = Grassmannian { k: 2, n: 4 };

// Stiefel St(k, n) — orthonormal k-frames in R^n
let st = Stiefel { k: 2, n: 4 };

// Product manifold — combine any manifolds
let product = ProductManifold::new(vec![
    (Box::new(Sphere { dim: 2 }), 3),
    (Box::new(EuclideanSpace { dim: 2 }), 2),
]);
```

All manifold types implement the `Manifold` trait:

```rust
pub trait Manifold {
    fn dimension(&self) -> usize;
    fn inner_product(&self, base: &FixedVector, u: &FixedVector, v: &FixedVector) -> FixedPoint;
    fn exp_map(&self, base: &FixedVector, tangent: &FixedVector) -> Result<FixedVector, _>;
    fn log_map(&self, base: &FixedVector, target: &FixedVector) -> Result<FixedVector, _>;
    fn distance(&self, p: &FixedVector, q: &FixedVector) -> Result<FixedPoint, _>;
    fn parallel_transport(&self, base: &FixedVector, target: &FixedVector, tangent: &FixedVector) -> Result<FixedVector, _>;
}
```

### L3B: Differential geometry

Christoffel symbols, curvature tensors, geodesic integration. All tensor contractions at compute tier.

```rust
use g_math::fixed_point::imperative::curvature::*;

// Christoffel symbols Γ^k_{ij} — compute-tier contractions
let gamma = christoffel(&metric_fn, &point, dim);

// Riemann curvature tensor R^l_{ijk}
let riemann = riemann_curvature(&metric_fn, &point, dim);

// Ricci tensor R_{ij} and scalar curvature R
let ricci = ricci_tensor(&metric_fn, &point, dim);
let scalar = scalar_curvature(&metric_fn, &point, dim);

// Sectional curvature K(u, v)
let k = sectional_curvature(&metric_fn, &point, &u, &v, dim);

// Geodesic integration via RK4 on the geodesic ODE
let geodesic = geodesic_integrate(&metric_fn, &point, &velocity, dim, t_end, h);

// Parallel transport of a vector along a geodesic
let transported = parallel_transport_ode(&metric_fn, &point, &velocity, &vector, dim, t_end, h);
```

### L4A: Lie groups

Six Lie group implementations. SO(3) and SE(3) use **fused sincos** at compute tier — a single shared range reduction computes both sin(θ) and cos(θ), with scalar coefficients (sinc, half_cosc) computed entirely at tier N+1. Zero mid-chain materializations between trig computation and the matrix formula.

```rust
use g_math::fixed_point::imperative::lie_group::*;

// SO(3) — 3D rotations via closed-form Rodrigues
// Fused sincos + compute-tier coefficients → 0-1 ULP roundtrip
let omega = FixedVector::from_slice(&[fp("0.5"), fp("0.3"), fp("0.7")]);
let r = SO3::rodrigues_exp(&omega).unwrap();
let omega_back = SO3::rodrigues_log(&r).unwrap();

// SE(3) — 3D rigid motions (rotation + translation)
// Fused sincos + compute-tier V·v via mul_vector_compute → 0-1 ULP roundtrip
let xi = FixedVector::from_slice(&[fp("0.1"), fp("0.2"), fp("0.3"), fp("1"), fp("2"), fp("3")]);
let g = SE3::se3_exp(&xi).unwrap();
let xi_back = SE3::se3_log(&g).unwrap();

// SO(n) — general rotations via matrix_exp fallback
let son = SOn { n: 4 };
let r4 = son.lie_exp(&xi_4d).unwrap();

// GL(n) — invertible matrices
let gln = GLn { n: 3 };

// O(n) — orthogonal matrices (det = ±1)
let on = On { n: 3 };

// SL(n) — unit-determinant matrices
let sln = SLn { n: 3 };
```

All Lie groups implement both the `Manifold` and `LieGroup` traits:

```rust
pub trait LieGroup: Manifold {
    fn algebra_dim(&self) -> usize;
    fn matrix_dim(&self) -> usize;
    fn identity_element(&self) -> FixedMatrix;
    fn compose(&self, g1: &FixedMatrix, g2: &FixedMatrix) -> FixedMatrix;
    fn group_inverse(&self, g: &FixedMatrix) -> Result<FixedMatrix, _>;
    fn lie_exp(&self, xi: &FixedVector) -> Result<FixedMatrix, _>;
    fn lie_log(&self, g: &FixedMatrix) -> Result<FixedVector, _>;
    fn hat(&self, xi: &FixedVector) -> FixedMatrix;
    fn vee(&self, xi_hat: &FixedMatrix) -> FixedVector;
    fn adjoint(&self, g: &FixedMatrix, xi: &FixedVector) -> Result<FixedVector, _>;
    fn bracket(&self, xi: &FixedVector, eta: &FixedVector) -> FixedVector;
    fn act(&self, g: &FixedMatrix, point: &FixedVector) -> FixedVector;
}
```

### L4B: Projective geometry

Homogeneous coordinates, cross-ratios, stereographic projection, Möbius transformations.

```rust
use g_math::fixed_point::imperative::projective::*;

// Homogeneous coordinates
let h = to_homogeneous(&p);
let p_back = from_homogeneous(&h).unwrap();

// Cross-ratio (projective invariant)
let cr = cross_ratio(&a, &b, &c, &d, &dir);

// Stereographic projection S^n → R^n and back
let projected = stereo_project(&point_on_sphere);
let lifted = stereo_unproject(&point_in_plane, dim);

// Möbius transformations (complex plane)
let m = Moebius::new(a, b, c, d);
let w = m.apply(z);
let m2 = m.compose(&other);
```

### L5A: Fiber bundles

Trivial, vector, and principal bundles with connection coefficients, horizontal lift, parallel transport, and curvature. All accumulations at compute tier.

```rust
use g_math::fixed_point::imperative::fiber_bundle::*;

// Trivial bundle — direct product of base and fiber
let trivial = TrivialBundle::new(base_dim, fiber_dim);
let (base, fiber) = trivial.project(&total);
let lifted = trivial.lift(&base, &fiber);

// Vector bundle with connection coefficients A^a_{bi}
let bundle = VectorBundle::new(base_dim, fiber_dim);
bundle.set_coeff(a, b, i, value);
let h_lift = bundle.horizontal_lift(&base_tangent, &fiber);       // compute-tier accumulation
let transported = bundle.parallel_transport_along(&path, &fiber); // compute-tier per step
let curv = vector_bundle_curvature(&bundle);                      // R^a_{bij} tensor

// Principal bundle with structure group transitions
let principal = PrincipalBundle::new(base_dim, group_dim, num_charts);
principal.set_transition(i, j, &matrix);
assert!(principal.verify_cocycle(i, j, k));
```

### S1: Serialization

Profile-tagged big-endian binary encoding for wire transport and consensus. Compact, deterministic, cross-platform identical.

```rust
use g_math::fixed_point::imperative::serialization::*;

// FixedPoint: [u8 profile tag][raw bytes]
let bytes = fp_val.to_bytes();
let restored = FixedPoint::from_bytes(&bytes).unwrap();

// FixedVector: [u32 len][elements...]
let bytes = vec.to_bytes();
let restored = FixedVector::from_bytes(&bytes).unwrap();

// ManifoldPoint: [u8 manifold tag][point data]
let mp = ManifoldPoint::new(MANIFOLD_TAG_SPHERE, &point);
let bytes = mp.to_bytes();
```

### Fused sincos

`FixedPoint::try_sincos()` computes sin(x) and cos(x) from a single shared range reduction at compute tier. More efficient than separate `try_sin` + `try_cos`, and used internally by Rodrigues (SO3), SE3 exp/log, and `evaluate_tan` in the FASC pipeline.

```rust
let theta = FixedPoint::from_str("1.2345");
let (sin_t, cos_t) = theta.try_sincos().unwrap(); // single range reduction, both at 0 ULP
```

### Precision guarantees

All precision claims are empirically measured against mpmath at 50+ digit precision, not theoretical. Validated with 797 tests across all modules.

| Operation | ULP | Measurement |
|-----------|-----|-------------|
| Transcendentals (all 18) | 0 | 18/18 × 3 profiles, mpmath 250-digit refs |
| Vector dot product | 1 | compute-tier accumulation |
| Matrix multiply | 1 per entry | compute-tier dot per output element |
| LU/Cholesky solve | 0-1 | Well-conditioned systems |
| Manifold geodesics | 0-2 | Sphere, hyperbolic, SPD, Grassmannian |
| Lie group exp/log roundtrip | 0-1 | SO(3) fused sincos Rodrigues, SE(3) compute-tier V·v |
| Matrix exp (Padé [6/6]) | 1-7 | ComputeMatrix throughout |
| Matrix pow (log→exp chain) | 1-7 | Single compute-tier pipeline, zero mid-chain downscale |
| exp(log(A)) roundtrip | 2 | Was 301 trillion before ComputeMatrix |
| Hilbert 4×4 residual | 0 | After iterative refinement |
| ODE RK4 step | 1 per step | compute-tier weighted sums |
| Tensor contraction | 1 per entry | compute-tier dot products |
| Frobenius / 1-norm / inf-norm | 1 | compute-tier accumulation |

**Practical limitations:** Values like 0.3 and 0.7 are repeating binary fractions with 1 ULP representation error. Operations with high condition numbers (Hilbert matrices, rotation formulas) amplify this input error. This is a fundamental limit of finite-precision arithmetic — binary, decimal, or otherwise — not an implementation deficiency. The roundtrip precision (which cancels input errors) proves the implementation is mathematically correct.

**Determinism guarantee:** All results are bit-identical across x86_64, ARM, RISC-V, and any other architecture. Every operation is pure integer arithmetic on Q-format storage. No floating-point anywhere in the pipeline.

**Profile support:** All geometric operations work across all three profiles (embedded Q64.64, balanced Q128.128, scientific Q256.256) via compile-time `#[cfg]` gates. The same source code, same algorithms, same precision guarantees.

## Design notes

This crate is opinionated.

It does not pretend all arithmetic should collapse into one representation. It does not assume floating point is the only practical route. It tries to preserve exactness when possible, promote gracefully when necessary, and keep the main API compact.

That is the wager.

## Author note

I write software like a builder from first principles, not a committee. This is a library I built because I needed a precise and deterministic fixed-point library.

Instead of focusing on front-end apps, I prefer to rebuild from first principles keystone libraries so these are future-proof and allow me to build software and paradigms that didn't exist before.

So yes, some of this project carries personal style, philosophy, and a slightly stubborn tone. That is intentional.

If this crate is useful to you, then use it, stress it, break it, and tell me where it fails. It could contain flaws but I have not found them myself. I validated all operations against mpmath — run the comprehensive test to see for yourself.

If you want to support the work:

| Currency | Address |
|----------|---------|
| Bitcoin (BTC) | bc1qwf78fjgapt2gcts4mwf3gnfkclvqgtlg4gpu4d |
| Ethereum (ETH) | 0xf38b517Dd2005d93E0BDc1e9807665074c5eC731 / nierto.eth |
| Monero (XMR) | 8BPaSoq1pEJH4LgbGNQ92kFJA3oi2frE4igHvdP9Lz2giwhFo2VnNvGT8XABYasjtoVY2Qb3LVHv6CP3qwcJ8UnyRtjWRZ5 |

Please star the project on GitHub if it was useful to you. Thank you sincerely.

I am building this in the middle of life, work, pressure, family, and limited time. That does not make the project weaker. It is the reason it exists at all. We don't do things because they are easy, but because they are hard.

## Disclaimer

This software is provided **"as is"**, without warranty of any kind, express or implied. Use of this library is entirely at your own risk. In no event shall the author or contributors be held liable for any damages, data loss, financial loss, or other consequences arising from the use or inability to use this software. By using gMath, you accept full responsibility for verifying its suitability for your use case.

See the license texts for the full legal terms.

## License

Licensed under either of

* [Apache License, Version 2.0](LICENSE-APACHE)
* [MIT License](LICENSE-MIT)

at your option.
