# Upgrading from 0.6 to 0.7

0.7.0 makes the fixed-point format a type. A 0.6 build chose one format for
the whole program through `GMATH_PROFILE` (and `GMATH_FRAC_BITS` on
realtime), and the library called it `FixedPoint`. A 0.7 build has every
format, and your code says which one each value uses.

Most programs change in three places: one `type` line, the containers that
name their element, and the `evaluate` calls of the canonical layer. The
examples below compile against 0.7 (they are checked as doc tests).

## 1. Pick the type for your old profile

| 0.6 build | 0.7 type | storage | fraction bits |
|---|---|---|---|
| `GMATH_PROFILE=realtime` (default split) | `Q16_16` = `Fixed<i32, 16>` | i32 | 16 |
| `GMATH_PROFILE=realtime GMATH_FRAC_BITS=10` | `Fixed<i32, 10>` | i32 | 10 |
| realtime with any other split `f` | `Fixed<i32, f>` (2 <= f <= 30) | i32 | f |
| `GMATH_PROFILE=compact` | `Q32_32` = `Fixed<i64, 32>` | i64 | 32 |
| `GMATH_PROFILE=embedded` (and no profile) | `Q64_64` = `Fixed<i128, 64>` | i128 | 64 |
| `GMATH_PROFILE=balanced` | `Q128_128` = `Fixed<I256, 128>` | I256 | 128 |
| `GMATH_PROFILE=scientific` | `Q256_256` = `Fixed<I512, 256>` | I512 | 256 |

Remove `GMATH_PROFILE` / `GMATH_FRAC_BITS` from build scripts, CI and
`.cargo/config.toml`, and the profile features (`realtime`, `compact`,
`embedded`, `balanced`, `scientific`) from `Cargo.toml`: 0.7 does not read
them. The advice to clear `target/debug/incremental/` between profiles no
longer applies.

## 2. The imperative types

A one-line alias keeps most 0.6 code as it was:

```rust
use g_math::fixed_point::{Fixed, FixedVector, FixedMatrix};

// was: use g_math::fixed_point::FixedPoint;  (built with GMATH_PROFILE=realtime GMATH_FRAC_BITS=10)
type FixedPoint = Fixed<i32, 10>;

let x = FixedPoint::from_str("1.5");
let y = (x * x).sqrt() + FixedPoint::from_int(2);
assert_eq!(y, FixedPoint::from_str("3.5"));

// containers name their element: there is no default type any more
let v: FixedVector<FixedPoint> = FixedVector::from_slice(&[x, y]);
let m: FixedMatrix<FixedPoint> = FixedMatrix::identity(2);
assert_eq!(m.mul_vector(&v), v);
```

Different formats coexist in one program, and in one dependency graph two
crates can use different formats:

```rust
use g_math::fixed_point::{Fixed, Q64_64};

let fast = Fixed::<i32, 10>::from_str("0.125");      // Q22.10
let wide = Q64_64::from_str("0.1");                  // Q64.64
assert_eq!(fast.raw(), 128);
assert_eq!(wide.to_string().len() > 10, true);
```

Generic code takes the format as a parameter; the traits that describe a
format are `FixedNum` (every format), `Storage` and `WideInt` (its words)
and `Scalar` (what the geometry and linear algebra layers need):

```rust
use g_math::fixed_point::{Fixed, FixedNum, Storage, Q16_16, Q128_128};

fn half<S: Storage, const F: u32>(x: Fixed<S, F>) -> Fixed<S, F>
where
    Fixed<S, F>: FixedNum<Storage = S>,
{
    x / Fixed::from_int(2)
}

assert_eq!(half(Q16_16::from_str("3")), Q16_16::from_str("1.5"));
assert_eq!(half(Q128_128::from_str("3")), Q128_128::from_str("1.5"));
```

The geometry, ODE, tensor, interval and fused APIs keep their names and
arguments; their types carry the format (`Interval<Q64_64>`,
`ManifoldPoint<Q64_64>`, `LieGroup<T>` and `Manifold<T>` implemented for
every format).

## 3. The canonical layer

`gmath`, `gmath_parse`, `evaluate`, `evaluate_sincos`, `evaluate_sinhcosh`
and `evaluate_matrix` have the names they had; they are evaluated in a
format, which one annotation usually fixes for the whole tree:

```rust
use g_math::canonical::{evaluate, gmath, gmath_parse};
use g_math::fixed_point::Q64_64;

// was: evaluate(&(gmath("1.5") + gmath("2.5")))
let sum = evaluate::<Q64_64>(&(gmath("1.5") + gmath("2.5"))).unwrap();
assert_eq!(sum.to_string(), "4.0");

// runtime strings: name the format on the parse
let e = gmath_parse::<Q64_64>("0.5").unwrap().exp();
let v = evaluate(&e).unwrap();          // format inferred from `e`
assert!(v.to_string().starts_with("1.6487212707001281468"));
```

Each thread keeps one evaluator per format it uses; `set_gmath_mode` and
`reset_gmath_mode` are unchanged.

## 4. The compute tier

`g_math::compute_tier` no longer needs the `inference` feature. Its
functions take the format as a type parameter, and the compute value is
`Compute<T>` (the format's compute word, `2F` fraction bits):

```rust
use g_math::compute_tier as ct;
use g_math::fixed_point::Fixed;

type Q = Fixed<i32, 10>;
let x = ct::from_fixed(Q::from_str("0.5"));    // format inferred
let s = ct::to_fixed::<Q>(ct::sigmoid::<Q>(x));
assert_eq!(s, Q::from_str("0.6225"));           // nearest at 10 fraction bits
```

`compute_tier::FRAC_BITS`, `COMPUTE_FRAC_BITS` and `ComputeStorage` are
gone: use `F`, `2 * F` and `Compute<T>`.

## 5. Stored bytes

`to_bytes` writes a new three-byte tag (the word and the split), and
`from_bytes` requires it. Data written by 0.6.x reads with
`from_bytes_legacy`; write it back with `to_bytes` to migrate:

```rust
use g_math::fixed_point::Q64_64;

let x = Q64_64::from_str("2.25");
let new = x.to_bytes();
assert_eq!(Q64_64::from_bytes(&new).unwrap(), x);

// 0.6.x (embedded) wrote [legacy_tag][raw big-endian]
let mut old = vec![Q64_64::legacy_tag()];
old.extend_from_slice(&x.raw().to_be_bytes());
assert_eq!(Q64_64::from_bytes_legacy(&old).unwrap(), x);
```

`to_raw_bytes` / `from_raw_bytes` and the compact form are unchanged.
`profile_tag()` is `legacy_tag()`.

## 6. Ternary inference

The `tq19` module and the `inference` feature moved to the separate,
separately licensed crate `g_math_inference`, with the same types and
functions. The kernels take activation slices of the format instead of raw
integers; `Fixed::from_raw_slice` and `raw_slice` convert without copying:

```rust,ignore
// was: use g_math::tq19::TQ19Matrix;  m.matvec(&raw_i32_activations)
use g_math_inference::TQ19Matrix;
use g_math::fixed_point::Fixed;

type Q = Fixed<i32, 10>;
let acts: &[Q] = Q::from_raw_slice(&raw_i32_activations);
let y: Vec<Q> = m.matvec(acts);
let y_raw: &[i32] = Q::raw_slice(&y);
```

Its outputs are bit-identical to 0.6.7's module.

## 7. Results that change, and new panics

Run your own value checks once after upgrading. What moves:

- **Correctly rounded now**, so a result can differ from 0.6.7 where 0.6.7
  was not: every transcendental (mostly by one unit, by much more near the
  top of a format's range on 0.6.7), canonical chains (one rounding for the
  whole tree), the closed-form geometry maps (sphere, hyperboloid, SO(3),
  SE(3)), `sqrt` (about 1 input in 2,000 at Q22.10), the decimal
  transcendentals at small D, `DecimalFixed::to_f64_lossy`.
- **Unchanged**: arithmetic, conversions, parsing, `to_f64`, linear algebra,
  decompositions, fused kernels, intervals and predicates, the wide tier,
  the inference kernels.
- **Panics where 0.6.7 returned a wrong value**: `asin` / `acos` outside
  `[-1, 1]`, `atanh` outside `(-1, 1)`, `acosh` below 1, the ternary TQ80.80
  and TQ320.320 add and subtract on overflow. Use the `try_` / `_checked`
  forms to get an `Err`.
- **`raise_index` / `lower_index`** keep the mapped index in its position
  (0.6.7 moved it first, transposing the result for `idx > 0`).
- **Panic messages** say `Fixed` instead of `FixedPoint`; a test that
  matches a panic text may need the new wording.

## 8. Removed

`FixedPoint`, `BinaryStorage`, `ComputeStorage`, `frac_config`,
`DeploymentProfile`, `profile_tag`, `DecimalFixed::from_raw_checked` (it
rejected valid raws), the `legacy-tests` feature, the profile features and
environment variables, the `tq19` module and the `inference` feature.

The full list of changes is in [CHANGELOG.md](../CHANGELOG.md).
