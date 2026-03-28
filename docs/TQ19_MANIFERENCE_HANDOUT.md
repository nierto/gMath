# TQ1.9 Migration Handout — Maniference Dev Team

**Version**: g_math 0.3.89 → TQ1.9 standalone module
**Date**: 2026-03-28
**Impact**: Drop-in replacement for `model.rs::TQ19Matrix` + `ternary.rs::ternary_matvec`

---

```yaml
@CONTEXT: {
  what: "gMath now ships a dedicated TQ1.9 module — decoupled from FASC/routing/shadows",
  why: "centralize ternary math in gMath, gain SIMD+rayon for free, remove inline arithmetic from Maniference",
  scope: "TQ1.9 matvec (196 calls/token) + native ternary matvec (TriLM path)",
  precision: "identical — compute-tier accumulation, single /SCALE per row"
}

@WHAT_CHANGED_IN_GMATH: {
  new-module: "g_math::tq19 (also g_math::fixed_point::tq19)",
  new-types: {
    TQ19Matrix: "row-major i16 weights, rows/cols validated at construction",
  },
  new-functions: {
    tq19_dot: "sum(w[i]*a[i])/19683 — compute-tier accumulation, single division",
    trit_dot: "zero-multiply dot for pre-decoded trits (i8 in {-1,0,1})",
    packed_trit_dot: "zero-multiply dot from packed 5-trits/byte with per-block scale",
    packed_trit_matvec: "row-wise packed trit matvec with per-row scales",
  },
  new-constants: {
    SCALE: "19683 (i32)",
    MAX_RAW: "29524 (i16)",
    MIN_RAW: "-29524 (i16)",
    TRIT_DECODE_TABLE: "const [[i8;5];256] — replaces Maniference TRIT_TABLE",
  },
  features: {
    parallel: "enables rayon — TQ19Matrix::matvec_par, matvec_batch_par, packed_trit_matvec_par",
    default: "zero deps — scalar only, no rayon"
  },
  simd: {
    target: "x86_64 + Q16.16 (realtime profile) — auto-detected at runtime",
    tq19_dot: "8× i32 multiply-accumulate per AVX2 cycle (_mm256_mul_epi32)",
    trit_dot: "8× zero-multiply per AVX2 cycle (_mm256_sign_epi32)",
    fallback: "scalar on all other profiles/architectures — transparent"
  }
}

@MIGRATION_STEPS: {

  step-1-cargo-toml: {
    action: "update g_math dependency",
    before: 'g_math = { path = "../gMath" }',
    after:  'g_math = { path = "../gMath", features = ["parallel"] }',
    note: "parallel enables rayon row-dispatch — replaces your inline into_par_iter()"
  },

  step-2-replace-tq19-matrix: {
    action: "replace model.rs::TQ19Matrix with g_math::tq19::TQ19Matrix",
    remove: [
      "pub struct TQ19Matrix { pub rows, pub cols, pub data: Vec<i16> }",
      "impl TQ19Matrix { pub fn mul_vector(...) }",
      "pub fn from_f16(...)",
      "const SCALE_TQ1_9: i16"
    ],
    add: [
      "use g_math::tq19::{TQ19Matrix, SCALE};",
    ],
    construction: {
      before: "TQ19Matrix { rows, cols, data }",
      after: "TQ19Matrix::new(rows, cols, data)",
      note: "constructor validates rows×cols == data.len()"
    },
    matvec: {
      before: "tq.mul_vector(v)  // returns FixedVector",
      after: "tq.matvec_par(v.as_raw_slice())  // returns Vec<BinaryStorage>",
      note: "wrap result: FixedVector::from_raw_slice(&result) if needed"
    },
    batch: {
      before: "// no batch support — called in a loop",
      after: "tq.matvec_batch_par(&[&v1_raw, &v2_raw, ...])",
      note: "weight-centric iteration — row data stays in L1 across batch vectors"
    }
  },

  step-3-replace-ternary-matvec: {
    action: "replace ternary.rs::ternary_matvec with packed_trit_matvec_par",
    before: "ternary::ternary_matvec(tm, v)",
    after: "g_math::tq19::packed_trit_matvec_par(tm.packed_data(), tm.rows, tm.cols, v_raw, tm.scales())",
    note: [
      "TRIT_TABLE in ternary.rs is replaced by g_math::tq19::TRIT_DECODE_TABLE",
      "per-block scales become per-row scales in the new API",
      "your prepare_for_inference() pre-decode step is now handled internally"
    ]
  },

  step-4-replace-projection-dispatch: {
    action: "update ProjectionWeight::matvec dispatch",
    before: |
      pub fn matvec(&self, v: &FixedVector) -> Result<FixedVector> {
          match self {
              Self::Ternary(tm) => ternary::ternary_matvec(tm, v),
              Self::TQ19(tq) => Ok(tq.mul_vector(v)),
          }
      }
    after: |
      pub fn matvec(&self, v: &FixedVector) -> Result<FixedVector> {
          let raw_v: Vec<BinaryStorage> = v.iter().map(|x| x.raw()).collect();
          let result_raw = match self {
              Self::Ternary(tm) => packed_trit_matvec_par(
                  tm.packed_data(), tm.rows, tm.cols, &raw_v, tm.scales()
              ),
              Self::TQ19(tq) => tq.matvec_par(&raw_v),
          };
          Ok(FixedVector::from_raw_slice(&result_raw))
      }
  },

  step-5-remove-inline-rayon: {
    action: "remove into_par_iter() from model.rs mul_vector and ternary.rs ternary_matvec",
    reason: "parallelism is now inside gMath — double-rayon nesting wastes thread pool overhead",
    check: "grep -n into_par_iter src/model.rs src/ternary.rs → should find 0 matches after migration"
  },

  step-6-remove-trit-table: {
    action: "remove TRIT_TABLE and decode_byte_trits from ternary.rs",
    reason: "replaced by g_math::tq19::TRIT_DECODE_TABLE (const, compiled into .rodata, same encoding)"
  }
}

@PERFORMANCE_EXPECTATIONS: {
  precision: "identical to current — compute-tier accumulation with single downscale",

  throughput-gains: {
    rayon: "already used — no change here, just moved into gMath",
    batch-matvec: {
      mechanism: "weight-centric iteration: load each row once, apply to all batch vectors",
      expected: "~2-4× for prefill phase (batch_size=4-16) due to weight cache reuse",
      how: "replace per-token matvec loop with matvec_batch_par for prefill tokens"
    },
    simd-realtime: {
      when: "GMATH_PROFILE=realtime (Q16.16, 4 decimal digits)",
      tq19: "8× multiply-accumulate per cycle via AVX2 → ~4× wall-clock speedup per dot",
      trit: "8× zero-multiply per cycle via _mm256_sign_epi32 → ~6× speedup per dot",
      note: "requires compiling with realtime profile — lower precision tradeoff"
    },
    simd-compact: {
      when: "GMATH_PROFILE=compact (Q32.32, current Maniference default)",
      status: "scalar — native i64 multiply is already 1-cycle throughput on x86-64",
      future: "AVX-512 could help (4× i64 multiply) but not supported yet"
    }
  },

  cache-profile: {
    row-weights: "cols × 2 bytes (4096 dim → 8 KB, fits in L1d)",
    activation-vector: "cols × sizeof(BinaryStorage) (compact → 32 KB, fits in L1d)",
    total-per-row: "~40 KB — tight for 32KB L1d, fine for 48-64KB L1d",
    batch-benefit: "row loaded once, reused across batch — eliminates redundant L2→L1 transfers"
  }
}

@API_REFERENCE: {
  imports: |
    use g_math::tq19::{
        TQ19Matrix, SCALE, MAX_RAW, MIN_RAW, TRIT_DECODE_TABLE,
        tq19_dot, trit_dot, packed_trit_dot, packed_trit_matvec,
    };
    // With parallel feature:
    use g_math::tq19::packed_trit_matvec_par;
    // Storage type:
    use g_math::fixed_point::imperative::BinaryStorage;

  TQ19Matrix: {
    new: "TQ19Matrix::new(rows, cols, data: Vec<i16>) → panics if len != rows×cols",
    from_fn: "TQ19Matrix::from_fn(rows, cols, |row, col| -> i16)",
    matvec: "matvec(&[BinaryStorage]) → Vec<BinaryStorage>",
    matvec_par: "matvec_par(&[BinaryStorage]) → Vec<BinaryStorage> [parallel feature]",
    matvec_batch: "matvec_batch(&[&[BinaryStorage]]) → Vec<Vec<BinaryStorage>>",
    matvec_batch_par: "matvec_batch_par(&[&[BinaryStorage]]) → Vec<Vec<BinaryStorage>> [parallel feature]",
    matvec_fp: "matvec_fp(&[BinaryStorage]) → Vec<FixedPoint> (convenience wrapper)",
    rows: "rows() → usize",
    cols: "cols() → usize",
    data: "data() → &[i16]",
    row_slice: "row_slice(row) → &[i16]",
    get: "get(row, col) → i16",
  }
}

@TESTING_CHECKLIST: [
  "cargo test --lib tq19 — runs unit tests in gMath",
  "cargo test --test tq19_validation — runs integration tests",
  "after migration: cargo test in Maniference — verify 0 regressions",
  "compare token output: pre-migration vs post-migration should be BIT-IDENTICAL",
  "benchmark: run inference on a 100-token prompt, compare t/s before and after"
]

@NOTES: {
  no-breaking-change: "g_math::tq19 is additive — existing FASC/UGOD API is untouched",
  from-f16-conversion: "TQ19Matrix::new() takes raw i16 data — your from_f16() stays in Maniference as the conversion layer (it's model-format-specific, not math)",
  distributed-gap: "TQ1.9 still cannot be sharded (no SVD path) — this module doesn't change that",
  future: "signal analysis, quantization-aware training, and other domains can use g_math::tq19 directly — it's not inference-specific"
}
```

---

**Questions?** File issues on github.com/nierto/gMath or reach out directly.
