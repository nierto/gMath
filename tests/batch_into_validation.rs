//! Caller-provided output buffers (0.6.7): every `_into` form writes the
//! same integers its allocating twin returns.
//!
//! The batch matvecs write flat and batch-major, `out[b * rows + r]`; the
//! fused forms write the mix, the weights and the dots into slices. Batch
//! sizes cover the tile boundaries of the kernels (1, 3, 4, 5, 9, 33) and
//! the row counts the groups of four.

#![cfg(feature = "inference")]

use g_math::fixed_point::imperative::fused;
use g_math::fixed_point::FixedPoint;
use g_math::tq19::{HybridTQ19, PlanarTQ19, TQ19Matrix};

type Raw = <FixedPoint as RawOf>::Raw;
trait RawOf {
    type Raw;
}
impl RawOf for FixedPoint {
    type Raw = g_math::fixed_point::universal::fasc::stack_evaluator::BinaryStorage;
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    /// A value k / 16 in [-2, 2): exact at every split from 4 fractional bits.
    fn small(&mut self) -> FixedPoint {
        FixedPoint::from_int((self.next() % 64) as i32 - 32) / FixedPoint::from_int(16)
    }
    fn weights(&mut self, n: usize) -> Vec<i16> {
        let mut w = Vec::with_capacity(n);
        for _ in 0..n {
            let v = ((self.next() % 59049) as i32 - 29524) as i16;
            w.push(if self.next() % 3 == 0 { 0 } else { v });
        }
        w
    }
    fn batch(&mut self, nb: usize, cols: usize) -> Vec<Vec<Raw>> {
        (0..nb).map(|_| (0..cols).map(|_| self.small().raw()).collect()).collect()
    }
}

const SHAPES: &[(usize, usize)] = &[(1, 1), (3, 7), (4, 16), (9, 33), (13, 5)];
const BATCHES: &[usize] = &[0, 1, 3, 4, 5, 9, 33];

fn flat<T: Clone>(v: Vec<Vec<T>>) -> Vec<T> {
    v.into_iter().flatten().collect()
}

#[test]
fn tq19_matrix_forms_write_what_they_return() {
    let mut rng = Rng(0x0BA7_C401);
    for &(rows, cols) in SHAPES {
        let dense = TQ19Matrix::new(rows, cols, rng.weights(rows * cols));
        let planar = PlanarTQ19::from_tq19(&dense);
        let hybrid = HybridTQ19::from_tq19(&dense);
        for &nb in BATCHES {
            let batch = rng.batch(nb, cols);
            let refs: Vec<&[Raw]> = batch.iter().map(|v| &v[..]).collect();
            let want = flat(dense.matvec_batch_par(&refs));
            assert_eq!(want.len(), nb * rows);

            let mut out = vec![FixedPoint::ZERO.raw(); nb * rows];
            dense.matvec_batch_par_into(&refs, &mut out);
            assert_eq!(out, want, "TQ19Matrix {rows}x{cols} batch {nb}");

            let mut out = vec![FixedPoint::ZERO.raw(); nb * rows];
            planar.matvec_batch_par_into(&refs, &mut out);
            assert_eq!(out, flat(planar.matvec_batch_par(&refs)), "PlanarTQ19 {rows}x{cols} batch {nb}");
            assert_eq!(out, want);

            let mut out = vec![FixedPoint::ZERO.raw(); nb * rows];
            hybrid.matvec_batch_par_into(&refs, &mut out);
            assert_eq!(out, flat(hybrid.matvec_batch_par(&refs)), "HybridTQ19 {rows}x{cols} batch {nb}");
            assert_eq!(out, want);
        }
    }
}

#[cfg(any(table_format = "q16_16", table_format = "q32_32"))]
#[test]
fn row_scaled_tq19_writes_what_it_returns() {
    use g_math::tq19::RowScaledTQ19;
    let mut rng = Rng(0x0BA7_C402);
    for &(rows, cols) in SHAPES {
        let scales: Vec<u64> = (0..rows).map(|_| rng.next() % (1u64 << 30)).collect();
        let m = RowScaledTQ19::from_parts(rows, cols, rng.weights(rows * cols), scales);
        for &nb in BATCHES {
            let batch = rng.batch(nb, cols);
            let refs: Vec<&[Raw]> = batch.iter().map(|v| &v[..]).collect();
            let mut out = vec![FixedPoint::ZERO.raw(); nb * rows];
            m.matvec_batch_par_into(&refs, &mut out);
            assert_eq!(out, flat(m.matvec_batch_par(&refs)), "RowScaledTQ19 {rows}x{cols} batch {nb}");
        }
    }
}

#[cfg(table_format = "q16_16")]
#[test]
fn row_scaled_tq5_writes_what_it_returns() {
    use g_math::tq19::RowScaledTQ5;
    let mut rng = Rng(0x0BA7_C403);
    // wide enough for the AVX2 tile, with rows past the last group of four
    for &(rows, cols) in &[(1usize, 1usize), (4, 64), (9, 100), (13, 257), (16, 32)] {
        let data: Vec<i8> = (0..rows * cols).map(|_| ((rng.next() % 243) as i32 - 121) as i8).collect();
        let scales: Vec<u64> = (0..rows).map(|_| rng.next() % (1u64 << 28)).collect();
        let m = RowScaledTQ5::from_parts(rows, cols, data, scales);
        for &nb in BATCHES {
            // Raw activations, the same at every split. Small ones (high
            // halves all zero) take the tile path; every third vector has
            // dense high halves and takes the per-row path. |x| < 2^18
            // keeps every scaled row in range.
            let batch: Vec<Vec<i32>> = (0..nb)
                .map(|i| {
                    (0..cols)
                        .map(|_| if i % 3 == 2 { (rng.next() as u32 as i32) >> 13 } else { (rng.next() % 8192) as i32 - 4096 })
                        .collect()
                })
                .collect();
            let refs: Vec<&[i32]> = batch.iter().map(|v| &v[..]).collect();
            let each: Vec<i32> = refs.iter().flat_map(|x| m.matvec(x)).collect();
            let mut out = vec![0i32; nb * rows];
            m.matvec_batch_par_into(&refs, &mut out);
            assert_eq!(out, each, "RowScaledTQ5 {rows}x{cols} batch {nb}: buffer form against single matvecs");
            assert_eq!(flat(m.matvec_batch_par(&refs)), each, "RowScaledTQ5 {rows}x{cols} batch {nb}: allocating form");
        }
    }
}

#[test]
#[should_panic(expected = "out length is not batch.len() * rows")]
fn a_wrong_buffer_length_panics() {
    let m = TQ19Matrix::new(2, 2, vec![1, 2, 3, 4]);
    let x = [FixedPoint::one().raw(), FixedPoint::one().raw()];
    let mut out = vec![FixedPoint::ZERO.raw(); 3];
    m.matvec_batch_par_into(&[&x[..]], &mut out);
}

#[test]
fn fused_forms_write_what_they_return() {
    let mut rng = Rng(0x0BA7_C404);
    for &(n, dim) in &[(1usize, 1usize), (2, 3), (7, 16), (40, 33), (130, 8)] {
        let scores: Vec<FixedPoint> = (0..n).map(|_| rng.small()).collect();
        let values: Vec<FixedPoint> = (0..n * dim).map(|_| rng.small()).collect();
        let (want_out, want_w) = fused::softmax_mix_flat(&scores, &values, dim).unwrap();

        let mut out = vec![FixedPoint::ZERO; dim];
        let mut w = vec![FixedPoint::ZERO; n];
        fused::softmax_mix_flat_into(&scores, &values, dim, &mut out, &mut w).unwrap();
        assert_eq!((out, w), (want_out.clone(), want_w), "softmax_mix_flat_into n={n} dim={dim}");

        let mut out = vec![FixedPoint::ZERO; dim];
        fused::softmax_mix_flat_values_into(&scores, &values, dim, &mut out).unwrap();
        assert_eq!(out, want_out, "softmax_mix_flat_values_into n={n} dim={dim}");
        assert_eq!(fused::softmax_mix_flat_values(&scores, &values, dim).unwrap(), want_out);

        // dot_many: `dim` is the query length, the keys are the value rows
        let query: Vec<FixedPoint> = (0..dim).map(|_| rng.small()).collect();
        let mut dots = vec![FixedPoint::ZERO; n];
        fused::dot_many_into(&query, &values, dim, &mut dots);
        assert_eq!(dots, fused::dot_many(&query, &values, dim), "dot_many_into n={n} dim={dim}");
        let each: Vec<FixedPoint> = values.chunks(dim).map(|k| fused::dot(&query, k)).collect();
        assert_eq!(dots, each);
    }
    // empty scores: nothing is written
    let mut out = vec![FixedPoint::one(); 3];
    fused::softmax_mix_flat_values_into(&[], &[], 3, &mut out).unwrap();
    assert_eq!(out, vec![FixedPoint::one(); 3]);
}
