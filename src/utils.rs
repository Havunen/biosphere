use ndarray::{ArrayBase, Data, Ix1, Ix2};
use rand::Rng;

/// Compute `indices` such that `data.select(indices)` is sorted.
///
/// Parameters
/// ----------
/// data: Array1<f64> or ArrayView1<f64>
pub fn argsort(data: &ArrayBase<impl Data<Elem = f64>, Ix1>) -> Vec<usize> {
    let mut indices = (0..data.len()).collect::<Vec<usize>>();
    indices.sort_unstable_by(|a, b| data[*a].total_cmp(&data[*b]));
    indices
}

pub fn sample_weights(n: usize, rng: &mut impl Rng) -> Vec<usize> {
    let mut counts = vec![0; n];

    for _ in 0..n {
        counts[rng.random_range(0..n)] += 1
    }

    counts
}

pub fn sample_indices_from_weights(weights: &[usize], indices: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let mut samples = Vec::<Vec<usize>>::with_capacity(indices.len());

    for feature_indices in indices {
        let mut sample = Vec::<usize>::with_capacity(feature_indices.len());
        for &index in feature_indices {
            for _ in 0..weights[index] {
                sample.push(index);
            }
        }
        samples.push(sample);
    }
    samples
}

pub fn oob_samples_from_weights(weights: &[usize]) -> Vec<usize> {
    let mut oob_samples = Vec::<usize>::with_capacity(weights.len());

    for (idx, &weight) in weights.iter().enumerate() {
        if weight == 0 {
            oob_samples.push(idx);
        }
    }
    oob_samples
}

pub fn sorted_samples(
    X: &ArrayBase<impl Data<Elem = f64>, Ix2>,
    samples: &[usize],
) -> Vec<Vec<usize>> {
    let mut samples_out: Vec<Vec<usize>> = Vec::with_capacity(X.ncols());

    for idx in 0..X.ncols() {
        let mut samples_ = samples.to_vec();
        samples_.sort_unstable_by(|a, b| X[[*a, idx]].total_cmp(&X[[*b, idx]]));
        samples_out.push(samples_);
    }
    samples_out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::is_sorted;
    use ndarray::{Array1, Array2, Axis};
    use rand::rngs::StdRng;
    use rand::SeedableRng;
    use std::cmp::Ordering;

    #[test]
    fn test_argsort() {
        let seed = 7;
        let mut rng = StdRng::seed_from_u64(seed);
        let n = 100;
        let x = Array1::from_shape_fn(n, |_| rng.random::<f64>());

        let indices = argsort(&x);
        assert!(is_sorted(&x.select(Axis(0), &indices)));
    }

    #[test]
    fn test_sample_weights() {
        let seed = 7;
        let mut rng = StdRng::seed_from_u64(seed);
        let n = 100;

        let weights = sample_weights(n, &mut rng);

        assert!(weights.iter().sum::<usize>() == n);
    }

    #[test]
    fn test_sample_indices_from_weights() {
        let seed = 7;
        let mut rng = StdRng::seed_from_u64(seed);
        let n = 100;
        let d = 8;

        let X = Array2::from_shape_fn((n, d), |_| rng.random::<f64>());

        let indices: Vec<Vec<usize>> = (0..d).map(|idx| argsort(&X.column(idx))).collect();
        let weights = sample_weights(n, &mut rng);

        let samples = sample_indices_from_weights(&weights, &indices);

        for feature in 0..X.ncols() {
            assert!(is_sorted(
                &X.column(feature).select(Axis(0), &samples[feature])
            ));
        }
    }

    #[test]
    fn test_argsort_does_not_panic_on_nan() {
        let x = Array1::from_vec(vec![1.0, f64::NAN, -1.0, 0.0]);
        let indices = argsort(&x);
        let sorted = x.select(Axis(0), &indices);
        for w in sorted.windows(2) {
            assert_ne!(w[0].total_cmp(&w[1]), Ordering::Greater);
        }
    }
}
