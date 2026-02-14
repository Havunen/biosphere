use crate::tree::{DecisionTree, DecisionTreeParameters, MaxFeatures};
use crate::utils::{argsort, sample_indices_from_weights, sample_weights};
use ndarray::{Array1, ArrayView1, ArrayView2};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};
use rayon::ThreadPoolBuilder;

#[derive(Clone, Debug)]
pub struct RandomForestParameters {
    decision_tree_parameters: DecisionTreeParameters,
    n_estimators: usize,
    seed: u64,
    // The number of jobs to run in parallel for `fit` and `fit_predict_oob`.
    // `None` means 1. `-1` means using all processors.
    n_jobs: Option<i32>,
}

impl Default for RandomForestParameters {
    fn default() -> Self {
        RandomForestParameters {
            decision_tree_parameters: DecisionTreeParameters::default(),
            n_estimators: 100,
            seed: 0,
            n_jobs: None,
        }
    }
}

impl RandomForestParameters {
    pub fn new(
        n_estimators: usize,
        seed: u64,
        max_depth: Option<usize>,
        max_features: MaxFeatures,
        min_samples_leaf: usize,
        min_samples_split: usize,
        n_jobs: Option<i32>,
    ) -> Self {
        RandomForestParameters {
            decision_tree_parameters: DecisionTreeParameters::new(
                max_depth,
                max_features,
                min_samples_split,
                min_samples_leaf,
                0,
            ),
            n_estimators,
            seed,
            n_jobs,
        }
    }

    pub fn with_n_estimators(mut self, n_estimators: usize) -> Self {
        self.n_estimators = n_estimators;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn with_max_depth(mut self, max_depth: Option<usize>) -> Self {
        self.decision_tree_parameters = self.decision_tree_parameters.with_max_depth(max_depth);
        self
    }

    pub fn with_max_features(mut self, max_features: MaxFeatures) -> Self {
        self.decision_tree_parameters = self
            .decision_tree_parameters
            .with_max_features(max_features);
        self
    }

    pub fn with_min_samples_leaf(mut self, min_samples_leaf: usize) -> Self {
        self.decision_tree_parameters = self
            .decision_tree_parameters
            .with_min_samples_leaf(min_samples_leaf);
        self
    }

    pub fn with_min_samples_split(mut self, min_samples_split: usize) -> Self {
        self.decision_tree_parameters = self
            .decision_tree_parameters
            .with_min_samples_split(min_samples_split);
        self
    }

    pub fn with_n_jobs(mut self, n_jobs: Option<i32>) -> Self {
        self.n_jobs = n_jobs;
        self
    }
}

pub struct RandomForest {
    random_forest_parameters: RandomForestParameters,
    trees: Vec<DecisionTree>,
}

impl Default for RandomForest {
    fn default() -> Self {
        RandomForest::new(RandomForestParameters::default())
    }
}

impl RandomForest {
    pub fn new(random_forest_parameters: RandomForestParameters) -> Self {
        RandomForest {
            random_forest_parameters,
            trees: Vec::new(),
        }
    }

    pub fn predict(&self, X: &ArrayView2<f64>) -> Array1<f64> {
        assert!(
            !self.trees.is_empty(),
            "RandomForest has not been fitted yet"
        );
        let mut predictions = Array1::<f64>::zeros(X.nrows());
        let normalization = self.trees.len() as f64;

        for (prediction, row) in predictions.iter_mut().zip(X.outer_iter()) {
            let mut sum = 0.0;
            for tree in &self.trees {
                sum += tree.predict_row(&row);
            }
            *prediction = sum / normalization;
        }

        predictions
    }

    pub fn fit(&mut self, X: &ArrayView2<f64>, y: &ArrayView1<f64>) {
        let mut thread_pool_builder = ThreadPoolBuilder::new();

        // If n_jobs = 1 or None, use a single process. If n_jobs = -1, use all processes.
        let n_jobs_usize = match self.random_forest_parameters.n_jobs {
            Some(n_jobs) => {
                if n_jobs >= 1 {
                    Some(n_jobs as usize)
                } else {
                    None
                }
            }
            None => Some(1),
        };

        if let Some(n_jobs) = n_jobs_usize {
            thread_pool_builder = thread_pool_builder.num_threads(n_jobs);
        }

        let thread_pool = thread_pool_builder.build().unwrap();

        let indices: Vec<Vec<usize>> = thread_pool.install(|| {
            (0..X.ncols())
                .into_par_iter()
                .map(|idx| argsort(&X.column(idx)))
                .collect()
        });

        let mut rng = StdRng::seed_from_u64(self.random_forest_parameters.seed);
        let seeds: Vec<u64> = (0..self.random_forest_parameters.n_estimators)
            .map(|_| rng.random::<u64>())
            .collect();

        self.trees = thread_pool.install(|| {
            seeds
                .into_par_iter()
                .map(|seed| {
                    let mut rng = StdRng::seed_from_u64(seed);
                    let mut tree = DecisionTree::new(
                        self.random_forest_parameters
                            .decision_tree_parameters
                            .clone()
                            .with_random_state(seed),
                    );

                    let weights = sample_weights(X.nrows(), &mut rng);
                    let mut samples = sample_indices_from_weights(&weights, &indices);

                    let samples_as_slices = samples.iter_mut().map(|x| x.as_mut_slice()).collect();
                    tree.fit_with_sorted_samples(X, y, samples_as_slices);
                    tree
                })
                .collect()
        })
    }

    pub fn fit_predict_oob(&mut self, X: &ArrayView2<f64>, y: &ArrayView1<f64>) -> Array1<f64> {
        let mut thread_pool_builder = ThreadPoolBuilder::new();

        // If n_jobs = 1 or None, use a single process. If n_jobs = -1, use all processes.
        let n_jobs_usize = match self.random_forest_parameters.n_jobs {
            Some(n_jobs) => {
                if n_jobs >= 1 {
                    Some(n_jobs as usize)
                } else {
                    None
                }
            }
            None => Some(1),
        };

        if let Some(n_jobs) = n_jobs_usize {
            thread_pool_builder = thread_pool_builder.num_threads(n_jobs);
        }

        let thread_pool = thread_pool_builder.build().unwrap();
        let nrows = X.nrows();
        let ncols = X.ncols();

        let mut rng = StdRng::seed_from_u64(self.random_forest_parameters.seed);
        let seeds: Vec<u64> = (0..self.random_forest_parameters.n_estimators)
            .map(|_| rng.random::<u64>())
            .collect();

        let tree_parameters = self
            .random_forest_parameters
            .decision_tree_parameters
            .clone();

        let (mut trees, mut oob_sum, oob_count) = thread_pool.install(move || {
            let indices: Vec<Vec<usize>> = (0..ncols)
                .into_par_iter()
                .map(|idx| argsort(&X.column(idx)))
                .collect();

            seeds
                .into_par_iter()
                .enumerate()
                .fold(
                    || {
                        (
                            Vec::<(usize, DecisionTree)>::new(),
                            vec![0.0; nrows],
                            vec![0usize; nrows],
                        )
                    },
                    |(mut trees, mut oob_sum, mut oob_count), (tree_idx, seed)| {
                        let mut rng = StdRng::seed_from_u64(seed);
                        let mut tree = DecisionTree::new(
                            tree_parameters
                                .clone()
                                .with_random_state(rng.random::<u64>()),
                        );

                        let weights = sample_weights(nrows, &mut rng);
                        let mut samples = sample_indices_from_weights(&weights, &indices);

                        let samples_as_slices =
                            samples.iter_mut().map(|x| x.as_mut_slice()).collect();
                        tree.fit_with_sorted_samples(X, y, samples_as_slices);

                        for (sample, &weight) in weights.iter().enumerate() {
                            if weight == 0 {
                                oob_sum[sample] += tree.predict_row(&X.row(sample));
                                oob_count[sample] += 1;
                            }
                        }

                        trees.push((tree_idx, tree));
                        (trees, oob_sum, oob_count)
                    },
                )
                .reduce(
                    || {
                        (
                            Vec::<(usize, DecisionTree)>::new(),
                            vec![0.0; nrows],
                            vec![0usize; nrows],
                        )
                    },
                    |(mut trees_a, mut sum_a, mut count_a), (trees_b, sum_b, count_b)| {
                        trees_a.extend(trees_b);
                        for (a, b) in sum_a.iter_mut().zip(sum_b) {
                            *a += b;
                        }
                        for (a, b) in count_a.iter_mut().zip(count_b) {
                            *a += b;
                        }
                        (trees_a, sum_a, count_a)
                    },
                )
        });

        trees.sort_by_key(|(idx, _)| *idx);
        self.trees = trees.into_iter().map(|(_, tree)| tree).collect();

        for (prediction, count) in oob_sum.iter_mut().zip(oob_count) {
            if count > 0 {
                *prediction /= count as f64;
            } else {
                *prediction = f64::NAN;
            }
        }

        Array1::from_vec(oob_sum)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testing::load_iris;
    use ndarray::{s, Array1, Array2};
    use std::collections::HashSet;

    #[test]
    fn test_predict_matches_average_of_tree_predictions() {
        let data = load_iris();
        let X = data.slice(s![.., 0..4]);
        let y = data.slice(s![.., 4]);

        let random_forest_parameters =
            RandomForestParameters::new(10, 0, Some(4), MaxFeatures::Sqrt, 1, 2, Some(1));
        let mut forest = RandomForest::new(random_forest_parameters);
        forest.fit(&X, &y);

        let predictions = forest.predict(&X);

        let mut manual = Array1::<f64>::zeros(X.nrows());
        for tree in forest.trees.iter() {
            let tree_predictions = tree.predict(&X);
            for (a, b) in manual.iter_mut().zip(tree_predictions.iter()) {
                *a += b;
            }
        }
        let normalization = forest.trees.len() as f64;
        manual.mapv_inplace(|x| x / normalization);

        assert_eq!(predictions, manual);
    }

    #[test]
    fn test_random_forest_parameters_new_argument_order() {
        let parameters = RandomForestParameters::new(1, 0, None, MaxFeatures::None, 7, 9, Some(1));

        assert_eq!(parameters.decision_tree_parameters.min_samples_leaf, 7);
        assert_eq!(parameters.decision_tree_parameters.min_samples_split, 9);
    }

    #[test]
    fn test_fit_predict_oob_sets_nan_when_no_oob_predictions() {
        let data = load_iris();
        let X = data.slice(s![.., 0..4]);
        let y = data.slice(s![.., 4]);

        let random_forest_parameters =
            RandomForestParameters::new(1, 0, Some(4), MaxFeatures::Sqrt, 1, 2, Some(1));
        let mut forest = RandomForest::new(random_forest_parameters);
        let oob_predictions = forest.fit_predict_oob(&X, &y);

        assert!(oob_predictions.iter().any(|x| x.is_nan()));
        assert!(!oob_predictions.iter().any(|x| x.is_infinite()));
    }

    #[test]
    fn test_fit_predict_oob_does_not_append_trees_on_refit() {
        let data = load_iris();
        let X = data.slice(s![.., 0..4]);
        let y = data.slice(s![.., 4]);

        let random_forest_parameters =
            RandomForestParameters::new(3, 0, Some(4), MaxFeatures::Sqrt, 1, 2, Some(1));
        let mut forest = RandomForest::new(random_forest_parameters);
        let _ = forest.fit_predict_oob(&X, &y);
        assert_eq!(forest.trees.len(), 3);

        let _ = forest.fit_predict_oob(&X, &y);
        assert_eq!(forest.trees.len(), 3);
    }

    #[test]
    fn test_predicts_three_classes_on_separable_data() {
        let n = 90;
        let X = Array2::from_shape_fn((n, 1), |(i, _)| i as f64);
        let y = Array1::from_shape_fn(n, |i| {
            if i < 30 {
                0.
            } else if i < 60 {
                1.
            } else {
                2.
            }
        });

        let random_forest_parameters =
            RandomForestParameters::new(30, 0, Some(2), MaxFeatures::None, 1, 2, Some(1));
        let mut forest = RandomForest::new(random_forest_parameters);
        forest.fit(&X.view(), &y.view());

        let predictions = forest.predict(&X.view());
        let predicted_classes: HashSet<i32> =
            predictions.iter().map(|x| x.round() as i32).collect();
        assert!(predicted_classes.contains(&0));
        assert!(predicted_classes.contains(&1));
        assert!(predicted_classes.contains(&2));

        assert_eq!(predictions[0].round(), 0.);
        assert_eq!(predictions[45].round(), 1.);
        assert_eq!(predictions[89].round(), 2.);

        let correct = predictions
            .iter()
            .zip(y.iter())
            .filter(|(pred, target)| pred.round() == **target)
            .count();
        let accuracy = correct as f64 / n as f64;
        assert!(
            accuracy >= 0.95,
            "Expected accuracy >= 0.95, got {}",
            accuracy
        );
    }

    #[test]
    fn test_refit_changes_class_predictions() {
        let n = 20;
        let X = Array2::from_shape_fn((n, 1), |(i, _)| i as f64);
        let y0 = Array1::<f64>::zeros(n);
        let y2 = Array1::<f64>::from_elem(n, 2.);

        let random_forest_parameters =
            RandomForestParameters::new(5, 0, Some(1), MaxFeatures::None, 1, 2, Some(1));
        let mut forest = RandomForest::new(random_forest_parameters);

        forest.fit(&X.view(), &y0.view());
        let predictions0 = forest.predict(&X.view());
        assert!(predictions0.iter().all(|&x| x == 0.));

        forest.fit(&X.view(), &y2.view());
        let predictions2 = forest.predict(&X.view());
        assert!(predictions2.iter().all(|&x| x == 2.));
    }
}
