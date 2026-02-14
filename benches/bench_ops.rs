use biosphere::{
    DecisionTree, DecisionTreeParameters, MaxFeatures, RandomForest, RandomForestParameters,
};

#[cfg(test)]
use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use ndarray::{s, Array1, Array2};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::hint::black_box;

#[allow(non_snake_case)]
fn data(n: usize, d: usize, rng: &mut impl Rng) -> (Array2<f64>, Array1<f64>) {
    let mut X = Array2::<f64>::zeros((n, d));

    for i in 0..d {
        if i % 2 == 0 {
            X.slice_mut(s![.., i])
                .assign(&Array1::from_shape_fn(n, |_| rng.random::<f64>()));
        } else {
            X.slice_mut(s![.., i])
                .assign(&Array1::from_shape_fn(n, |_| {
                    if rng.random_bool(0.3) {
                        1.0
                    } else {
                        0.0
                    }
                }));
        }
    }

    let y = Array1::from_shape_fn(n, |_| rng.random::<f64>())
        + X.column(0)
        + X.column(1).map(|x| x - x * x);

    (X, y)
}

#[allow(non_snake_case)]
pub fn benchmark_ops(c: &mut Criterion) {
    let mut rng = StdRng::seed_from_u64(0);

    // --------------------
    // Decision tree: fit
    // --------------------
    let mut tree_fit = c.benchmark_group("tree_fit");
    for &(n, d, max_depth, max_features) in &[
        (10_000usize, 10usize, 8usize, 10usize),
        (100_000usize, 10usize, 8usize, 10usize),
    ] {
        let (X, y) = data(n, d, &mut rng);
        let X_view = X.view();
        let y_view = y.view();
        tree_fit.throughput(Throughput::Elements(n as u64));

        let params = DecisionTreeParameters::default()
            .with_max_depth(Some(max_depth))
            .with_max_features(MaxFeatures::Value(max_features));

        tree_fit.bench_with_input(
            BenchmarkId::from_parameter(format!("n={}, d={}", n, d)),
            &(X_view, y_view),
            |b, (X, y)| {
                b.iter(|| {
                    let mut tree = DecisionTree::new(params.clone());
                    tree.fit(X, y);
                    black_box(tree)
                })
            },
        );
    }
    tree_fit.finish();

    // -----------------------
    // Decision tree: predict
    // -----------------------
    let mut tree_predict = c.benchmark_group("tree_predict");
    for &(n, d, max_depth, max_features) in &[
        (10_000usize, 10usize, 8usize, 10usize),
        (100_000usize, 10usize, 8usize, 10usize),
    ] {
        let (X, y) = data(n, d, &mut rng);
        let X_view = X.view();
        let y_view = y.view();
        tree_predict.throughput(Throughput::Elements(n as u64));

        let params = DecisionTreeParameters::default()
            .with_max_depth(Some(max_depth))
            .with_max_features(MaxFeatures::Value(max_features));

        let mut tree = DecisionTree::new(params);
        tree.fit(&X_view, &y_view);

        tree_predict.bench_with_input(
            BenchmarkId::from_parameter(format!("n={}, d={}", n, d)),
            &X_view,
            |b, X| b.iter(|| black_box(tree.predict(X))),
        );
    }
    tree_predict.finish();

    // ---------------------
    // Random forest: fit
    // ---------------------
    let mut forest_fit = c.benchmark_group("forest_fit");
    forest_fit.sample_size(10);
    for &(n, d, n_estimators, max_depth) in &[(100_000usize, 10usize, 100usize, 8usize)] {
        let (X, y) = data(n, d, &mut rng);
        let X_view = X.view();
        let y_view = y.view();
        forest_fit.throughput(Throughput::Elements(n as u64));

        for &n_jobs in &[1i32, 4i32] {
            let params = RandomForestParameters::default()
                .with_n_estimators(n_estimators)
                .with_max_depth(Some(max_depth))
                .with_n_jobs(Some(n_jobs));

            forest_fit.bench_with_input(
                BenchmarkId::new(format!("n_jobs={}", n_jobs), format!("n={}, d={}", n, d)),
                &(X_view, y_view),
                |b, (X, y)| {
                    b.iter(|| {
                        let mut forest = RandomForest::new(params.clone());
                        forest.fit(X, y);
                        black_box(forest)
                    })
                },
            );
        }
    }
    forest_fit.finish();

    // ------------------------
    // Random forest: predict
    // ------------------------
    let mut forest_predict = c.benchmark_group("forest_predict");
    for &(n, d, n_estimators, max_depth) in &[(100_000usize, 10usize, 100usize, 8usize)] {
        let (X, y) = data(n, d, &mut rng);
        let X_view = X.view();
        let y_view = y.view();
        forest_predict.throughput(Throughput::Elements(n as u64));

        let params = RandomForestParameters::default()
            .with_n_estimators(n_estimators)
            .with_max_depth(Some(max_depth))
            .with_n_jobs(Some(4));

        let mut forest = RandomForest::new(params);
        forest.fit(&X_view, &y_view);

        forest_predict.bench_with_input(
            BenchmarkId::from_parameter(format!("n={}, d={}", n, d)),
            &X_view,
            |b, X| b.iter(|| black_box(forest.predict(X))),
        );
    }
    forest_predict.finish();
}

criterion_group!(
    name = bench_ops;
    config = Criterion::default().sample_size(10);
    targets = benchmark_ops
);
criterion_main!(bench_ops);
