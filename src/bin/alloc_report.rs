use biosphere::{
    DecisionTree, DecisionTreeParameters, MaxFeatures, RandomForest, RandomForestParameters,
};
use ndarray::{s, Array1, Array2};
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

struct CountingAlloc;

static ALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static ALLOC_ZEROED_CALLS: AtomicU64 = AtomicU64::new(0);
static DEALLOC_CALLS: AtomicU64 = AtomicU64::new(0);
static REALLOC_CALLS: AtomicU64 = AtomicU64::new(0);

static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
static DEALLOC_BYTES: AtomicU64 = AtomicU64::new(0);
static REALLOC_OLD_BYTES: AtomicU64 = AtomicU64::new(0);
static REALLOC_NEW_BYTES: AtomicU64 = AtomicU64::new(0);

#[global_allocator]
static GLOBAL_ALLOCATOR: CountingAlloc = CountingAlloc;

unsafe impl GlobalAlloc for CountingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        System.alloc(layout)
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOC_ZEROED_CALLS.fetch_add(1, Ordering::Relaxed);
        ALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        System.alloc_zeroed(layout)
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        DEALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        DEALLOC_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        System.dealloc(ptr, layout)
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        REALLOC_CALLS.fetch_add(1, Ordering::Relaxed);
        REALLOC_OLD_BYTES.fetch_add(layout.size() as u64, Ordering::Relaxed);
        REALLOC_NEW_BYTES.fetch_add(new_size as u64, Ordering::Relaxed);
        System.realloc(ptr, layout, new_size)
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct AllocSnapshot {
    alloc_calls: u64,
    alloc_zeroed_calls: u64,
    dealloc_calls: u64,
    realloc_calls: u64,
    alloc_bytes: u64,
    dealloc_bytes: u64,
    realloc_old_bytes: u64,
    realloc_new_bytes: u64,
}

fn snapshot() -> AllocSnapshot {
    AllocSnapshot {
        alloc_calls: ALLOC_CALLS.load(Ordering::Relaxed),
        alloc_zeroed_calls: ALLOC_ZEROED_CALLS.load(Ordering::Relaxed),
        dealloc_calls: DEALLOC_CALLS.load(Ordering::Relaxed),
        realloc_calls: REALLOC_CALLS.load(Ordering::Relaxed),
        alloc_bytes: ALLOC_BYTES.load(Ordering::Relaxed),
        dealloc_bytes: DEALLOC_BYTES.load(Ordering::Relaxed),
        realloc_old_bytes: REALLOC_OLD_BYTES.load(Ordering::Relaxed),
        realloc_new_bytes: REALLOC_NEW_BYTES.load(Ordering::Relaxed),
    }
}

fn reset_counters() {
    ALLOC_CALLS.store(0, Ordering::Relaxed);
    ALLOC_ZEROED_CALLS.store(0, Ordering::Relaxed);
    DEALLOC_CALLS.store(0, Ordering::Relaxed);
    REALLOC_CALLS.store(0, Ordering::Relaxed);
    ALLOC_BYTES.store(0, Ordering::Relaxed);
    DEALLOC_BYTES.store(0, Ordering::Relaxed);
    REALLOC_OLD_BYTES.store(0, Ordering::Relaxed);
    REALLOC_NEW_BYTES.store(0, Ordering::Relaxed);
}

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

#[derive(Clone, Debug)]
struct RunConfig {
    op: String,
    n: usize,
    d: usize,
    iters: usize,
    seed: u64,
    max_depth: usize,
    max_features: usize,
    n_estimators: usize,
    n_jobs: i32,
}

fn usage() -> ! {
    eprintln!(
        "Usage:\n  cargo run --release --bin alloc_report -- <op> [options]\n\n\
Ops:\n  tree_fit | tree_predict | forest_fit | forest_predict | forest_fit_predict_oob\n\n\
Options:\n  --n <usize>              (default: 100000)\n  --d <usize>              (default: 10)\n  --iters <usize>          (default: 1)\n  --seed <u64>             (default: 0)\n  --max-depth <usize>      (default: 8)\n  --max-features <usize>   (default: d)\n  --n-estimators <usize>   (default: 100)\n  --n-jobs <i32>           (default: 4)\n"
    );
    std::process::exit(2);
}

fn parse_args() -> RunConfig {
    let mut args = std::env::args().skip(1);
    let op = match args.next() {
        Some(op) if op == "--help" || op == "-h" => usage(),
        Some(op) => op,
        None => usage(),
    };

    let mut cfg = RunConfig {
        op,
        n: 100_000,
        d: 10,
        iters: 1,
        seed: 0,
        max_depth: 8,
        max_features: 0, // 0 means "use d" below
        n_estimators: 100,
        n_jobs: 4,
    };

    while let Some(flag) = args.next() {
        match flag.as_str() {
            "--n" => {
                cfg.n = args
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
            "--d" => {
                cfg.d = args
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
            "--iters" => {
                cfg.iters = args
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
            "--seed" => {
                cfg.seed = args
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
            "--max-depth" => {
                cfg.max_depth = args
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
            "--max-features" => {
                cfg.max_features = args
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
            "--n-estimators" => {
                cfg.n_estimators = args
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
            "--n-jobs" => {
                cfg.n_jobs = args
                    .next()
                    .unwrap_or_else(|| usage())
                    .parse()
                    .unwrap_or_else(|_| usage())
            }
            _ => usage(),
        }
    }

    if cfg.iters == 0 {
        eprintln!("--iters must be >= 1");
        std::process::exit(2);
    }

    if cfg.max_features == 0 {
        cfg.max_features = cfg.d;
    }

    cfg
}

fn run_iters<F: FnMut()>(iters: usize, mut f: F) -> Duration {
    let start = Instant::now();
    for _ in 0..iters {
        f();
    }
    start.elapsed()
}

fn main() {
    let cfg = parse_args();
    let mut rng = StdRng::seed_from_u64(cfg.seed);

    let (x, y) = data(cfg.n, cfg.d, &mut rng);
    let x_view = x.view();
    let y_view = y.view();

    let elapsed = match cfg.op.as_str() {
        "tree_fit" => {
            let params = DecisionTreeParameters::default()
                .with_max_depth(Some(cfg.max_depth))
                .with_max_features(MaxFeatures::Value(cfg.max_features));

            reset_counters();
            run_iters(cfg.iters, || {
                let mut tree = DecisionTree::new(params.clone());
                tree.fit(&x_view, &y_view);
                std::hint::black_box(tree);
            })
        }
        "tree_predict" => {
            let params = DecisionTreeParameters::default()
                .with_max_depth(Some(cfg.max_depth))
                .with_max_features(MaxFeatures::Value(cfg.max_features));
            let mut tree = DecisionTree::new(params);
            tree.fit(&x_view, &y_view);

            reset_counters();
            run_iters(cfg.iters, || {
                let preds = tree.predict(&x_view);
                std::hint::black_box(preds);
            })
        }
        "forest_fit" => {
            let params = RandomForestParameters::default()
                .with_n_estimators(cfg.n_estimators)
                .with_max_depth(Some(cfg.max_depth))
                .with_n_jobs(Some(cfg.n_jobs));

            reset_counters();
            run_iters(cfg.iters, || {
                let mut forest = RandomForest::new(params.clone());
                forest.fit(&x_view, &y_view);
                std::hint::black_box(forest);
            })
        }
        "forest_predict" => {
            let params = RandomForestParameters::default()
                .with_n_estimators(cfg.n_estimators)
                .with_max_depth(Some(cfg.max_depth))
                .with_n_jobs(Some(cfg.n_jobs));
            let mut forest = RandomForest::new(params);
            forest.fit(&x_view, &y_view);

            reset_counters();
            run_iters(cfg.iters, || {
                let preds = forest.predict(&x_view);
                std::hint::black_box(preds);
            })
        }
        "forest_fit_predict_oob" => {
            let params = RandomForestParameters::default()
                .with_n_estimators(cfg.n_estimators)
                .with_max_depth(Some(cfg.max_depth))
                .with_n_jobs(Some(cfg.n_jobs));

            reset_counters();
            run_iters(cfg.iters, || {
                let mut forest = RandomForest::new(params.clone());
                let preds = forest.fit_predict_oob(&x_view, &y_view);
                std::hint::black_box(preds);
            })
        }
        _ => usage(),
    };

    let stats = snapshot();
    let nanos = elapsed.as_nanos() as u128;

    // Single-line JSON for easy diffing/parsing.
    println!(
        "{{\"op\":\"{}\",\"n\":{},\"d\":{},\"iters\":{},\"seed\":{},\"max_depth\":{},\"max_features\":{},\"n_estimators\":{},\"n_jobs\":{},\"elapsed_ns\":{},\"alloc_calls\":{},\"alloc_zeroed_calls\":{},\"dealloc_calls\":{},\"realloc_calls\":{},\"alloc_bytes\":{},\"dealloc_bytes\":{},\"realloc_old_bytes\":{},\"realloc_new_bytes\":{}}}",
        cfg.op,
        cfg.n,
        cfg.d,
        cfg.iters,
        cfg.seed,
        cfg.max_depth,
        cfg.max_features,
        cfg.n_estimators,
        cfg.n_jobs,
        nanos,
        stats.alloc_calls,
        stats.alloc_zeroed_calls,
        stats.dealloc_calls,
        stats.realloc_calls,
        stats.alloc_bytes,
        stats.dealloc_bytes,
        stats.realloc_old_bytes,
        stats.realloc_new_bytes
    );
}
