# Optimizations and Safety Review

## Summary

- No Rust `unsafe` blocks found.
- Main wins came from removing avoidable allocations in prediction/OOB aggregation and from replacing `partial_cmp(...).unwrap()` with a total ordering.

## Findings (implemented)

1. `RandomForest::predict` allocated in a hot loop
   - Issue: `predictions = predictions + tree.predict(X)` allocates an intermediate prediction array for every tree and also allocates a new sum array every iteration.
   - Fix: accumulate directly into a single `predictions` buffer and normalize in place.
   - Implementation: `src/forest.rs`

2. `RandomForest::fit` had avoidable allocations while preparing inputs
   - Issue: built a temporary `Vec<usize>` for feature indices and manually built `Vec<&mut [usize]>` for samples.
   - Fix: use `(0..X.ncols()).into_par_iter()` directly and build the slice vector via iterator.
   - Implementation: `src/forest.rs`

3. `RandomForest::fit_predict_oob` had correctness + memory issues
   - Issues:
     - Collected per-tree OOB prediction vectors before reduction (high peak memory for large `n_estimators` / `nrows`).
     - Appended to `self.trees` on each call (refit grows the forest unexpectedly).
     - Divided by zero when a sample had zero OOB estimators (produced `inf`).
   - Fixes:
     - Use parallel `fold`/`reduce` to aggregate OOB sums + counts without storing all per-tree predictions.
     - Replace `self.trees` with the newly-trained set on every call.
     - Emit `NaN` for samples with `count == 0`.
   - Implementation: `src/forest.rs`

4. Sorting helpers could panic on NaNs
   - Issue: `partial_cmp(...).unwrap()` panics if an input contains NaNs.
   - Fix: use `f64::total_cmp` with `sort_unstable_by`.
   - Implementation: `src/utils.rs`

5. Python bindings had small avoidable allocations + a parameter-order bug
   - Issues:
     - `max_features` parsing extracted a `String` when `&str` suffices.
     - `RandomForestParameters::new` was called with `min_samples_split`/`min_samples_leaf` swapped.
   - Fixes:
     - Extract `&str` for string parsing.
     - Pass `min_samples_leaf`/`min_samples_split` in the correct order.
   - Implementation: `biosphere-py/src/utils.rs`, `biosphere-py/src/random_forest.rs`

## Tests added

- `src/utils.rs`: `test_argsort_does_not_panic_on_nan`
- `src/forest.rs`: `test_predict_matches_average_of_tree_predictions`
- `src/forest.rs`: `test_random_forest_parameters_new_argument_order`
- `src/forest.rs`: `test_fit_predict_oob_sets_nan_when_no_oob_predictions`
- `src/forest.rs`: `test_fit_predict_oob_does_not_append_trees_on_refit`
