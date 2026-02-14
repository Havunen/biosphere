#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

BASE_REF="${1:-main}"
NEW_REF="${2:-HEAD}"
# Criterion filter is a positional substring filter (see `cargo bench -- --help`).
CRITERION_FILTER="${3:-}"

BENCHES_DEFAULT="bench_utils bench_tree bench_forest"
BENCHES="${BENCHES:-$BENCHES_DEFAULT}"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${ROOT_DIR}/benchmarks/${RUN_ID}_${BASE_REF//\//-}_vs_${NEW_REF//\//-}"

WORK_DIR="$(mktemp -d "/tmp/biosphere-bench.${RUN_ID}.XXXXXX")"
BASE_WT="${WORK_DIR}/base"
NEW_WT="${WORK_DIR}/new"

cleanup() {
  git -C "${ROOT_DIR}" worktree remove --force "${BASE_WT}" >/dev/null 2>&1 || true
  git -C "${ROOT_DIR}" worktree remove --force "${NEW_WT}" >/dev/null 2>&1 || true
  rm -rf "${WORK_DIR}"
}
trap cleanup EXIT

mkdir -p "${OUT_DIR}/base" "${OUT_DIR}/new"

git -C "${ROOT_DIR}" worktree add --detach "${BASE_WT}" "${BASE_REF}" >/dev/null
git -C "${ROOT_DIR}" worktree add --detach "${NEW_WT}" "${NEW_REF}" >/dev/null

(cd "${BASE_WT}" && {
  echo "ref=${BASE_REF}"
  echo "sha=$(git rev-parse HEAD)"
  cargo -V
  rustc -V
}) > "${OUT_DIR}/base/versions.txt"

(cd "${NEW_WT}" && {
  echo "ref=${NEW_REF}"
  echo "sha=$(git rev-parse HEAD)"
  cargo -V
  rustc -V
}) > "${OUT_DIR}/new/versions.txt"

run_bench() {
  local worktree_dir="$1"
  local bench_name="$2"
  shift 2

  local bench_file="${worktree_dir}/benches/${bench_name}.rs"
  if [[ ! -f "${bench_file}" ]]; then
    echo "Skipping ${bench_name} (missing: ${bench_file})" >&2
    return 0
  fi

  (cd "${worktree_dir}" && cargo bench --bench "${bench_name}" -- --noplot --format terse "$@")
}

echo "==> Running base benchmarks (saving baseline 'base')"
for bench in ${BENCHES}; do
  if [[ -n "${CRITERION_FILTER}" ]]; then
    run_bench "${BASE_WT}" "${bench}" --save-baseline base "${CRITERION_FILTER}" \
      | tee "${OUT_DIR}/base/${bench}.txt"
  else
    run_bench "${BASE_WT}" "${bench}" --save-baseline base \
      | tee "${OUT_DIR}/base/${bench}.txt"
  fi
done

if [[ -d "${BASE_WT}/target/criterion" ]]; then
  mkdir -p "${NEW_WT}/target"
  rm -rf "${NEW_WT}/target/criterion"
  cp -a "${BASE_WT}/target/criterion" "${NEW_WT}/target/criterion"
fi

echo "==> Running new benchmarks (comparing to baseline 'base')"
for bench in ${BENCHES}; do
  if [[ -n "${CRITERION_FILTER}" ]]; then
    run_bench "${NEW_WT}" "${bench}" --baseline-lenient base "${CRITERION_FILTER}" \
      | tee "${OUT_DIR}/new/${bench}.txt"
  else
    run_bench "${NEW_WT}" "${bench}" --baseline-lenient base \
      | tee "${OUT_DIR}/new/${bench}.txt"
  fi
done

cp -a "${BASE_WT}/target/criterion" "${OUT_DIR}/base/criterion" 2>/dev/null || true
cp -a "${NEW_WT}/target/criterion" "${OUT_DIR}/new/criterion" 2>/dev/null || true

echo "==> Saved results to: ${OUT_DIR}"
echo "    Base output: ${OUT_DIR}/base"
echo "    New output:  ${OUT_DIR}/new"
