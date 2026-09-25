#!/usr/bin/env bash
set -euo pipefail

root="$(git rev-parse --show-toplevel)"
cd "$root"

status="$(git status --porcelain --untracked-files=normal)"
if [[ -n "$status" ]]; then
  printf 'M3 qualification requires a clean checkout:\n%s\n' "$status" >&2
  exit 1
fi

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "M3 qualification requires macOS" >&2
  exit 1
fi
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "M3 qualification requires Apple-Silicon arm64" >&2
  exit 1
fi

revision="$(git rev-parse HEAD)"
report_dir="${RUNEN_GPU_M3_REPORT_DIR:-$root/target/runengpu-m3-qualification}"
report="${report_dir}/report.json"
mkdir -p "$report_dir"

export WGPU_BACKEND=metal
export RUNEN_GPU_PROOF_REVISION="$revision"
export RUNEN_GPU_QUALIFICATION_MODE=m3
export RUNEN_GPU_QUALIFICATION_REPORT="$report"

cargo +stable test --test gpu_q_metal_qualification \
  metal_qualification_records_exact_public_api_evidence \
  --locked -- --ignored --nocapture --test-threads=1

test -s "$report"

status="$(git status --porcelain --untracked-files=normal)"
if [[ -n "$status" ]]; then
  printf 'M3 qualification changed repository state:\n%s\n' "$status" >&2
  exit 1
fi

printf 'RunenGPU actual-M3 qualification report: %s\n' "$report"
