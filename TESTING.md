# Testing and validation

## Canonical command

```text
cargo validate
```

This command is implemented by the repository-local `xtask` and is the merge
readiness baseline for the standalone RunenGPU repository.

## Baseline checks

The validator covers:

- a clean starting repository and required authority files;
- the standalone source/dependency boundary and private-backend rule;
- locked Cargo metadata and normal dependency trees, including the Wasm target;
- rustfmt;
- locked workspace tests;
- the independent downstream 4097-element prefix-scan package;
- strict Clippy;
- rustdoc with warnings denied;
- an executable Rust 1.87 MSRV workspace check;
- product identity and GPL license consistency;
- Git whitespace checks; and
- validation not mutating repository state.

The validator starts from a clean repository and verifies that the repository
remains unchanged after the checks.

## Proof portfolio

The retained proof mapping is intentionally successor-local:

| Evidence | Successor proof |
| --- | --- |
| Public contract | API contract, capability, resource, program, submission, readback, surface, and lifecycle tests under `tests/` |
| Compute | `gpu_prefix_scan_native` proves exact 4097-element inclusive/exclusive results; `gpu_game_of_life_native` proves the fixed 160x90 final-grid oracle and exact 17-frame compute-to-render visual sequence |
| Render/runtime | G5 transfer and G5R initial-content tests, indexed offscreen known-pattern output, generated indirect drawing, and G7A2 native surface presentation |
| Characterization | The direct-WGPU cost portfolio and graph-preparation scale report remain explicitly direct-WGPU/CPU measurements, separate from the public API contract |
| Browser/Wasm | `gpu_browser_webgpu` compiles for `wasm32-unknown-unknown` and executes the compute/offscreen proof in Chrome WebGPU |
| Downstream | `conformance/downstream` uses only the public crate API for the 4097-element prefix scan |

Native GPU assertions are run on Ubuntu 24.04 with Mesa Lavapipe and Xvfb in
the successor-owned conformance workflow. Browser evidence is actual Chrome
WebGPU execution, not only Wasm compilation. Generated PNG/JSON artifacts are
retained by CI for the relevant bounded portfolios.

## CI

The workflow in `.github/workflows/validation.yml` is intentionally thin. It
pins the accepted `dornglut/github-workflows` reusable Rust validation workflow
to an immutable commit and delegates meaning to `cargo +stable validate`.

The shared workflow proves the exact caller feature head before validation and
provisions stable plus any Cargo-declared `rust-version` values needed by the
checked-out repository. The successor-owned
`.github/workflows/runengpu-conformance.yml` adds the native Lavapipe,
Wasm/browser, artifact, and independent-downstream proof jobs.

Local validation is preparation. Pull-request acceptance requires both
repository-owned workflows against the exact reviewed feature head.
