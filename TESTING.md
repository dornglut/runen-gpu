# Testing and validation

## Canonical command

```text
cargo validate
```

This command is implemented by the repository-local `xtask` and is the merge
readiness baseline for this template.

## Baseline checks

Validation fails closed when:

- a required template authority file is missing;
- Rust formatting is not clean;
- workspace tests fail;
- Clippy emits warnings;
- rustdoc emits warnings;
- `git diff --check` reports whitespace errors;
- validation changes repository state.

The validator starts from a clean repository and verifies that the repository
remains unchanged after the checks.

## CI

The workflow in `.github/workflows/validation.yml` is intentionally thin. It
pins the accepted `dornglut/github-workflows` reusable Rust validation workflow
to an immutable commit and delegates meaning to `cargo +stable validate`.

The shared workflow proves the exact caller feature head before validation and
provisions stable plus any Cargo-declared `rust-version` values needed by the
checked-out repository.

## Local versus independent evidence

Local validation is preparation. Pull-request acceptance requires independent
repository-owned CI against the exact reviewed feature head.

Product-specific target matrices, dependency policy, benchmarks, native/browser
proofs, and downstream conformance workloads do not belong in this generic
baseline.
