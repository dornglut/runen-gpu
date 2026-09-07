# Testing and validation

## Canonical command

```text
cargo validate
```

This command is implemented by the repository-local `xtask` and is the merge
readiness baseline for the bootstrap repository.

## Baseline checks

At bootstrap, this proves repository/package integrity rather than the future
GX-X02 GPU runtime portfolio. It covers:

- a clean starting repository and required authority files;
- rustfmt;
- locked workspace tests;
- strict Clippy;
- rustdoc with warnings denied;
- an executable Rust 1.87 MSRV workspace check;
- product identity and GPL license consistency;
- Git whitespace checks; and
- validation not mutating repository state.

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

Future extraction work separately owns native, Wasm, browser, downstream,
runtime, benchmark, and conformance evidence. No such proof is claimed by this
bootstrap baseline or moved into `runen-gpu` yet.
