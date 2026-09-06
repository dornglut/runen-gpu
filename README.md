# Rust Framework Template

`dornglut/rust-framework-template` is a one-time bootstrap baseline for new
Dornglut Rust framework repositories.

It provides a minimal repository shape, repository-owned validation entrypoint,
bootstrap guidance, and licensing/provenance rules. It is not a product,
framework runtime, or ongoing synchronization authority.

## Maturity

This repository is intentionally minimal bootstrap infrastructure. It proves a
working repository shape and validation baseline; it does not define a product
runtime or public framework API.

## Boundary

The template owns only generic bootstrap infrastructure:

- Rust package and workspace conventions;
- root architecture, testing, and agent entrypoints;
- one canonical `cargo validate` command;
- a thin immutable CI caller;
- bootstrap guidance for identity, ownership, licensing, toolchain, settings,
  validation extensions, conformance, provenance, and deviations.

Generated repositories own their implementation, public API, architecture,
dependencies, compatibility, releases, and product-specific validation.

## Bootstrap

A repository created from this template must resolve its own:

1. repository, package, and crate identity;
2. profile, lifecycle, and contribution classification;
3. public license class and required license files;
4. product MSRV and toolchain;
5. repository visibility, merge policy, branch protection, and security controls;
6. canonical validation extensions;
7. downstream conformance workload when a public framework contract exists;
8. extraction and source provenance when applicable;
9. every intentional deviation from this baseline.

The template is irrelevant after bootstrap. Do not add synchronization or
template-update machinery to generated repositories.

See [BOOTSTRAP.md](BOOTSTRAP.md).

## Validation

`cargo validate` is the single repository-owned validation command.

It verifies the required template authority files, formatting, workspace tests,
Clippy with warnings denied, rustdoc with warnings denied, Git whitespace, and
unchanged repository state.

CI invokes the same command through the accepted immutable
`dornglut/github-workflows` reusable Rust validation workflow.

See [TESTING.md](TESTING.md).

## Architecture and policy

- [Architecture](ARCHITECTURE.md)
- [Testing](TESTING.md)
- [Bootstrap](BOOTSTRAP.md)
- [Agent guide](AGENTS.md)
- [Organization contribution guidance](https://github.com/dornglut/.github/blob/main/CONTRIBUTING.md)
- [Organization security policy](https://github.com/dornglut/.github/blob/main/SECURITY.md)

## License

This template repository is available under the [Apache License 2.0](LICENSE).

A generated repository must select its own product license before accepting
substantive implementation. The template's Apache-2.0 license does not become
the generated product's licensing authority.
