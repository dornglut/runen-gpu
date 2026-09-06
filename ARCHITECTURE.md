# Architecture

## Boundary

`rust-framework-template` is bootstrap infrastructure. Its purpose is to
provide a minimal, reusable starting shape for a new Dornglut Rust framework
repository.

It owns no runtime behavior, public product contract, domain model, renderer,
GPU subsystem, ECS, service, application, or product data.

## Repository shape

The baseline consists of:

- a tiny non-published root Rust library used to prove the package baseline;
- a local `xtask` that owns canonical validation;
- a thin immutable shared-workflow caller;
- root agent, architecture, testing, and bootstrap documentation;
- the repository's Apache-2.0 license.

No empty taxonomy directories or product-specific modules are created.

## Dependency direction

```text
repository source
    └── root package

validation authority
    └── xtask
          └── cargo / git commands

CI orchestration
    └── dornglut/github-workflows
          └── cargo +stable validate
```

The reusable workflow orchestrates validation but does not define its meaning.
The `xtask` is repository-local validation authority.

## Generated repositories

A generated repository replaces the placeholder package identity and source,
selects its own license and toolchain contract, establishes its repository
settings, and extends validation only for proven product-specific requirements.

After bootstrap, the template is not an architectural dependency and must not
remain a synchronization authority.
