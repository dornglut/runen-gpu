# RunenGPU bootstrap and provenance

This record captures stable bootstrap facts and the boundary before extraction.
It is not a branch, pull-request, workflow-run, or current-head ledger.

## Repository recreation

The originally reserved empty shell was deleted and recreated from the accepted
template. The repository identity therefore changed:

```text
prior empty-shell repository ID: 1358301376
current recreated repository ID: 1359449274
```

The deleted shell contained no commits, refs, package, implementation, release,
or semantic authority. Its deletion and recreation did not duplicate source
authority.

## Accepted template provenance

```text
template repository: dornglut/rust-framework-template
accepted template commit: 500461d51fe155febc806e288e5bc013e413a785
accepted template tree: 1e1ae24713cd48b5ea2c3fe1da87cf8dd8f8358a
generated RunenGPU initial commit: e30d05729ed7ac9cabdfb649ad8af6212b8cc0b3
generated RunenGPU initial tree: 1e1ae24713cd48b5ea2c3fe1da87cf8dd8f8358a
```

The generated initial tree exactly matched the accepted template tree, and
generated-repository validation passed during bootstrap proof. The historical
template material originated under the Apache-2.0 framework-template grant; that
historical grant remains historical and is not an ongoing synchronization or
architecture authority.

## Product decisions

```text
package: runen-gpu
crate: runen_gpu
version: 0.1.0
edition: 2024
MSRV: 1.87
publish: false
features: default=[]
license: GPL-3.0-only
```

RunenGPU is a standalone product repository with one framework package. `xtask`
is repository tooling only. Runenwerk remains the sole semantic implementation
authority during this bootstrap; this issue transfers no implementation.

The repository classification is `profile=rust-framework`, `lifecycle=active`,
and `contribution=owner-only`, with public visibility and `main` as the default
branch. These are repository posture decisions, not implementation authority.

## Intentional deviations from the accepted template

1. RunenGPU repository, package, and crate identity.
2. Initial standalone SemVer `0.1.0`.
3. Product MSRV `1.87` rather than the template tooling baseline.
4. GPL-3.0-only current representation and `LICENSING.md`.
5. RunenGPU-local README, architecture, testing, agent, and bootstrap guidance.
6. RunenGPU validation and workflow identity.
7. Product identity, license, and MSRV validation guards.
8. Implementation-empty RunenGPU crate documentation.
9. RunenGPU repository profile, settings, and owner-only contribution posture.
10. Removal of the template `unsafe_code = "forbid"` lint because GX did not
    accept that source constraint; no replacement unsafe-code policy is added.

## Future extraction provenance

The later, separately authorized extraction has this stable boundary:

```text
predecessor: dornglut/runenwerk
transfer boundary: engine/src/plugins/gpu/**
predecessor origin: 5bbdab36ae661d99432bfe5d215062c397aac975
accepted GX census base: a27dbf341220205e69f8adfc92617d08646c8165
```

Engineering #9 transfers zero RunenGPU implementation source and establishes
only repository authority/readiness. The later extraction must be owned by a
RunenGPU-local issue and follow ADR-0008: accepted Runenwerk implementation,
unmerged successor candidate, successor acceptance, then exact-pin cutover and
predecessor deletion in Runenwerk.
