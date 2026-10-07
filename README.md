# RunenGPU

RunenGPU is a standalone Rust framework for backend-neutral GPU execution
contracts. It owns reusable GPU execution semantics while keeping renderer and
application meaning above the framework boundary.

## Maturity

This repository contains the accepted standalone RunenGPU implementation and its
proof portfolio. `dornglut/runen-gpu` is the sole RunenGPU semantic implementation
authority. The public API is backend-neutral; WGPU is a private realization.

The ADR-0008 authority transfer is complete: Runenwerk consumes an exact accepted
RunenGPU revision and its predecessor RunenGPU implementation/namespace has been
deleted. Runenwerk integration remains a separate downstream responsibility and is
not owned by this repository.

## Boundary

RunenGPU owns reusable GPU execution semantics, resource/work submission
contracts, and private backend realization. It does not own renderer image
formation, scene/material/lighting semantics, ECS/UI/world/application behavior,
window/event-loop ownership, shader-file policy, product recovery, or media and
artifact persistence.

## Package

```text
package: runen-gpu
crate: runen_gpu
version: 0.1.0
edition: 2024
MSRV: 1.87
publish: false
```

## Adapter admission

Context creation automatically evaluates native adapters against the declared
capability requirements, limits, host compatibility, and allowed backends/classes.
Ordinary native discovery initializes WGPU's primary backend tier first and only
initializes secondary OpenGL if no primary candidate can satisfy admission. This
avoids activating compatibility drivers that are irrelevant to the selected context.
A non-empty `with_allowed_backends` restriction is applied before WGPU instance
creation, so forbidden backend families are not initialized merely to reject them later.

Within the active tier, RunenGPU chooses the best admitted preference rank. Equally
preferred adapters are resolved by stable observed facts where possible;
observationally identical native handles are interchangeable, so physical selection
across runs is not promised. Numeric hardware vendor and device IDs are diagnostics,
not preference scores.

Applications that need an explicit backend can use
`GpuContextDescriptor::with_backend_preference` or `with_allowed_backends`.
An explicit `WGPU_BACKEND` value remains an exact operator-level backend-set
override. An explicit RunenGPU preference that names OpenGL retains joint discovery
so the preference is not silently weakened.
Strict opt-in tie rejection uses
`with_adapter_selection_policy(GpuAdapterSelectionPolicy::RequireUnambiguous)`
and the process-local candidate retry contract. An explicit strict request may
return `GpuContextRequestErrorCategory::AmbiguousAdapterSelection`.

## Validation

`cargo validate` is the single repository-owned validation command. It verifies
the required authority files, standalone source/dependency boundary, dependency
audit, locked workspace tests, the independent downstream package, strict Clippy,
rustdoc with warnings denied, the declared MSRV, product identity and license
consistency, Git whitespace, and unchanged repository state.

See [TESTING.md](TESTING.md).

The standalone entry points are the public items re-exported from
[`runen_gpu`](src/lib.rs). A small independent consumer is maintained under
[`conformance/downstream`](conformance/downstream), and public compute,
render, runtime-binding, and native-host-surface examples live under
[`examples`](examples).

## Authority and policy

- [Architecture](ARCHITECTURE.md)
- [Roadmap](ROADMAP.md)
- [Testing](TESTING.md)
- [Bootstrap and provenance](BOOTSTRAP.md)
- [Executor guidance](AGENTS.md)
- [Organization contribution guidance](https://github.com/dornglut/.github/blob/main/CONTRIBUTING.md)
- [Organization security policy](https://github.com/dornglut/.github/blob/main/SECURITY.md)
- [Public license](LICENSE)
- [Commercial-license guidance](LICENSING.md)

## Contribution

Tracked-content contributions are currently `owner-only`. Issues, discussion,
reviews, and reproducible reports may still be used through the repository's
public channels. This posture remains until an accepted inbound mechanism
preserves the rights needed for commercial relicensing.

## License

RunenGPU is publicly represented under [GPL-3.0-only](LICENSE). A separately
governed commercial licensing path is described in [LICENSING.md](LICENSING.md).
