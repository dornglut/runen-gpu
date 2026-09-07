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
