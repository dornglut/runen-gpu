# RunenGPU

RunenGPU is a standalone Rust framework for backend-neutral GPU execution
contracts. It is intended to own reusable GPU execution semantics while keeping
renderer and application meaning above the framework boundary.

## Maturity

This repository is in GX bootstrap and extraction preparation. It establishes
the standalone repository authority, but it does not yet contain the transferred
RunenGPU implementation. Runenwerk remains the semantic implementation
authority until the later ADR-0008 successor-acceptance switch.

## Boundary

RunenGPU will own reusable GPU execution semantics, resource/work submission
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
the required authority files, formatting, locked workspace tests, strict
Clippy, rustdoc with warnings denied, the declared MSRV, product identity and
license consistency, Git whitespace, and unchanged repository state.

See [TESTING.md](TESTING.md).

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
