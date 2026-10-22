# RunenGPU

RunenGPU is a Rust framework for backend-neutral GPU execution. It provides
reusable resource, program, work-graph and submission contracts without owning
renderer, application or product policy. WGPU is its private implementation.

## Using RunenGPU

The ordinary execution path is:

1. **Request a context.** `GpuContext::request` checks adapter capabilities,
   required limits and backend policy before creating a device.
2. **Declare resources and programs.** `GpuResourceScope` creates logical
   handles; `GpuProgramSourceRegistry` admits canonical WGSL for pipeline
   contracts.
3. **Describe work.** `GpuWorkFragment` contains compute, render, transfer,
   readback and Present operations. Compose and prepare the graph to validate
   resource access, dependencies and initialization.
4. **Prepare and submit.** `GpuContext::prepare_submission` realizes the
   context-bound execution plan; `submit_prepared` returns a submission to
   observe. `GpuContext::submit_work` is a convenience path through the same
   execution authority.
5. **Observe completion.** `GpuContext::progress` advances nonblocking
   observations. Submission acceptance is not GPU completion, readback
   materialization or proof of application-level results.

See [the public crate](src/lib.rs), the
[standalone consumer](conformance/downstream), and the
[compute, render and surface examples](examples). The
[architecture](ARCHITECTURE.md) explains responsibility boundaries.

## Adapter selection

RunenGPU evaluates admitted adapters against requested capabilities, limits,
classes and allowed backends. Primary native backends are considered before
secondary OpenGL fallback when no primary candidate qualifies; an explicit
backend allowlist restricts discovery. Stable observable facts break preference
ties where possible, but physically indistinguishable adapters do not promise
a stable physical choice.

Use `GpuContextDescriptor::with_allowed_backends` or
`with_backend_preference` when the host needs an explicit choice.
`WGPU_BACKEND` is an operator-level backend override. Strict tie rejection is
available through
`GpuAdapterSelectionPolicy::RequireUnambiguous` and can return
`GpuContextRequestErrorCategory::AmbiguousAdapterSelection`.

## Retained resources and failure

Retained data does not acquire automatic scheduling, recovery or retry.
`GpuContext::submit_prepared` may reject changed retained state with
`GpuSubmissionRejectionKind::RetainedContinuityChanged`. Its
`retained_continuity_evidence()` describes the first rejected resource,
context generation and typed cause. `InFlightParticipation` records a
reservation conflict *at acceptance*, including read-only participation; it
does not prove that later conditions would succeed or identify the other
submission. A rejected prepared ticket cannot be retried automatically:
preserve source inputs and freshly prepare after observing the relevant
state.

Descriptor, lifecycle, coverage and reconstruction failures are distinct from
temporary reservation conflicts. Check
`GpuContext::retained_resource_reconstruction_requirement` for
current-generation obligations; zero initialization does not restore lost
semantic data. Unknown future rejection causes must fail closed.

## Package and validation

```text
package: runen-gpu
crate: runen_gpu
version: 0.1.0
edition: 2024
MSRV: 1.87
publish: false
```

Run `cargo validate` for the repository-owned baseline: source/dependency
boundaries, locked builds and tests, strict linting and documentation checks,
independent downstream conformance, MSRV and license checks. Platform-specific
native, browser and Metal proofs remain separate, as described in
[TESTING.md](TESTING.md).

## Authority and policy

- [Architecture](ARCHITECTURE.md)
- [Roadmap](ROADMAP.md)
- [Testing](TESTING.md)
- [Bootstrap and completed transfer provenance](BOOTSTRAP.md)
- [Executor guidance](AGENTS.md)
- [Organization contribution guidance](https://github.com/dornglut/.github/blob/main/CONTRIBUTING.md)
- [Organization security policy](https://github.com/dornglut/.github/blob/main/SECURITY.md)
- [Public license](LICENSE)
- [Commercial-license guidance](LICENSING.md)

## Contribution and license

RunenGPU is the sole semantic implementation authority; Runenwerk owns only
its downstream integration and pins an exact accepted RunenGPU revision.
Tracked-content contributions are currently `owner-only` pending an accepted
contribution mechanism that preserves commercial relicensing rights.

RunenGPU is publicly licensed under [GPL-3.0-only](LICENSE). See
[LICENSING.md](LICENSING.md) for the separately governed commercial path.
