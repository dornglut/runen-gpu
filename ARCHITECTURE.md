# RunenGPU architecture

## Dependency direction

```text
Runenwerk host/integration adapters
    -> RunenRender semantic rendering
        -> RunenGPU generic GPU execution
            -> private backend implementation
```

Independent non-render consumers may also depend directly downward on RunenGPU.
RunenGPU has no upward dependency on Runenwerk or domain integration.

## Ownership

RunenGPU owns reusable GPU execution semantics: backend-neutral public contracts
for capabilities, resources, work, submission, uploads/readback, surfaces, and
device outcomes. WGPU is the accepted private backend realization; public
contracts remain backend-neutral and expose no general raw-WGPU escape hatch.

The source boundary is explicit: `src/api/**` is the public contract surface,
`src/backend/wgpu/**` is private realization, and `src/lib.rs` re-exports only
the backend-neutral API. Tests and examples may use WGPU directly when they
are proving native realization or characterization; that is not a public crate
dependency surface.

RunenGPU does not own renderer image formation; scene, material, lighting,
visibility, or presentation semantics; ECS, UI, world, or application behavior;
application scheduling; window/event-loop ownership; shader filesystem/watch,
reload, or last-known-good policy; product recovery; or persisted PNG, video, or
other artifact policy.

## Repository boundary

RunenGPU is one product package: `runen-gpu` / `runen_gpu`. `xtask` is tooling,
not a second product or release boundary. Dependency direction is one-way and
there are no compatibility, forwarding, mirror, or template-synchronization
paths.

The independent downstream proof depends only on the package's public path
dependency and computes the accepted 4097-element inclusive and exclusive
prefix-scan oracle. It does not import `wgpu`, Runenwerk, or a workspace
dependency.

## Authority-transfer history

During bootstrap and the unmerged extraction candidate, Runenwerk remained the
sole RunenGPU semantic implementation authority. Under ADR-0008, accepted
successor default-branch publication switched semantic authority to
`dornglut/runen-gpu`; the Runenwerk predecessor then became frozen and
deletion-bound until the exact-revision downstream cutover completed.

That transfer is now complete. `dornglut/runen-gpu` is the sole RunenGPU semantic
implementation authority. Runenwerk consumes an exact accepted RunenGPU revision
and retains only downstream integration; the predecessor RunenGPU source and
namespace were deleted. The paragraph above records historical authority
transition, not an active migration state.
