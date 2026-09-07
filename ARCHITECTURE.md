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
device outcomes. The accepted extraction target uses WGPU as a private backend
realization; public contracts remain backend-neutral and expose no general raw-
WGPU escape hatch.

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

## Authority transfer

During bootstrap and an unmerged extraction candidate, Runenwerk remains the sole
semantic implementation authority. Under ADR-0008, accepted successor default-
branch publication switches authority to RunenGPU; the predecessor then remains
only as a frozen, deletion-bound copy until exact-revision downstream cutover
and predecessor deletion are accepted.
