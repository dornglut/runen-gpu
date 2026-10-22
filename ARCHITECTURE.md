# RunenGPU architecture

## Dependency direction

```text
Runenwerk host/integration adapters
    -> RunenRender semantic rendering
        -> RunenGPU generic GPU execution
            -> private WGPU realization
```

Other compute or rendering consumers may use RunenGPU directly. RunenGPU has no
upward dependency on consumers. RunenShader supplies shader authoring/compilation
to consumers; RunenGPU admits canonical WGSL independently of that sibling.

## Ownership

RunenGPU owns backend-neutral GPU contracts for capabilities, resources,
programs, work graphs, submission, transfers, physical surface configuration,
and device outcomes. `src/api/**` defines those contracts;
`src/backend/wgpu/**` implements them privately. `src/lib.rs` exports only
the backend-neutral API. Native implementation/characterization tests may use
WGPU without making it part of the public contract.

RunenGPU does not own renderer image formation, scenes, materials, lighting,
visibility, exposure, tone/gamut mapping or other presentation *policy*; ECS,
UI, world and application behavior; host scheduling and window/event loops;
shader asset discovery/watch/reload policy; product recovery; or image/video
artifact persistence. Physical surface facts, acquisition and Present execution
remain RunenGPU responsibilities.

## Work lifecycle

A consumer requests a context with explicit capability requirements, declares
logical resources and admitted shader programs, and builds one or more GPU work
fragments. Graph preparation validates dependencies, resource access and
initialization. Context-bound preparation validates the admitted device and
realizes physical work. Submission returns an observable ticket; accepted,
completed, failed, readback and presentation facts are not interchangeable.
Retained resources require explicit continuity and reconstruction handling;
the framework does not schedule or retry application work.

## Repository boundary

RunenGPU has one product package: `runen-gpu` / `runen_gpu`. `xtask` is
tooling, not a second product. The independent
[`conformance/downstream`](conformance/downstream) program uses only the
public crate path dependency to verify the exact 4097-element inclusive and
exclusive prefix-scan results.
There are no compatibility, forwarding, mirror or template-synchronization
paths.

The completed cross-repository extraction and cutover are recorded in
[`BOOTSTRAP.md`](BOOTSTRAP.md) and governed by Engineering ADR-0008.
