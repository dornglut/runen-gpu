# RunenGPU roadmap

This document is the durable long-term sequence authority for RunenGPU.
It describes the generic GPU capability surface RunenGPU intends to own and the
order in which that surface should mature. Live priority, issue state, pull
requests, assignees, and validation runs belong to GitHub and the Dornglut
Engineering Portfolio rather than this file.

A roadmap entry does **not** authorize implementation by itself. Investigation
and delivery work still requires an accepted issue in the repository that owns
the behavior, based on then-current accepted source and writer state.

RunenGPU is a backend-neutral GPU execution framework. Renderer, scene, material,
lighting, UI, world, application, and product policy remain above this boundary.
WGPU remains a private implementation authority below it.

## Capability policy

RunenGPU proactively covers mature generic GPU functionality. A downstream
consumer does not need to be blocked before ordinary GPU vocabulary can enter the
roadmap.

A capability is eligible for proactive planning when all of the following hold:

1. it is generic GPU vocabulary owned by RunenGPU rather than renderer, product,
   or application policy;
2. its semantics are mature enough to normalize independently of WGPU naming;
3. a backend-neutral public contract is clear;
4. capability and limit admission can report support truthfully rather than
   manufacture guarantees;
5. private WGPU realization is available or has a bounded implementation path;
6. focused public-API conformance can prove the contract.

A bounded investigation or concrete consumer pressure is still required when the
correct abstraction is unresolved, the concept carries policy, portability
semantics are divergent, or the underlying facility remains experimental or
backend-specific.

RunenGPU does not mirror the WGPU public API or feature-bit set. WGPU capability
existence is evidence, not automatic RunenGPU authority.

## Classification

Capability maturity and roadmap disposition are separate questions.

| Class | Meaning |
| --- | --- |
| `CORE` | Mature generic GPU vocabulary RunenGPU is expected to provide proactively. |
| `ADVANCED` | Mature generic vocabulary that is capability-gated, has narrower platform support, or needs a larger coherent contract. |
| `DEFERRED` | Semantics, portability, standards, or backend maturity are not yet sufficient for stable RunenGPU authority. |
| `OUT-OF-SCOPE` | The concept belongs outside RunenGPU's public semantic boundary. |

Roadmap dispositions are:

| Disposition | Meaning |
| --- | --- |
| `CURRENT` | The accepted RunenGPU contract already provides the intended semantic capability. |
| `PLAN` | The capability family is intentionally part of the durable roadmap. |
| `DEFER` | Keep the capability visible, but wait for its stated maturity or ownership gate. |
| `OUT-OF-SCOPE` | Deliberately exclude it from public RunenGPU authority. |

Optional hardware support does not imply optional roadmap support. An `ADVANCED`
capability may still be a deliberate `PLAN` item while truthful adapter/device
admission determines whether a particular context can use it.

## Durable sequence

```text
accepted backend-neutral execution foundation
    -> R0 contract integrity and truthful admission
        -> R1 resource and data vocabulary completeness
            -> R2 sampling and raster completeness
            -> R3 shader and binding capability completeness
            -> R4 queries and GPU-driven/layered execution
            -> R5 modern physical presentation
        -> R6 safe interop

R7 measured execution optimization is pressure/measurement driven.
Q  platform/backend/browser qualification runs continuously across all phases.
```

R2 and R3 may advance independently after their shared R1 foundations are
accepted and normal writer/concurrency rules permit it. R4 layered work depends
on authoritative texture-view semantics. R5 can advance once the physical format
vocabulary it needs is accepted. R7 must not become a gate that delays semantic
capability breadth.

### R0 — Contract integrity and truthful admission

Goal: every already-public semantic has one authority from construction through
admission, realization, execution, and proof.

- remove contradictions between public texture-view semantics and realized view
  dimensions;
- keep one normalized sample-count authority across resources and pipelines;
- normalize limits required to admit RunenGPU-owned public descriptors, including
  relevant buffer, texture-dimension/array, vertex-input, binding, and compute
  limits;
- keep unrelated backend limits private instead of copying WGPU's complete limit
  structure;
- route newly discovered existing-contract defects as correctness work rather
  than hiding them inside capability expansion.

### R1 — Resource and data vocabulary completeness

Goal: ordinary modern GPU data representation should not require repeated
one-off framework expansion.

- coherent portable scalar/vector integer, float, normalized, and packed texture
  format families;
- practical 16-bit and 32-bit render/storage/intermediate formats;
- complete depth/stencil resource vocabulary;
- authoritative D1/D2/D2-array/cube/cube-array/D3 view semantics;
- portable vertex-format families with correct per-format alignment rather than
  assumptions tied to the initial float32/u32/i32 set;
- capability-gated BC, ETC2, and ASTC compression families with truthful block,
  copy, and usage semantics;
- preserve per-format sampled/storage/render/copy role admission from actual
  device facts.

### R2 — Sampling and raster completeness

Goal: provide the mature generic fixed-function vocabulary expected by modern
renderers without importing renderer policy.

- sampler anisotropy with correct validation;
- full blend factors and operations with independent color/alpha state;
- depth bias, slope scale, and applicable clamp semantics;
- complete stencil front/back operations and masks;
- mature portable optional raster capabilities behind truthful capability gates,
  such as depth-clip control and dual-source blending when their prerequisites are
  satisfied;
- keep native-only raster modes deferred until a stable RunenGPU contract is
  justified.

### R3 — Shader and binding capability completeness

Goal: normalized program requirements should drive device admission instead of
forcing callers to know backend feature bits.

- retain canonical WGSL as the public program-source authority;
- add `f16` and other mature standardized optional WGSL capabilities through
  normalized requirement discovery/admission;
- integrate optional float/filter/blend and shader built-in requirements with
  existing format/capability authority rather than creating redundant policy;
- preserve truthful fixed binding-array admission;
- broaden array/indexing semantics only when their portable contract and support
  maturity justify it.

Subgroups, subgroup-size control, immediates, native integer/f64 extensions, and
similar facilities remain deferred while WebGPU/WGPU semantics or deployment are
not sufficiently converged for stable RunenGPU authority.

### R4 — Queries and GPU-driven/layered execution

Goal: cover mature generic GPU-driven execution without adding renderer meaning.

- occlusion queries alongside existing timestamp query authority;
- standardized indirect-first-instance support;
- multiview and multisampled-array/layered execution as advanced capability-gated
  contracts after texture-view correctness is established;
- add further multi-draw or bindless-style execution only when their portability
  and semantic model meet the normal admission policy.

### R5 — Modern physical presentation

Goal: expose the physical surface facts required for modern SDR, wide-gamut, and
HDR presentation while keeping image-formation policy above RunenGPU.

- per-format surface capability facts rather than only formats usable with an
  automatic color space;
- explicit backend-neutral surface color-space selection;
- wide-gamut/HDR-capable physical formats and modes where the platform reports
  them;
- truthful qualification on platforms that actually support each path.

RunenGPU owns physical surface capability and configuration. Exposure, tone
mapping, gamut mapping, radiance interpretation, visualization policy, and other
color-management decisions remain RunenRender or product authority.

### R6 — Safe interop

Goal: allow external ownership only through normalized owner-safe contracts.

- derive a decision-complete imported-resource source contract covering lifetime,
  context/device affinity, usage facts, failure semantics, and ownership before
  authorizing its implementation;
- add external media textures only after implementation and portability maturity
  support a stable contract;
- never use raw WGPU, Vulkan, DX12, or Metal handles as a public escape hatch.

### R7 — Measured execution optimization

Goal: improve cost without creating duplicate semantic authority.

Candidate work includes reusable graph preparation, transient-resource aliasing,
pass fusion, dead-work elimination, command preparation/scheduling, and private
backend caches. These remain private or derived optimizations unless measurement
proves value and an observable public contract genuinely requires new semantics.

Explicit public backend queues and public backend pipeline-cache objects are not a
roadmap objective.

## Capability disposition summary

This is intentionally family-level rather than a mirror of WGPU's feature list.

| Family | Class | Disposition |
| --- | --- | --- |
| Backend-neutral resource/work/submission/surface execution foundation | `CORE` | `CURRENT` |
| Contract integrity and normalized admission limits | `CORE` | `PLAN` — R0 |
| Portable texture/view/vertex/depth-stencil vocabulary | `CORE` | `PLAN` — R1 |
| BC/ETC2/ASTC compression | `ADVANCED` | `PLAN` — R1, capability-gated |
| Anisotropy and complete portable raster/blend/depth/stencil state | `CORE` | `PLAN` — R2 |
| WGSL `f16` and mature standardized optional shader features | `ADVANCED` | `PLAN` — R3 |
| Fixed binding arrays | `ADVANCED` | `CURRENT`, retain truthful admission |
| Partially-bound/non-uniform/bindless-style native extensions | `ADVANCED` | `DEFER` pending portable contract or concrete advanced-native pressure |
| Occlusion queries | `CORE` | `PLAN` — R4 |
| Indirect-first-instance | `ADVANCED` | `PLAN` — R4 |
| Multiview and multisampled arrays | `ADVANCED` | `PLAN` — R4 after view/resource foundations |
| Multi-draw-count and pipeline statistics | `ADVANCED` | `DEFER` while backend/platform scope remains narrow |
| Explicit surface color spaces and wide-gamut/HDR physical presentation | `ADVANCED` | `PLAN` — R5 |
| Normalized imported-resource contract | `ADVANCED` | `PLAN` — R6, contract investigation precedes implementation |
| External media textures | `ADVANCED` | `DEFER` until backend/portable maturity improves |
| Subgroups and subgroup-size control | `ADVANCED` | `DEFER` until WebGPU/WGPU semantics and conformance converge |
| Immediates | `ADVANCED` | `DEFER` until standards/deployment maturity is sufficient |
| Ray tracing/query, mesh shaders, cooperative matrices | `DEFERRED` | `DEFER` while WGPU treats the facilities as experimental |
| Raw backend resources/devices/queues/handles | `OUT-OF-SCOPE` | `OUT-OF-SCOPE` |
| Public passthrough shader languages | `OUT-OF-SCOPE` | `OUT-OF-SCOPE` under canonical WGSL authority |
| Renderer/product exposure, tone mapping, gamut and image policy | `OUT-OF-SCOPE` | `OUT-OF-SCOPE` |

A deferred row is a deliberate decision, not a forgotten gap. When its revisit
condition becomes true, reassess it against the same capability policy rather
than automatically promoting it.

## Continuous qualification — Q

API existence and qualification are separate. `TESTING.md` owns the current proof
portfolio; this roadmap only defines how qualification should broaden.

Maintain the deterministic software/native and browser proof oracles, then expand
qualification as reproducible infrastructure permits toward:

1. real Vulkan hardware;
2. Windows DX12;
3. macOS Metal;
4. browser WebGPU surface/presentation;
5. a broader browser/backend matrix only when it provides maintainable evidence.

Do not claim cross-platform support from a WGPU enum, one backend implementation,
or one passing adapter. Every new capability needs focused public-API proof at the
appropriate semantic boundary; broader platform qualification grows continuously
alongside the roadmap.

## Refresh rule

Re-audit external capability coverage when one of these events occurs:

- an accepted WGPU **major** dependency upgrade;
- WebGPU or WGSL stabilizes a materially relevant capability family;
- a roadmap phase is about to close and its external maturity assumptions matter;
- a `DEFER` revisit condition becomes true.

Patch releases and calendar intervals alone do not require roadmap churn. Update
this authority only when the semantic disposition or durable sequence materially
changes.
