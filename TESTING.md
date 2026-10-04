# Testing and validation

## Canonical command

```text
cargo validate
```

This command is implemented by the repository-local `xtask` and is the merge
readiness baseline for the standalone RunenGPU repository.

## Baseline checks

The validator covers:

- a clean starting repository and required authority files;
- the standalone source/dependency boundary and private-backend rule;
- locked Cargo metadata and normal dependency trees, including the Wasm target;
- rustfmt;
- locked workspace tests;
- the independent downstream 4097-element prefix-scan package;
- strict Clippy;
- rustdoc with warnings denied;
- an executable Rust 1.87 MSRV workspace check;
- product identity and GPL license consistency;
- Git whitespace checks; and
- validation not mutating repository state.

The validator starts from a clean repository and verifies that the repository
remains unchanged after the checks.

## Proof portfolio

The retained proof mapping is intentionally successor-local:

| Evidence | Successor proof |
| --- | --- |
| Public contract | API contract, capability, resource, program, submission, readback, surface, and lifecycle tests under `tests/` |
| Submission occurrence correlation | `gpu_execution_lifecycle` proves exact opaque `GpuWorkNodeId` membership survives multi-fragment graph composition into the returned `GpuSubmission`, distinguishes foreign same-local-number nodes, survives submission cloning, and combines with terminal `Completed` status for successful occurrence evidence while failed submissions retain membership without becoming success evidence. |
| Compute | `gpu_prefix_scan_native` proves exact 4097-element inclusive/exclusive results; `gpu_game_of_life_native` proves the fixed 160x90 final-grid oracle and exact 17-frame compute-to-render visual sequence |
| Render/runtime | G5 transfer and G5R initial-content tests, indexed offscreen known-pattern output, generated indirect drawing, normalized indirect first-instance with zero-only regression plus capability-gated nonzero exact-readback execution, depth-clip control with a color-only clipped-vs-unclipped exact-readback oracle, transient-attachment color/MSAA-resolve/depth execution with conditional Stencil8 coverage, and G7A2 native surface presentation |
| Characterization | The direct-WGPU cost portfolio and graph-preparation scale report remain explicitly direct-WGPU/CPU measurements, separate from the public API contract |
| Fixed binding arrays | Generic Metal qualification is the retained capability-bearing public execution target and records per-resource-class outcomes on the correlated adapter; actual-browser WebGPU proves the native fixed-array features unsupported with zero normalized array limits and typed admission rejection. `gpu_fixed_binding_array_native` is ignored by default but explicitly invoked by retained Conformance: real Vulkan adapters run the full capability-gated suite, while the pinned llvmpipe/Lavapipe target records storage-buffer arrays as `UNQUALIFIED` after an observed driver-level device loss and continues the sampled-texture/sampler/storage-texture families where advertised. |
| Compressed textures | The fourteen BC, ten ETC2/EAC, and twenty-eight ASTC LDR formats have exact public block and role semantics and share the public upload/copy/readback graph, including two array layers and terminal mips. ETC2/EAC retains decoded block sampling into an RGBA8 render target with exact readback for representative 8-byte and 16-byte color and EAC blocks. ASTC LDR proves all fourteen legal 2D block dimensions across both linear and sRGB variants with exact 16-byte physical blocks, exhaustive variable-block upload/copy/readback, terminal-edge mips, decoded 4x4/8x6/12x12 sampling, and a mid-gray linear-vs-sRGB transfer oracle. Actual-Chrome BrowserWebGpu and generic Metal must execute the complete public ASTC path when their correlated direct-WGPU adapters advertise ASTC; retained Vulkan/Lavapipe proves typed absence when the feature is unavailable. ASTC HDR and sliced-3D compression are not qualified by this slice. |
| Optional format roles | `gpu_r1_optional_format_roles` records normalized Rgba32Float filtering and blending and Bgra8Unorm storage-write support. Each advertised role admits an exact public-API shader/readback oracle; absent roles report `UNSUPPORTED`. Retained Vulkan Conformance, generic Metal qualification, and actual-Chrome BrowserWebGpu run the same proof on their correlated adapters. The browser and Metal reports retain exact-revision dispositions. Synthetic admission tests cover the whole R/Rg/Rgba float32 optional-role family, BGRA8 storage roles, portability, and private device-feature closure. |
| Non-uniform fixed binding-array indexing | `gpu_r3_binding_array_non_uniform_indexing` derives requirements from Naga structured uniformity analysis and retains full fixed-array occupancy. Generic Metal qualification executes divergent texture/sampler, storage-buffer, and storage-texture routing on the correlated adapter when the normalized prerequisite sets are advertised. Actual-browser WebGPU proves the native non-uniform capabilities unsupported through typed context-admission rejection. Retained Vulkan Conformance executes capability-gated families and inherits the narrow accepted llvmpipe/Lavapipe fixed storage-buffer-array qualification exception without changing public capability facts. |
| Dual-source blending | `gpu_r2_dual_source_blending` proves compiler-derived `DualSourceBlending` requirements, strict primary/secondary fragment-output parity, typed unsupported admission, and an exact secondary-source blend oracle whose drawn pixel must resolve to green with alpha zero. Retained Vulkan Conformance and generic Metal qualification census the normalized capability and require advertised support to execute the oracle; the retained actual-Chrome WebGPU adapter currently reports the normalized capability unsupported and proves typed rejection rather than fabricating support. |
| Clip distances | `gpu_r3_clip_distances` proves canonical WGSL `enable clip_distances` derives the normalized `ClipDistances` requirement through Naga capability analysis, while use without the extension and arrays beyond the standardized eight-distance bound fail closed. The public render oracle emits positive clip distance for one triangle and negative clip distance for another, requiring exact retained-versus-clipped pixel readback. Retained Vulkan Conformance, actual-Chrome BrowserWebGpu, and generic Metal qualification each census normalized support and require advertised support to execute the oracle; unsupported adapters prove typed required-capability rejection instead. The Metal report correlates normalized `ClipDistances` support with the corresponding private WGPU feature fact. Generic hosted Metal remains generic Metal evidence, not actual-M3 evidence. |
| Primitive index | `gpu_r3_primitive_index` proves canonical WGSL `enable primitive_index` derives the normalized `PrimitiveIndex` requirement through Naga capability analysis, while use without the extension fails closed. The public render oracle draws two primitives in one draw and requires exact red/green readback from fragments keyed by `@builtin(primitive_index)`. Retained Vulkan Conformance, actual-Chrome BrowserWebGpu, and generic Metal qualification each census normalized support and require advertised support to execute the oracle; unsupported adapters prove typed required-capability rejection instead. The Metal report also correlates normalized `PrimitiveIndex` support with the corresponding private WGPU feature fact. Generic hosted Metal remains generic Metal evidence, not actual-M3 evidence. |
| Contiguous multiview | `gpu_r4_multiview` proves the normalized 2..=31 contiguous D2Array contract, exact pass/pipeline state parity, the two-view default workload budget and independent realization-time limit rejection, exact selected-layer graph hazards/initialization, layered-Discard rejection, and `@builtin(view_index)` routing on a nonzero-base two-layer view while preserving a disjoint sentinel layer. Retained Vulkan Conformance and generic Metal qualification census `Multiview` and require every advertised normalized implementation to execute the primary, shader-without-view-index, and clear-only public oracles. Actual-browser WebGPU remains normalized unsupported in the pinned backend and proves both zero normalized max-view count and typed feature/limit admission rejection. |
| Layered multisample arrays / MSAA | `gpu_r4_multiview` also proves the independent normalized `MultisampleArray` gate for owned multisampled D2 arrays, contiguous layered-MSAA pass/pipeline parity, nonzero-base two-layer color rendering and layered resolve into a distinct single-sampled D2Array, exact resolved colors, and preservation of a disjoint sentinel layer. When `Depth32Float` advertises the normalized depth/stencil role, the same retained pass includes a matching multisampled layered depth attachment and requires `DepthAttachment` admission. Retained Vulkan Conformance executes every advertised path; generic Metal qualification runs the same correlated public oracle and binds color/depth dispositions into its exact-revision report. Pinned actual-browser WebGPU remains normalized unsupported for `MultisampleArray` and proves typed rejection. The owner-run actual-M3 harness invokes this same Metal qualification test when used; hosted generic Metal evidence does not establish an M3 claim. |
| Browser/Wasm | `gpu_browser_webgpu` compiles for `wasm32-unknown-unknown` and executes the compute/offscreen plus transient-attachment color/MSAA-resolve/depth proof in Chrome WebGPU, proves native fixed binding arrays unsupported with zero normalized array limits and typed admission rejection, exercises transient Stencil8 when the normalized role is advertised, and records the normalized dual-source-blending disposition; the retained Chrome adapter currently reports `DualSourceBlending` unsupported and the proof requires typed feature-admission rejection. The retained browser presentation probe runs under the workflow-owned Xvfb + SwiftShader/Vulkan compositor path. On one public RunenGPU surface it queries per-format color-space facts, then configures, acquires, clears, presents, and reacquires first an advertised sRGB pair and then an advertised `Rgba8Unorm` or `Bgra8Unorm` + `DisplayP3` pair. The exact-head JSON artifact records the selected public pair and completed lifecycle. A separate pinned-WGPU canvas retains the direct `Rgba8Unorm + DisplayP3` characterization as diagnostic evidence. This proves public physical pair selection and execution in that browser environment; it does not measure display gamut, HDR output, image encoding, or color accuracy. |
| Downstream | `conformance/downstream` uses only the public crate API for the 4097-element prefix scan |

Native GPU assertions are run on Ubuntu 24.04 with Mesa Lavapipe and Xvfb in
the successor-owned conformance workflow. Browser evidence is actual Chrome
WebGPU execution, not only Wasm compilation. Browser presentation evidence additionally
uses an Xvfb-backed SwiftShader/Vulkan compositor path; the workflow retains this
environment explicitly because canvas acquisition without a valid presentation/compositor
path is not accepted as surface qualification. Generated PNG/JSON artifacts are
retained by CI for the relevant bounded portfolios.

## CI

The workflow in `.github/workflows/validation.yml` is intentionally thin. It
pins the accepted `dornglut/github-workflows` reusable Rust validation workflow
to an immutable commit and delegates meaning to `cargo +stable validate`.

The shared workflow proves the exact caller feature head before validation and
provisions stable plus any Cargo-declared `rust-version` values needed by the
checked-out repository. The successor-owned
`.github/workflows/runengpu-conformance.yml` adds the native Lavapipe,
Wasm/browser, artifact, and independent-downstream proof jobs.

## Metal and Apple M3 qualification

`.github/workflows/runengpu-metal-qualification.yml` adds a separate generic
Metal qualification lane on GitHub-hosted Apple-Silicon macOS. The workflow
does not treat the runner label as support evidence: the retained test requires
an actual normalized `GpuBackendFamily::Metal` context, records the exact Git
revision plus macOS/architecture/hardware, sanitized adapter facts, normalized
adapter/device/workload limits, normalized capability support/enabled facts, and a bounded
private-WGPU feature/limit characterization correlated to the same Metal adapter. Ordinary Metal
qualification retains zero fixed-array device/workload budgets; separate adapter-correlated
capability-bearing contexts execute supported fixed-array resource classes. The lane also executes
the exact 4097-element prefix-scan oracle in both modes, executes indexed and
compute-generated-indirect offscreen exact-readback oracles, executes the normalized indirect-first-instance zero path plus capability-gated nonzero oracle, reuses the retained
8-bit/16-bit/packed vertex suites, blend and baseline depth-bias suites, executes the
color-only depth-clip-control semantic oracle on the correlated Metal adapter, realizes
anisotropic sampling, executes the retained transient-attachment color/MSAA-resolve/depth
suite plus transient Stencil8 when the normalized role is advertised, records fixed-array
storage-buffer/uniform-buffer/sampled-texture/sampler/storage-texture outcomes as
`EXERCISED` or `UNSUPPORTED`. It also executes the normalized non-uniform fixed binding-array
texture/sampler, storage-buffer, and storage-texture oracles on the correlated adapter and requires
each advertised family to report `EXERCISED`, then retains the JSON report. The same correlated-adapter qualification also records `DualSourceBlending` support and requires the exact secondary-source RGB/alpha readback oracle to report `EXERCISED` whenever that normalized capability is advertised, otherwise `UNSUPPORTED`. It additionally records the normalized `Multiview` maximum and requires all three retained multiview public oracles to report `EXERCISED` whenever support is advertised; absence remains `UNSUPPORTED`. It also preserves the
current conservative Metal `TimestampQuery` suppression.

Generic hosted Metal evidence is not Apple M3 evidence. The trusted owner-run
actual-M3 path is:

```text
scripts/qualify-m3.sh
```

That harness requires a clean exact checkout on macOS arm64, reuses the same
public-API qualification test, normalized capability/limit evidence, and the
bounded private-WGPU characterization, requires the runtime chip identity to be
Apple M3/M3 Pro/M3 Max, writes its report under
ignored `target/` state by default,
and verifies that the repository remains clean. The report intentionally
records model/chip and sanitized GPU adapter facts without retaining serial
numbers or platform UUIDs.

The proof levels are therefore distinct:

- retained portable oracle: Linux Lavapipe/Vulkan plus actual-browser WebGPU;
- generic Metal qualification: hosted Apple-Silicon execution with a proven
  Metal adapter;
- actual M3 qualification: trusted owner-run evidence on an identified M3-class
  machine;
- performance characterization: non-gating measurement, separate from semantic
  support authority.

Local validation is preparation. Pull-request acceptance requires the
repository-owned exact-head workflows applicable to the change.
