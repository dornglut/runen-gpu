# Testing and validation

## Canonical command

```text
cargo validate
```

The repository-local `xtask` owns this merge-readiness baseline. It checks
standalone source/dependency boundaries, required files and license identity;
locked Cargo metadata and dependency trees (including Wasm); formatting,
workspace tests, strict Clippy and rustdoc; the independent downstream consumer;
Rust 1.87 MSRV; Git whitespace; and unchanged repository state. Start from a
clean checkout.

These checks are separate from GPU conformance. An API being available does not
prove that a specific adapter or platform supports it.

## Proof map

This table points to the executable sources of truth. Historical G/R milestone
codes in test filenames identify retained proof entrypoints, **not** architectural
owners. Details of expected pixels, buffers, limits and failure categories are
asserted by the named tests; this table does not replace those assertions.

| Area | Test or proof owner | Evidence and limits |
| --- | --- | --- |
| API contracts | `tests/gpu_*contracts.rs`, source unit tests | Resource, capability, shader, pipeline, graph, submission and failure contracts. |
| Retained submission continuity | `gpu_g5_retained_rejection_native`, source-owned retained-continuity tests | First failed resource/cause/generation, reservation masking, content/reconstruction validity, read-only conflicts and fresh post-terminal submission. No automatic retry or downstream Runenwerk acceptance claim. |
| Submission identity and lifecycle | `gpu_execution_lifecycle` | Opaque node membership across composed fragments and submission clones. A failed ticket retaining node IDs does not prove execution succeeded. |
| Compute and multi-frame | `gpu_prefix_scan_native`, `gpu_game_of_life_native`, `gpu_reaction_diffusion_native`, `gpu_boids_native` | Exact 4097-element prefix scan, 160×90 Game of Life final state and 17-frame visual sequence, persistent reaction-diffusion and bounded boids. |
| Rendering and transfers | `gpu_g5b_native_render`, `gpu_initial_content_native`, `gpu_offscreen_indexed_native`, `gpu_compute_generated_indirect_native` | GPU transfer/initial-content and exact-readback offscreen/indirect results. |
| Raster and input | `gpu_r1_*vertex*`, `gpu_r2_*`, `gpu_render_pipeline_state_contracts` | Vertex formats, depth/stencil, blend, depth bias, depth clipping, anisotropy and dual-source blending. Hardware-dependent features require advertised admission. |
| Texture format families | `gpu_r1_compressed_texture_formats`, `gpu_r1_optional_format_roles`, related `gpu_r1_*` | Exact block/role/copy/readback semantics for 14 BC, 10 ETC2/EAC and 28 ASTC LDR formats; optional Rgba32Float filter/blend and BGRA8 storage roles. ASTC HDR and sliced-3D compression are not claimed. |
| Fixed binding arrays | `gpu_fixed_binding_array_native`, `gpu_q_metal_qualification`, `gpu_browser_webgpu` | Qualified per resource class on the observed adapter. The pinned Vulkan software driver has a narrow storage-buffer-array device-loss exception; actual-browser WebGPU proves typed unsupported admission for unavailable native arrays. |
| Divergent indexing | `gpu_r3_binding_array_non_uniform_indexing_native`, Metal and browser suites | Compiler-derived requirements and fixed-array occupancy; execute supported texture/sampler, storage-buffer and storage-texture routing. Browser absence is a typed rejection, not a successful dispatch. |
| Shader extensions | `gpu_r3_shader_f16`, `gpu_r3_clip_distances`, `gpu_r3_primitive_index` | Capability discovery and exact execution where supported; optional unsupported adapters prove typed rejection. Generic Metal is not M3-specific evidence. |
| Indirect and layered rendering | `gpu_r4_indirect_first_instance`, `gpu_r4_multiview` | Zero-only versus admitted nonzero first instance; contiguous 2..=31 multiview and independent multisample-array requirements, layered resolves and depth usage where supported. No sparse masks, layered discard or general transient layered-array claim. |
| Transient attachments | `gpu_transient_attachment_native` and cross-platform suites | Normalized color/MSAA resolve/depth semantics; stencil only when the format role is advertised. |
| Physical surface and Present | `gpu_g7a2_native_surface`, `gpu_browser_webgpu/surface_probe.rs` | Acquire/clear/Present/reacquire lifecycle and per-format public physical color-space selection. Chrome's DisplayP3 execution proof does not measure actual display gamut, HDR, encoding or color accuracy. |
| Downstream public consumer | `conformance/downstream` | Independent exact inclusive/exclusive prefix scan using only the public RunenGPU package path dependency. |
| Non-gating characterization | `gpu_direct_wgpu_cost_native`, `gpu_graph_scale` | Direct-WGPU/CPU timing and structural scaling reports. Hosted timing ratios are evidence, not passing performance budgets or proof of public-API GPU cost. |

## Conformance environments and CI

- `.github/workflows/validation.yml` is a thin immutable caller of
  `cargo +stable validate`; the repository owns its meaning.
- `.github/workflows/runengpu-conformance.yml` executes native software
  Vulkan (Mesa Lavapipe on Ubuntu with Xvfb), exact native readback oracles,
  Wasm compilation and **actual Chrome WebGPU execution**. Browser surface
  qualification uses a valid Xvfb/SwiftShader/Vulkan compositor; Wasm
  compilation alone does not prove Present. Applicable JSON/PNG artifacts
  record exact revision and environment.
- `.github/workflows/runengpu-metal-qualification.yml` runs the public
  qualification suite against a confirmed Metal adapter on a hosted
  Apple-Silicon macOS runner. It records correlated normalized device/adapter
  facts and bounded private WGPU characterization, including supported/
  unsupported capability dispositions. It preserves conservative Metal
  `TimestampQuery` suppression.
- `scripts/qualify-m3.sh` is the trusted **owner-run actual M3** path.
  It requires a clean exact macOS arm64 checkout and detected Apple
  M3/M3 Pro/M3 Max hardware, runs the correlated public Metal suite and writes
  a sanitized report under ignored `target/` state by default. Machine/chip
  and adapter facts are retained, not serial numbers or platform UUIDs.

The proof levels are distinct: native software Vulkan, actual-browser WebGPU,
generic hosted Metal, owner-run actual M3 and non-gating performance
characterization. No one level silently qualifies another.

For optional capabilities, the conformance rule is **execute an exact oracle
when the normalized capability and role are advertised; otherwise require
an explicit unsupported/typed-rejection outcome**. A driver-level failure
or infrastructure restriction is recorded as unqualified rather than
manufactured support.

PR acceptance requires the exact reviewed feature-head checks applicable to
the change, independent review and the repository's protected merge policy.
Do not infer success from a different revision, different adapter, API
existence or an unexecuted test target.
