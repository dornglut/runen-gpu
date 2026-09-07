//! Native G6-I02 conformance harness for the retained bounded boids workload.
//!
//! The reviewable RunenGPU authoring surface lives in `workload.rs`.

mod workload;

use runen_gpu::*;
use serde_json::json;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use workload::*;

const MANIFEST_SCHEMA_VERSION: u32 = 1;

fn label(value: &str) -> GpuResourceLabel {
    GpuResourceLabel::new(value).unwrap()
}

fn native_context() -> GpuContext {
    let requirements = GpuCapabilityProfile::ComputeBaseline
        .requirements()
        .merge(&GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements())
        .unwrap();
    let descriptor = GpuContextDescriptor::new(requirements)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::ColorAttachment)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::CopySource)
        .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
        .with_allowed_backends([GpuBackendFamily::Vulkan])
        .with_label("G6-I02 bounded offscreen boids proof");
    let context = pollster::block_on(GpuContext::request(descriptor))
        .expect("native conformance environment must provide a Vulkan fallback adapter");
    assert_eq!(context.adapter_facts().backend(), GpuBackendFamily::Vulkan);
    assert_eq!(
        context.adapter_facts().fallback(),
        GpuFallbackStatus::ConfirmedFallback,
        "G6-I02 conformance must execute through the required Vulkan software path"
    );
    context
}

fn assert_prepared_materialization(graph: &GpuPreparedWorkGraph, buffer: &GpuBufferHandle) {
    assert!(
        matches!(
            buffer.descriptor().initialization(),
            GpuBufferInitialization::Prepared(_)
        ),
        "every persistent boids proof buffer must enter through Prepared initial data"
    );
    let summary = graph
        .initialization()
        .iter()
        .find(|summary| summary.resource().diagnostic_identity() == buffer.diagnostic_identity())
        .expect("Prepared boids buffer must publish initialization evidence");
    assert!(
        summary.initial().is_none(),
        "Prepared descriptor metadata alone must not claim initial coverage before canonical materialization"
    );
    let final_coverage = summary
        .final_coverage()
        .expect("canonical initial-content materialization must establish readable coverage");
    assert_eq!(
        final_coverage.buffer_values(),
        Some(
            &[GpuBufferCoverage::Dense(
                GpuBufferRange::whole(buffer).unwrap()
            )][..]
        ),
        "canonical materialization must cover the complete boids proof buffer"
    );
}

fn assert_graph_contract(graph: &GpuPreparedWorkGraph, resources: &ProofResources) {
    for feature in [
        GpuCapabilityFeature::Compute,
        GpuCapabilityFeature::RenderPipeline,
        GpuCapabilityFeature::Copy,
    ] {
        assert!(
            graph.requirements().get(feature).is_some(),
            "prepared boids graph must mechanically require {feature:?}"
        );
    }
    assert!(
        graph
            .dependencies()
            .iter()
            .flat_map(|dependency| dependency.reasons())
            .all(|reason| !matches!(reason, GpuDependencyReason::ExplicitNonData { .. })),
        "boids proof must derive ordering from resource hazards only"
    );

    for buffer in [
        &resources.state_a,
        &resources.state_b,
        &resources.cell_counts,
        &resources.cell_offsets,
        &resources.cell_cursors,
        &resources.sorted_indices,
    ] {
        assert_prepared_materialization(graph, buffer);
    }

    let publish = graph
        .nodes()
        .iter()
        .find(|prepared| prepared.node().label().as_str() == "boids tick 001 publish state")
        .expect("first fixed step must retain a publish operation");
    let render = graph
        .nodes()
        .iter()
        .find(|prepared| prepared.node().label().as_str() == "boids render frame 000")
        .expect("first fixed step must retain an instanced render operation");
    let state_range = GpuBufferRange::whole(&resources.state_a).unwrap();
    assert!(
        render.node().accesses().iter().any(|access| {
            matches!(
                access,
                GpuResourceAccess::Buffer(access)
                    if access.resource_identity() == resources.state_a.diagnostic_identity()
                        && access.kind() == GpuBufferAccessKind::VertexRead
                        && access.range() == state_range
            )
        }),
        "instanced graphics must consume the published boid state as a vertex buffer"
    );
    let dependency = graph
        .dependencies()
        .iter()
        .find(|dependency| dependency.before() == publish.id() && dependency.after() == render.id())
        .expect("published compute state must data-depend into the render consumer");
    assert!(
        dependency.reasons().iter().any(|reason| {
            matches!(
                reason,
                GpuDependencyReason::ReadAfterWrite {
                    resource,
                    region: GpuDependencyRegion::Buffer(range),
                } if *resource == resources.state_a.diagnostic_identity() && *range == state_range
            )
        }),
        "compute-to-graphics causality must contain the exact published-state RAW hazard"
    );
}

fn progress_to_readbacks(
    context: &GpuContext,
    submission: &GpuSubmission,
    ids: &[GpuReadbackId],
) -> Vec<GpuReadbackBytes> {
    let handles = ids
        .iter()
        .map(|id| {
            submission
                .readback(*id)
                .expect("accepted boids readback must remain observable")
                .clone()
        })
        .collect::<Vec<_>>();
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        context.progress();
        let mut all_ready = true;
        for handle in &handles {
            match handle.status() {
                GpuReadbackStatus::Ready(_) => {}
                GpuReadbackStatus::Failed(failure) => {
                    panic!("boids readback failed: {failure:?}")
                }
                GpuReadbackStatus::Pending => all_ready = false,
            }
        }
        match submission.status() {
            GpuSubmissionStatus::Failed(failure) => {
                panic!("boids submission failed: {failure:?}")
            }
            GpuSubmissionStatus::Completed if all_ready => break,
            GpuSubmissionStatus::Accepted | GpuSubmissionStatus::Completed => {}
        }
        assert!(
            Instant::now() < deadline,
            "boids submission/readbacks timed out"
        );
        std::thread::yield_now();
    }
    handles
        .into_iter()
        .map(|handle| match handle.status() {
            GpuReadbackStatus::Ready(bytes) => bytes,
            other => panic!("terminal boids readback must be ready, got {other:?}"),
        })
        .collect()
}

fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut digest = 0xcbf2_9ce4_8422_2325_u64;
    for byte in bytes {
        digest ^= u64::from(*byte);
        digest = digest.wrapping_mul(0x0000_0100_0000_01b3);
    }
    digest
}

fn validate_frame(bytes: &GpuReadbackBytes) -> u64 {
    assert_eq!(bytes.texture_format(), Some(GpuTextureFormat::Rgba8Unorm));
    assert_eq!(
        bytes.as_bytes().len(),
        usize::try_from(WIDTH * HEIGHT * 4).unwrap()
    );
    let mut pixel_chunks = bytes.as_bytes().chunks_exact(4);
    assert!(pixel_chunks.remainder().is_empty());
    let pixels = pixel_chunks
        .by_ref()
        .map(|pixel| <[u8; 4]>::try_from(pixel).unwrap())
        .collect::<Vec<_>>();
    assert!(pixels.iter().all(|pixel| pixel[3] == 255));
    let first = pixels[0];
    assert!(
        pixels.iter().any(|pixel| *pixel != first),
        "boids frame must contain spatially varying output"
    );
    assert!(
        pixels.iter().filter(|pixel| pixel[1] > 96).count() >= 32,
        "boids frame must contain a bounded visible population rather than only the clear target"
    );
    fnv1a64(bytes.as_bytes())
}

fn u32_words(bytes: &GpuReadbackBytes) -> Vec<u32> {
    let mut chunks = bytes.as_bytes().chunks_exact(4);
    assert!(chunks.remainder().is_empty());
    chunks
        .by_ref()
        .map(|word| u32::from_ne_bytes(word.try_into().unwrap()))
        .collect()
}

fn f32_at(bytes: &[u8], offset: usize) -> f32 {
    f32::from_ne_bytes(bytes[offset..offset + 4].try_into().unwrap())
}

fn validate_agents(bytes: &GpuReadbackBytes) {
    let stride = std::mem::size_of::<BoidAgent>();
    assert_eq!(
        bytes.as_bytes().len(),
        usize::try_from(BOID_COUNT).unwrap() * stride
    );
    for chunk in bytes.as_bytes().chunks_exact(stride) {
        let position = [f32_at(chunk, 0), f32_at(chunk, 4)];
        let velocity = [f32_at(chunk, 8), f32_at(chunk, 12)];
        let heading = [f32_at(chunk, 16), f32_at(chunk, 20)];
        assert!(
            position
                .iter()
                .all(|value| value.is_finite() && (0.0..1.0).contains(value))
        );
        assert!(velocity.iter().all(|value| value.is_finite()));
        assert!(heading.iter().all(|value| value.is_finite()));
        let speed = velocity[0].hypot(velocity[1]);
        assert!(
            speed <= MAX_SPEED + 0.001,
            "bounded boids workload must retain the declared maximum speed"
        );
        let heading_length = heading[0].hypot(heading[1]);
        assert!(
            (0.95..=1.05).contains(&heading_length),
            "visual heading must remain normalized"
        );
    }
}

fn validate_grid(
    counts_bytes: &GpuReadbackBytes,
    offsets_bytes: &GpuReadbackBytes,
    cursors_bytes: &GpuReadbackBytes,
    indices_bytes: &GpuReadbackBytes,
) {
    let counts = u32_words(counts_bytes);
    let offsets = u32_words(offsets_bytes);
    let cursors = u32_words(cursors_bytes);
    let mut indices = u32_words(indices_bytes);
    assert_eq!(counts.len(), usize::try_from(GRID_CELL_COUNT).unwrap());
    assert_eq!(offsets.len(), counts.len());
    assert_eq!(cursors.len(), counts.len());
    assert_eq!(indices.len(), usize::try_from(BOID_COUNT).unwrap());
    assert_eq!(counts.iter().copied().sum::<u32>(), BOID_COUNT);
    assert_eq!(offsets[0], 0);
    for cell in 1..counts.len() {
        assert_eq!(offsets[cell], offsets[cell - 1] + counts[cell - 1]);
    }
    assert_eq!(offsets.last().unwrap() + counts.last().unwrap(), BOID_COUNT);
    assert_eq!(
        cursors, counts,
        "scatter cursors must finish at each cell count"
    );
    assert!(indices.iter().all(|index| *index < BOID_COUNT));
    indices.sort_unstable();
    assert_eq!(indices, (0..BOID_COUNT).collect::<Vec<_>>());
}

fn artifact_root() -> PathBuf {
    std::env::var_os("RUNEN_GPU_PROOF_ARTIFACT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/runengpu-proof-artifacts"))
        .join("boids")
}

fn write_png(path: &Path, bytes: &GpuReadbackBytes) {
    image::save_buffer_with_format(
        path,
        bytes.as_bytes(),
        WIDTH,
        HEIGHT,
        image::ColorType::Rgba8,
        image::ImageFormat::Png,
    )
    .unwrap();
}

#[test]
#[ignore = "requires a real Vulkan fallback adapter; executed by RunenGPU Conformance CI"]
fn native_boids_proves_bounded_compute_to_instanced_render_sequence() {
    let context = native_context();
    let sources = admitted_sources();
    let work = offscreen_work(&sources);
    let graph = GpuPreparedWorkGraph::prepare(label(&work.graph_label), [work.fragment]).unwrap();
    assert_graph_contract(&graph, &work.resources);

    let prepared = pollster::block_on(context.prepare_submission(graph)).unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    let mut readback_ids = work.frame_readbacks.clone();
    readback_ids.extend([
        work.state_readback,
        work.counts_readback,
        work.offsets_readback,
        work.cursors_readback,
        work.indices_readback,
    ]);
    let readbacks = progress_to_readbacks(&context, &submission, &readback_ids);
    let frame_count = usize::try_from(FRAME_COUNT).unwrap();
    let (frames, state) = readbacks.split_at(frame_count);
    assert_eq!(state.len(), 5);

    let frame_hashes = frames.iter().map(validate_frame).collect::<Vec<_>>();
    assert_ne!(
        frame_hashes.first(),
        frame_hashes.last(),
        "fixed-step boids sequence must visibly evolve"
    );
    validate_agents(&state[0]);
    validate_grid(&state[1], &state[2], &state[3], &state[4]);

    let root = artifact_root();
    std::fs::create_dir_all(&root).unwrap();
    let mut frame_records = Vec::with_capacity(frame_count);
    for (frame_index, (bytes, hash)) in frames.iter().zip(frame_hashes.iter()).enumerate() {
        let file_name = format!("frame_{frame_index:03}.png");
        let path = root.join(&file_name);
        write_png(&path, bytes);
        assert!(path.metadata().unwrap().len() > 0);
        frame_records.push(json!({
            "logical_frame": frame_index,
            "png": file_name,
            "fnv1a64": format!("{hash:016x}"),
        }));
    }

    let manifest = json!({
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "workload": "runengpu-bounded-boids",
        "boid_count": BOID_COUNT,
        "grid": [GRID_CELLS_X, GRID_CELLS_Y],
        "fixed_delta_seconds": FIXED_DELTA_SECONDS,
        "logical_frames": FRAME_COUNT,
        "offscreen_extent": [WIDTH, HEIGHT],
        "workgroup_size": [WORKGROUP, 1, 1],
        "simulation_stages": [
            "clear-counts",
            "count-cells",
            "scan-counts",
            "reset-cursors",
            "scatter-sorted-indices",
            "simulate-neighbors",
            "publish-state"
        ],
        "initialization": "CPU deterministic seed plus Prepared canonical materialization",
        "render": "direct six-vertex local quad geometry with one instance per published BoidAgent",
        "backend": format!("{:?}", context.adapter_facts().backend()),
        "fallback": format!("{:?}", context.adapter_facts().fallback()),
        "programs": {
            "compute": {
                "source_key": sources.compute.identity().key().as_str(),
                "source_revision": sources.compute.identity().revision().get(),
                "canonical_wgsl_digest": sources.compute.digest().to_string(),
            },
            "render": {
                "source_key": sources.render.identity().key().as_str(),
                "source_revision": sources.render.identity().revision().get(),
                "canonical_wgsl_digest": sources.render.digest().to_string(),
            }
        },
        "frames": frame_records,
    });
    let manifest_path = root.join("manifest.json");
    std::fs::write(
        &manifest_path,
        serde_json::to_vec_pretty(&manifest).unwrap(),
    )
    .unwrap();
    assert!(manifest_path.metadata().unwrap().len() > 0);

    let stats = context.execution_stats();
    assert_eq!(stats.prepared_submissions(), 0);
    assert_eq!(stats.in_flight_submissions(), 0);
    assert_eq!(stats.upload_bytes_in_flight(), 0);
    assert_eq!(stats.readback_bytes_in_flight(), 0);
    assert_eq!(stats.pending_readbacks(), 0);
}
