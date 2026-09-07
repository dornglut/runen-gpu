use super::*;
use serde_json::json;
use std::path::{Path, PathBuf};

const MANIFEST_SCHEMA_VERSION: u32 = 1;
const RENDER_SOURCE_REVISION: u64 = 1;
const RENDER_WGSL: &str = include_str!("render.wgsl");
const VISUAL_FRAME_COUNT: usize = STEP_COUNT as usize + 1;

fn admitted_render_source() -> GpuAdmittedProgramSource {
    let [render] = admit_static_wgsl_sources([(
        "proof.game-of-life.render",
        RENDER_SOURCE_REVISION,
        RENDER_WGSL,
    )])
    .unwrap();
    render
}

fn render_pipeline(source: &GpuAdmittedProgramSource) -> GpuRenderPipelineDescriptor {
    GpuRenderPipelineDescriptor::ordinary_color(
        source.clone(),
        "vs_main",
        "fs_main",
        GpuTextureFormat::Rgba8Unorm,
    )
    .unwrap()
}

fn offscreen_target(resources: &mut GpuResourceScope) -> (GpuTextureHandle, GpuTextureViewHandle) {
    let texture = resources
        .texture(
            GpuTextureDescriptor::ordinary_owned_2d(
                "G5-C02 Game of Life visual target",
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                WIDTH,
                HEIGHT,
                GpuTextureFormat::Rgba8Unorm,
                [
                    GpuTextureUsage::ColorAttachment,
                    GpuTextureUsage::CopySource,
                ],
                GpuTextureInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();
    let view = resources
        .texture_view(
            GpuTextureViewDescriptor::ordinary_full_owned(
                "G5-C02 Game of Life visual target view",
                &texture,
            )
            .unwrap(),
        )
        .unwrap();
    (texture, view)
}

fn render_operation(
    pipeline: &GpuRenderPipelineDescriptor,
    state: &GpuBufferHandle,
    view: &GpuTextureViewHandle,
) -> GpuRenderOperation {
    let bindings = pipeline
        .runtime_bindings([GpuRuntimeBindingValue::whole_buffer(0, 0, state)])
        .unwrap();
    GpuRenderOperation::ordinary_color_full_target_direct(
        pipeline,
        bindings,
        view,
        GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 1.0).unwrap()),
        GpuAttachmentStore::Store,
        GpuDrawRange::new(0, 3).unwrap(),
        GpuDrawRange::new(0, 1).unwrap(),
    )
    .unwrap()
}

fn record_visual_frame(
    builder: &mut GpuWorkFragmentBuilder,
    pipeline: &GpuRenderPipelineDescriptor,
    state: &GpuBufferHandle,
    texture: &GpuTextureHandle,
    view: &GpuTextureViewHandle,
    logical_step: u32,
    readbacks: &mut Vec<GpuReadbackId>,
) -> Result<(), GpuWorkAuthoringError> {
    builder.operation(
        format!("game of life render frame {logical_step:03}"),
        render_operation(pipeline, state, view),
    )?;
    let region = GpuTextureCopyRegion::whole_base_mip(texture).unwrap();
    let readback = GpuReadbackOperation::ordinary(region.into()).unwrap();
    let readback_id = readback.id();
    builder.operation(
        format!("game of life readback frame {logical_step:03}"),
        readback,
    )?;
    readbacks.push(readback_id);
    Ok(())
}

fn author_visual_work(
    step_source: &GpuAdmittedProgramSource,
    render_source: &GpuAdmittedProgramSource,
    source_state: &[u32],
) -> (GpuWorkFragment, Vec<GpuReadbackId>) {
    assert_eq!(source_state.len(), CELL_COUNT);
    let prepared_source = PreparedGpuData::<TransferData>::ordinary_pod_transfer(
        "G5-C02 Game of Life visual source state",
        source_state,
    )
    .unwrap();

    let mut resources = GpuResourceScope::new();
    let state_a = state_buffer(&mut resources, "game of life visual state a");
    let state_b = state_buffer(&mut resources, "game of life visual state b");
    let (texture, view) = offscreen_target(&mut resources);
    let step = compute_pipeline(step_source);
    let render = render_pipeline(render_source);
    let mut readbacks = Vec::with_capacity(VISUAL_FRAME_COUNT);

    let fragment = GpuWorkFragment::build("G5-C02 Game of Life visual sequence", |work| {
        work.operation(
            "game of life visual upload source state a",
            GpuUploadOperation::whole_buffer(&state_a, prepared_source.clone()).unwrap(),
        )?;
        work.operation(
            "game of life visual upload source state b",
            GpuUploadOperation::whole_buffer(&state_b, prepared_source).unwrap(),
        )?;

        record_visual_frame(work, &render, &state_a, &texture, &view, 0, &mut readbacks)?;

        for step_index in 0..STEP_COUNT {
            let (input, output) = if step_index % 2 == 0 {
                (&state_a, &state_b)
            } else {
                (&state_b, &state_a)
            };
            work.operation(
                format!("game of life visual step {:02}", step_index + 1),
                step_operation(&step, input, output),
            )?;
            record_visual_frame(
                work,
                &render,
                output,
                &texture,
                &view,
                step_index + 1,
                &mut readbacks,
            )?;
        }
        Ok(())
    })
    .unwrap();

    assert_eq!(readbacks.len(), VISUAL_FRAME_COUNT);
    (fragment, readbacks)
}

fn native_visual_context() -> GpuContext {
    let requirements = GpuCapabilityProfile::ComputeBaseline
        .requirements()
        .merge(&GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements())
        .unwrap();
    let descriptor = GpuContextDescriptor::new(requirements)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::ColorAttachment)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::CopySource)
        .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
        .with_allowed_backends([GpuBackendFamily::Vulkan])
        .with_label("G5-C02 Game of Life visual sequence proof");
    let context = pollster::block_on(GpuContext::request(descriptor))
        .expect("native conformance environment must provide a Vulkan fallback adapter");
    assert_eq!(context.adapter_facts().backend(), GpuBackendFamily::Vulkan);
    assert_eq!(
        context.adapter_facts().fallback(),
        GpuFallbackStatus::ConfirmedFallback
    );
    context
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
                .expect("Game-of-Life visual readback must remain observable")
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
                    panic!("Game-of-Life visual readback failed: {failure:?}")
                }
                GpuReadbackStatus::Pending => all_ready = false,
            }
        }
        match submission.status() {
            GpuSubmissionStatus::Failed(failure) => {
                panic!("Game-of-Life visual submission failed: {failure:?}")
            }
            GpuSubmissionStatus::Completed if all_ready => break,
            GpuSubmissionStatus::Accepted | GpuSubmissionStatus::Completed => {}
        }
        assert!(
            Instant::now() < deadline,
            "Game-of-Life visual proof timed out"
        );
        std::thread::yield_now();
    }

    handles
        .into_iter()
        .map(|handle| match handle.status() {
            GpuReadbackStatus::Ready(bytes) => bytes,
            other => panic!("terminal Game-of-Life visual readback must be ready, got {other:?}"),
        })
        .collect()
}

fn cpu_frames(source_state: &[u32]) -> Vec<Vec<u32>> {
    let mut current = source_state.to_vec();
    let mut frames = Vec::with_capacity(VISUAL_FRAME_COUNT);
    frames.push(current.clone());
    for _ in 0..STEP_COUNT {
        current = cpu_step(&current);
        frames.push(current.clone());
    }
    frames
}

fn expected_rgba(cells: &[u32]) -> Vec<u8> {
    assert_eq!(cells.len(), CELL_COUNT);
    let mut rgba = Vec::with_capacity(CELL_COUNT * 4);
    for cell in cells {
        match cell {
            0 => rgba.extend_from_slice(&[0, 0, 0, 255]),
            1 => rgba.extend_from_slice(&[255, 255, 255, 255]),
            other => panic!("unexpected Game-of-Life cell value {other}"),
        }
    }
    rgba
}

fn fnv1a64_bytes(bytes: &[u8]) -> u64 {
    const OFFSET: u64 = 0xCBF2_9CE4_8422_2325;
    const PRIME: u64 = 0x0000_0100_0000_01B3;

    let mut hash = OFFSET;
    for byte in bytes {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(PRIME);
    }
    hash
}

fn artifact_root() -> PathBuf {
    std::env::var_os("RUNEN_GPU_PROOF_ARTIFACT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/runengpu-proof-artifacts"))
        .join("game-of-life")
}

fn write_png(path: &Path, rgba: &[u8]) {
    image::save_buffer_with_format(
        path,
        rgba,
        WIDTH,
        HEIGHT,
        image::ColorType::Rgba8,
        image::ImageFormat::Png,
    )
    .unwrap();
}

#[test]
#[ignore = "requires a real Vulkan fallback adapter; executed by RunenGPU Conformance CI"]
fn native_game_of_life_retains_exact_17_frame_visual_sequence() {
    let source_state = cpu_seed();
    let expected_frames = cpu_frames(&source_state);
    assert_eq!(expected_frames.len(), VISUAL_FRAME_COUNT);
    assert_canonical_oracle(expected_frames.last().unwrap());

    let context = native_visual_context();
    let step_source = admitted_step_source();
    let render_source = admitted_render_source();
    let (fragment, readback_ids) = author_visual_work(&step_source, &render_source, &source_state);
    let graph = GpuPreparedWorkGraph::prepare(
        label("G5-C02 Game of Life visual prepared graph"),
        [fragment],
    )
    .unwrap();

    assert_eq!(graph.topological_order().len(), graph.nodes().len());
    assert!(
        graph
            .dependencies()
            .iter()
            .flat_map(|dependency| dependency.reasons())
            .all(|reason| !matches!(reason, GpuDependencyReason::ExplicitNonData { .. })),
        "Game-of-Life visual sequence must be ordered by typed resource hazards"
    );
    for feature in [
        GpuCapabilityFeature::Compute,
        GpuCapabilityFeature::RenderPipeline,
        GpuCapabilityFeature::Copy,
    ] {
        assert!(graph.requirements().get(feature).is_some());
    }

    let prepared = pollster::block_on(context.prepare_submission(graph)).unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    let frames = progress_to_readbacks(&context, &submission, &readback_ids);
    assert_eq!(frames.len(), VISUAL_FRAME_COUNT);

    let root = artifact_root();
    std::fs::create_dir_all(&root).unwrap();
    let mut frame_records = Vec::with_capacity(VISUAL_FRAME_COUNT);

    for (logical_step, (bytes, expected_cells)) in
        frames.iter().zip(expected_frames.iter()).enumerate()
    {
        assert_eq!(bytes.texture_format(), Some(GpuTextureFormat::Rgba8Unorm));
        let expected = expected_rgba(expected_cells);
        assert_eq!(
            bytes.as_bytes(),
            expected.as_slice(),
            "Game-of-Life rendered frame {logical_step} must equal the exact CPU visual oracle"
        );

        let file_name = format!("frame_{logical_step:03}.png");
        let path = root.join(&file_name);
        write_png(&path, bytes.as_bytes());
        assert!(path.metadata().unwrap().len() > 0);

        frame_records.push(json!({
            "logical_step": logical_step,
            "png": file_name,
            "live_cells": expected_cells.iter().copied().sum::<u32>(),
            "raw_rgba_fnv1a64": format!("{:016x}", fnv1a64_bytes(bytes.as_bytes())),
        }));
    }

    let manifest = json!({
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "workload": "runengpu-game-of-life",
        "dimensions": [WIDTH, HEIGHT],
        "rule": "B3/S23",
        "boundary": "toroidal-wrap",
        "seed": "0xC0FF_EE11",
        "steps": STEP_COUNT,
        "logical_frames": VISUAL_FRAME_COUNT,
        "backend": format!("{:?}", context.adapter_facts().backend()),
        "fallback": format!("{:?}", context.adapter_facts().fallback()),
        "programs": {
            "compute": {
                "source_key": step_source.identity().key().as_str(),
                "source_revision": step_source.identity().revision().get(),
                "canonical_wgsl_digest": step_source.digest().to_string(),
            },
            "render": {
                "source_key": render_source.identity().key().as_str(),
                "source_revision": render_source.identity().revision().get(),
                "canonical_wgsl_digest": render_source.digest().to_string(),
            },
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
