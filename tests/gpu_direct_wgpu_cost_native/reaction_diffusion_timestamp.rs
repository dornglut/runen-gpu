use super::*;
use crate::common::summarize_u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TimedPassKind {
    Compute,
    Render,
}

impl TimedPassKind {
    const fn key(self) -> &'static str {
        match self {
            Self::Compute => "compute",
            Self::Render => "render",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TimedPass {
    kind: TimedPassKind,
    frame: u32,
    iteration: Option<u32>,
    label: String,
}

impl TimedPass {
    fn json(&self) -> Value {
        json!({
            "kind": self.kind.key(),
            "frame": self.frame,
            "iteration": self.iteration,
            "label": self.label,
        })
    }
}

fn timed_passes(envelope: retained::Envelope) -> Vec<TimedPass> {
    let count = envelope.frames * (envelope.iterations_per_frame + 1);
    let mut passes = Vec::with_capacity(usize::try_from(count).unwrap());
    for frame in 0..envelope.frames {
        for iteration in 0..envelope.iterations_per_frame {
            passes.push(TimedPass {
                kind: TimedPassKind::Compute,
                frame,
                iteration: Some(iteration),
                label: format!(
                    "{} frame {frame:03} iteration {iteration:03}",
                    envelope.name
                ),
            });
        }
        passes.push(TimedPass {
            kind: TimedPassKind::Render,
            frame,
            iteration: None,
            label: format!("{} render frame {frame:03}", envelope.name),
        });
    }
    assert_eq!(passes.len(), usize::try_from(count).unwrap());
    passes
}

fn query_count(passes: &[TimedPass]) -> u32 {
    u32::try_from(passes.len() * 2).unwrap()
}

fn timestamp_bytes(query_count: u32) -> u64 {
    u64::from(query_count) * 8
}

fn timestamp_context() -> GpuContext {
    let mut requirements = GpuCapabilityProfile::ComputeBaseline
        .requirements()
        .merge(&GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements())
        .unwrap();
    requirements
        .insert(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::TimestampQuery,
        ))
        .unwrap();
    let descriptor = GpuContextDescriptor::new(requirements)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::ColorAttachment)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::CopySource)
        .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
        .with_allowed_backends([GpuBackendFamily::Vulkan])
        .with_label("G6-P01 reaction-diffusion RunenGPU timestamp evidence");
    let context = pollster::block_on(GpuContext::request(descriptor)).unwrap();
    assert_eq!(context.adapter_facts().backend(), GpuBackendFamily::Vulkan);
    assert_eq!(
        context.adapter_facts().fallback(),
        GpuFallbackStatus::ConfirmedFallback
    );
    context
}

fn query_common(name: &str) -> GpuResourceCommon {
    let resource_label = label(name);
    GpuResourceCommon::owned(
        resource_label.clone(),
        GpuResourceLifetime::Transient,
        GpuMemoryIntent::Device,
        GpuReconstruction::SourceBacked,
        GpuResourceProvenance::new(resource_label, None, None),
    )
    .unwrap()
}

struct TimestampFragment {
    fragment: GpuWorkFragment,
    frame_readbacks: Vec<GpuReadbackId>,
    timestamp_readback: GpuReadbackId,
}

fn timestamp_fragment(
    sources: &retained::ProgramSources,
    envelope: retained::Envelope,
) -> TimestampFragment {
    let (_, base, frame_readbacks) = retained::offscreen_work(sources, envelope);
    assert!(base.inputs().is_empty());
    assert!(base.imports().is_empty());
    assert!(base.outputs().is_empty());
    assert!(base.explicit_orders().is_empty());

    let passes = timed_passes(envelope);
    let query_count = query_count(&passes);
    let mut resources = GpuResourceScope::new();
    let query_set = resources
        .query_set(
            GpuQuerySetDescriptor::new(
                query_common(&format!(
                    "reaction diffusion {} timestamp query set",
                    envelope.name
                )),
                GpuQueryKind::Timestamp,
                query_count,
            )
            .unwrap(),
        )
        .unwrap();
    let resolve = resources
        .buffer(
            GpuBufferDescriptor::ordinary_owned(
                format!("reaction diffusion {} timestamp resolve", envelope.name),
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                timestamp_bytes(query_count),
                [GpuBufferUsage::QueryResolve, GpuBufferUsage::CopySource],
                GpuBufferInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();
    let resolve_operation = GpuQueryResolveOperation::new(
        &query_set,
        GpuQueryRange::new(&query_set, 0, query_count).unwrap(),
        &resolve,
        0,
    )
    .unwrap();
    let timestamp_readback_operation =
        GpuReadbackOperation::ordinary(GpuBufferRegion::whole(&resolve).unwrap().into()).unwrap();
    let timestamp_readback = timestamp_readback_operation.id();

    let mut timed_index = 0_usize;
    let fragment = GpuWorkFragment::build(
        format!("{} reaction diffusion timestamp comparison", envelope.name),
        |work| {
            for node in base.nodes() {
                match node.operation() {
                    GpuWorkOperation::Compute(operation) => {
                        let expected = &passes[timed_index];
                        assert_eq!(expected.kind, TimedPassKind::Compute);
                        assert_eq!(node.label().as_str(), expected.label);
                        let beginning = u32::try_from(timed_index * 2).unwrap();
                        let writes =
                            GpuTimestampWrites::new(&query_set, Some(beginning), Some(beginning + 1))
                                .unwrap();
                        work.operation(
                            node.label().as_str(),
                            operation.clone().with_timestamp_writes(writes),
                        )?;
                        timed_index += 1;
                    }
                    GpuWorkOperation::Render(operation) => {
                        let expected = &passes[timed_index];
                        assert_eq!(expected.kind, TimedPassKind::Render);
                        assert_eq!(node.label().as_str(), expected.label);
                        let beginning = u32::try_from(timed_index * 2).unwrap();
                        let writes =
                            GpuTimestampWrites::new(&query_set, Some(beginning), Some(beginning + 1))
                                .unwrap();
                        let timestamped = GpuRenderOperation::new(
                            operation.color_attachments().iter().cloned(),
                            operation.depth_stencil_attachment().cloned(),
                            operation.draws().iter().cloned(),
                            Some(writes),
                        )
                        .unwrap();
                        work.operation(node.label().as_str(), timestamped)?;
                        timed_index += 1;
                    }
                    GpuWorkOperation::Upload(_) | GpuWorkOperation::Readback(_) => {
                        work.operation(node.label().as_str(), node.operation().clone())?;
                    }
                    other => panic!(
                        "retained reaction-diffusion timestamp clone found unexpected {:?} operation",
                        other.kind()
                    ),
                }
            }
            work.operation(
                "reaction diffusion timestamp resolve",
                resolve_operation.clone(),
            )?;
            work.operation(
                "reaction diffusion timestamp readback",
                timestamp_readback_operation,
            )?;
            Ok(())
        },
    )
    .unwrap();
    assert_eq!(timed_index, passes.len());
    assert_eq!(
        fragment.nodes().len(),
        base.nodes().len() + 2,
        "timestamp instrumentation must add only resolve + timestamp readback nodes"
    );

    TimestampFragment {
        fragment,
        frame_readbacks,
        timestamp_readback,
    }
}

fn timestamp_deltas(bytes: &[u8], query_count: u32) -> Vec<u64> {
    let timestamps = bytes.chunks_exact(8);
    assert!(timestamps.remainder().is_empty());
    let timestamps = timestamps
        .map(|timestamp| u64::from_ne_bytes(timestamp.try_into().unwrap()))
        .collect::<Vec<_>>();
    assert_eq!(timestamps.len(), usize::try_from(query_count).unwrap());
    let pairs = timestamps.chunks_exact(2);
    assert!(pairs.remainder().is_empty());
    pairs
        .map(|pair| {
            let beginning = pair[0];
            let end = pair[1];
            assert!(end >= beginning, "timestamp end must not precede beginning");
            end - beginning
        })
        .collect()
}

fn runengpu_timestamp_sample(
    context: &GpuContext,
    sources: &retained::ProgramSources,
    envelope: retained::Envelope,
) -> Vec<u64> {
    let TimestampFragment {
        fragment,
        mut frame_readbacks,
        timestamp_readback,
    } = timestamp_fragment(sources, envelope);
    let passes = timed_passes(envelope);
    let query_count = query_count(&passes);
    let graph = GpuPreparedWorkGraph::prepare(
        label(&format!(
            "G6-P01 reaction diffusion {} timestamp graph",
            envelope.name
        )),
        [fragment],
    )
    .unwrap();
    let prepared = pollster::block_on(context.prepare_submission(graph)).unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    frame_readbacks.push(timestamp_readback);
    let mut readbacks = progress_to_readbacks(context, &submission, &frame_readbacks);
    let timestamp = readbacks
        .pop()
        .expect("timestamp-instrumented reaction workload must return timestamp bytes");
    validate_runengpu_frames(&readbacks, envelope);
    let deltas = timestamp_deltas(timestamp.as_bytes(), query_count);
    assert_eq!(deltas.len(), passes.len());
    deltas
}

fn direct_timestamp_compute_dispatch(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    binding: &wgpu::BindGroup,
    envelope: retained::Envelope,
    query_set: &wgpu::QuerySet,
    pass_index: usize,
) {
    let beginning = u32::try_from(pass_index * 2).unwrap();
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("G6-P01 reaction-diffusion timestamp compute"),
        timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: Some(beginning),
            end_of_pass_write_index: Some(beginning + 1),
        }),
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, binding, &[]);
    pass.dispatch_workgroups(
        envelope.width.div_ceil(retained::WORKGROUP),
        envelope.height.div_ceil(retained::WORKGROUP),
        1,
    );
}

fn direct_timestamp_render(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::RenderPipeline,
    binding: &wgpu::BindGroup,
    view: &wgpu::TextureView,
    query_set: &wgpu::QuerySet,
    pass_index: usize,
) {
    let color_attachments = [Some(wgpu::RenderPassColorAttachment {
        view,
        depth_slice: None,
        resolve_target: None,
        ops: wgpu::Operations {
            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
            store: wgpu::StoreOp::Store,
        },
    })];
    let beginning = u32::try_from(pass_index * 2).unwrap();
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("G6-P01 reaction-diffusion timestamp render"),
        color_attachments: &color_attachments,
        depth_stencil_attachment: None,
        timestamp_writes: Some(wgpu::RenderPassTimestampWrites {
            query_set,
            beginning_of_pass_write_index: Some(beginning),
            end_of_pass_write_index: Some(beginning + 1),
        }),
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, binding, &[]);
    pass.draw(0..3, 0..1);
}

fn direct_timestamp_sample(
    context: &DirectWgpuContext,
    pipelines: &DirectPipelines,
    envelope: retained::Envelope,
) -> Vec<u64> {
    let passes = timed_passes(envelope);
    let query_count = query_count(&passes);
    let timestamp_bytes = timestamp_bytes(query_count);
    let resources = direct_resources(context, pipelines, envelope);
    let query_set = context.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("G6-P01 reaction-diffusion direct timestamp query set"),
        ty: wgpu::QueryType::Timestamp,
        count: query_count,
    });
    let resolve = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("G6-P01 reaction-diffusion direct timestamp resolve"),
        size: timestamp_bytes,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let timestamp_readback = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("G6-P01 reaction-diffusion direct timestamp readback"),
        size: timestamp_bytes,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("G6-P01 reaction-diffusion direct timestamp encoder"),
        });
    encoder.copy_buffer_to_buffer(
        &resources.seed_upload_a,
        0,
        &resources.state_a,
        0,
        state_bytes(envelope),
    );
    encoder.copy_buffer_to_buffer(
        &resources.seed_upload_b,
        0,
        &resources.state_b,
        0,
        state_bytes(envelope),
    );
    encoder.copy_buffer_to_buffer(
        &resources.params_upload,
        0,
        &resources.params,
        0,
        params_bytes(),
    );

    let physical_row = padded_bytes_per_row(logical_row_bytes(envelope));
    let mut current_is_a = true;
    let mut pass_index = 0_usize;
    for frame in 0..envelope.frames {
        for _ in 0..envelope.iterations_per_frame {
            let binding = if current_is_a {
                &resources.compute_ab
            } else {
                &resources.compute_ba
            };
            direct_timestamp_compute_dispatch(
                &mut encoder,
                &pipelines.compute,
                binding,
                envelope,
                &query_set,
                pass_index,
            );
            pass_index += 1;
            current_is_a = !current_is_a;
        }
        let render_binding = if current_is_a {
            &resources.render_a
        } else {
            &resources.render_b
        };
        direct_timestamp_render(
            &mut encoder,
            &pipelines.render,
            render_binding,
            &resources.target_view,
            &query_set,
            pass_index,
        );
        pass_index += 1;
        encoder.copy_texture_to_buffer(
            wgpu::TexelCopyTextureInfo {
                texture: &resources.target,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::TexelCopyBufferInfo {
                buffer: &resources.readbacks[usize::try_from(frame).unwrap()],
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(physical_row),
                    rows_per_image: None,
                },
            },
            wgpu::Extent3d {
                width: envelope.width,
                height: envelope.height,
                depth_or_array_layers: 1,
            },
        );
    }
    assert_eq!(pass_index, passes.len());
    encoder.resolve_query_set(&query_set, 0..query_count, &resolve, 0);
    encoder.copy_buffer_to_buffer(&resolve, 0, &timestamp_readback, 0, timestamp_bytes);
    let command_buffer = encoder.finish();

    let frame_count = resources.readbacks.len();
    let mut readback_refs = resources.readbacks.iter().collect::<Vec<_>>();
    readback_refs.push(&timestamp_readback);
    let submitted = submit_and_map(context, command_buffer, &readback_refs);
    assert_eq!(submitted.mapped.len(), frame_count + 1);
    let frames = submitted.mapped[..frame_count]
        .iter()
        .map(|mapped| {
            tightly_pack_texture_rows(
                mapped,
                envelope.width,
                envelope.height,
                BYTES_PER_PIXEL,
                physical_row,
            )
        })
        .collect::<Vec<_>>();
    validate_direct_frames(&frames, envelope);
    let deltas = timestamp_deltas(&submitted.mapped[frame_count], query_count);
    assert_eq!(deltas.len(), passes.len());
    deltas
}

fn aggregate_ticks(samples: &[Vec<u64>], passes: &[TimedPass], period_ns: Option<f64>) -> Value {
    assert_eq!(samples.len(), MEASURED_SAMPLES);
    assert!(samples.iter().all(|sample| sample.len() == passes.len()));

    let total_ticks = samples
        .iter()
        .map(|sample| sample.iter().sum::<u64>())
        .collect::<Vec<_>>();
    let compute_ticks = samples
        .iter()
        .map(|sample| {
            sample
                .iter()
                .zip(passes)
                .filter(|(_, pass)| pass.kind == TimedPassKind::Compute)
                .map(|(ticks, _)| *ticks)
                .sum::<u64>()
        })
        .collect::<Vec<_>>();
    let render_ticks = samples
        .iter()
        .map(|sample| {
            sample
                .iter()
                .zip(passes)
                .filter(|(_, pass)| pass.kind == TimedPassKind::Render)
                .map(|(ticks, _)| *ticks)
                .sum::<u64>()
        })
        .collect::<Vec<_>>();
    let delta_ns_samples = period_ns.map(|period| {
        samples
            .iter()
            .map(|sample| {
                sample
                    .iter()
                    .map(|ticks| *ticks as f64 * period)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    });

    json!({
        "raw_delta_ticks_samples": samples,
        "total_pass_ticks": {
            "samples": total_ticks,
            "summary": summarize_u64(&total_ticks),
        },
        "compute_pass_ticks": {
            "samples": compute_ticks,
            "summary": summarize_u64(&compute_ticks),
        },
        "render_pass_ticks": {
            "samples": render_ticks,
            "summary": summarize_u64(&render_ticks),
        },
        "delta_ns_samples": delta_ns_samples,
    })
}

fn measure_envelope(
    runengpu_context: &GpuContext,
    direct_context: &DirectWgpuContext,
    sources: &retained::ProgramSources,
    direct_pipelines: &DirectPipelines,
    envelope: retained::Envelope,
    runengpu_period_ns: f64,
    direct_period_ns: f64,
) -> Value {
    let passes = timed_passes(envelope);
    for _ in 0..WARMUP_SAMPLES {
        let _ = runengpu_timestamp_sample(runengpu_context, sources, envelope);
        let _ = direct_timestamp_sample(direct_context, direct_pipelines, envelope);
    }

    let mut runengpu = Vec::with_capacity(MEASURED_SAMPLES);
    let mut direct = Vec::with_capacity(MEASURED_SAMPLES);
    for _ in 0..MEASURED_SAMPLES {
        runengpu.push(runengpu_timestamp_sample(
            runengpu_context,
            sources,
            envelope,
        ));
        direct.push(direct_timestamp_sample(
            direct_context,
            direct_pipelines,
            envelope,
        ));
    }

    json!({
        "name": envelope.name,
        "width": envelope.width,
        "height": envelope.height,
        "frames": envelope.frames,
        "iterations_per_frame": envelope.iterations_per_frame,
        "compute_passes": envelope.frames * envelope.iterations_per_frame,
        "render_passes": envelope.frames,
        "timed_passes": passes.len(),
        "query_count": query_count(&passes),
        "pass_sequence": passes.iter().map(TimedPass::json).collect::<Vec<_>>(),
        "runengpu": aggregate_ticks(&runengpu, &passes, Some(runengpu_period_ns)),
        "direct_wgpu": aggregate_ticks(&direct, &passes, Some(direct_period_ns)),
    })
}

fn expected_adapter_identity(wall_evidence: &Value) -> (u64, u64) {
    let direct = &wall_evidence["adapter_equivalence"]["direct_wgpu"];
    (
        direct["vendor"].as_u64().unwrap(),
        direct["device"].as_u64().unwrap(),
    )
}

pub(crate) fn evidence(wall_evidence: &Value) -> Value {
    assert_eq!(
        wall_evidence["timestamp_evidence"]["supported_by_direct_adapter"], true,
        "accepted G6-P01 Lavapipe adapter reports timestamp-query support"
    );
    let (expected_vendor, expected_device) = expected_adapter_identity(wall_evidence);

    let runengpu_context = timestamp_context();
    let direct_context = DirectWgpuContext::request_timestamp(
        "G6-P01 reaction-diffusion direct WGPU timestamp evidence",
    );
    assert_equivalent_adapter_selection(&runengpu_context, &direct_context);
    assert_eq!(
        runengpu_context.adapter_facts().vendor().map(u64::from),
        Some(expected_vendor)
    );
    assert_eq!(
        runengpu_context.adapter_facts().device().map(u64::from),
        Some(expected_device)
    );
    assert_eq!(
        u64::from(direct_context.adapter_info.vendor),
        expected_vendor
    );
    assert_eq!(
        u64::from(direct_context.adapter_info.device),
        expected_device
    );

    let sources = retained::admitted_sources();
    let direct_pipelines = direct_pipelines(&direct_context);
    let runengpu_period_ns = f64::from(
        runengpu_context
            .timestamp_period_ns()
            .expect("timestamp-query-admitted RunenGPU context must expose timestamp period"),
    );
    let direct_period_ns = f64::from(direct_context.queue.get_timestamp_period());
    assert!(runengpu_period_ns.is_finite() && runengpu_period_ns > 0.0);
    assert!(direct_period_ns.is_finite() && direct_period_ns > 0.0);
    let first = measure_envelope(
        &runengpu_context,
        &direct_context,
        &sources,
        &direct_pipelines,
        retained::ENVELOPES[0],
        runengpu_period_ns,
        direct_period_ns,
    );
    let second = measure_envelope(
        &runengpu_context,
        &direct_context,
        &sources,
        &direct_pipelines,
        retained::ENVELOPES[1],
        runengpu_period_ns,
        direct_period_ns,
    );
    let total_timed_passes = retained::ENVELOPES
        .iter()
        .map(|envelope| envelope.frames * (envelope.iterations_per_frame + 1))
        .sum::<u32>();

    json!({
        "status": "measured",
        "separate_from_wall_clock_samples": true,
        "same_canonical_shader_sources": true,
        "pass_boundaries": "beginning/end of every retained compute and render pass; upload, texture-copy/readback, resolve, and host work are intentionally outside GPU pass timestamps",
        "warmup_samples": WARMUP_SAMPLES,
        "measured_samples": MEASURED_SAMPLES,
        "total_timed_passes_across_envelopes": total_timed_passes,
        "adapter_equivalence": {
            "criteria": "timestamp-enabled contexts resolve the same Vulkan forced-fallback vendor/device identity as wall-clock evidence",
            "runengpu": runengpu_adapter_facts_json(&runengpu_context),
            "direct_wgpu": {
                "facts": direct_context.facts_json(),
                "vendor": direct_context.adapter_info.vendor,
                "device": direct_context.adapter_info.device,
            },
        },
        "runengpu_timestamp_period_ns": runengpu_period_ns,
        "runengpu_timestamp_period_status": "reported by GpuContext::timestamp_period_ns()",
        "direct_wgpu_timestamp_period_ns": direct_period_ns,
        "envelopes": [first, second],
        "correctness": {
            "runengpu": "retained per-frame format/size/alpha/spatial-variation and sequence-evolution oracle passed on every timestamp sample",
            "direct_wgpu": "equivalent per-frame size/alpha/spatial-variation and sequence-evolution oracle passed on every timestamp sample",
        },
        "interpretation": "Both paths timestamp the exact ordered retained compute/render pass sequence on the same Vulkan fallback adapter. Raw ticks remain retained for both paths; each path converts its own ticks with its own reported timestamp period, while GPU pass timestamps remain separate from host wall-clock characterization.",
    })
}
