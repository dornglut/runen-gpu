use super::common::{
    DirectWgpuContext, MEASURED_SAMPLES, Measurements, WARMUP_SAMPLES, micros,
    padded_bytes_per_row, ratio_summary, submit_and_map, tightly_pack_texture_rows,
};
use bytemuck::{Pod, Zeroable};
use runen_gpu::*;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

#[path = "../gpu_reaction_diffusion_native/workload.rs"]
mod retained;

const COMPUTE_WGSL: &str = include_str!("../gpu_reaction_diffusion_native/compute.wgsl");
const RENDER_WGSL: &str = include_str!("../gpu_reaction_diffusion_native/render.wgsl");
const BYTES_PER_PIXEL: u32 = 4;

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ReactionCell {
    a: f32,
    b: f32,
}

#[repr(C)]
#[derive(Debug, Clone, Copy, Pod, Zeroable)]
struct ReactionParams {
    width: u32,
    height: u32,
    dt: f32,
    feed: f32,
    kill: f32,
    diffusion_a: f32,
    diffusion_b: f32,
    _pad: f32,
}

fn reaction_params(width: u32, height: u32) -> ReactionParams {
    ReactionParams {
        width,
        height,
        dt: 1.0,
        feed: 0.055,
        kill: 0.062,
        diffusion_a: 1.0,
        diffusion_b: 0.5,
        _pad: 0.0,
    }
}

fn fixed_seed(width: u32, height: u32) -> Vec<ReactionCell> {
    let mut cells = vec![ReactionCell { a: 1.0, b: 0.0 }; usize::try_from(width * height).unwrap()];
    let half_w = width / 2;
    let half_h = height / 2;
    let radius = (width.min(height) / 10).max(2);
    for y in (half_h - radius)..(half_h + radius) {
        for x in (half_w - radius)..(half_w + radius) {
            let index = usize::try_from(y * width + x).unwrap();
            cells[index] = ReactionCell { a: 0.0, b: 1.0 };
        }
    }
    cells
}

fn label(value: &str) -> GpuResourceLabel {
    GpuResourceLabel::new(value).unwrap()
}

fn runengpu_context() -> GpuContext {
    let requirements = GpuCapabilityProfile::ComputeBaseline
        .requirements()
        .merge(&GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements())
        .unwrap();
    let descriptor = GpuContextDescriptor::new(requirements)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::ColorAttachment)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::CopySource)
        .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
        .with_allowed_backends([GpuBackendFamily::Vulkan])
        .with_label("G6-P01 reaction-diffusion RunenGPU comparison");
    let context = pollster::block_on(GpuContext::request(descriptor)).unwrap();
    assert_eq!(context.adapter_facts().backend(), GpuBackendFamily::Vulkan);
    assert_eq!(
        context.adapter_facts().fallback(),
        GpuFallbackStatus::ConfirmedFallback
    );
    context
}

fn assert_equivalent_adapter_selection(
    runengpu_context: &GpuContext,
    direct_context: &DirectWgpuContext,
) {
    let runengpu = runengpu_context.adapter_facts();
    assert_eq!(runengpu.backend(), GpuBackendFamily::Vulkan);
    assert_eq!(runengpu.fallback(), GpuFallbackStatus::ConfirmedFallback);
    assert_eq!(direct_context.adapter_info.backend, wgpu::Backend::Vulkan);
    assert_eq!(runengpu.vendor(), Some(direct_context.adapter_info.vendor));
    assert_eq!(runengpu.device(), Some(direct_context.adapter_info.device));
}

fn runengpu_adapter_facts_json(context: &GpuContext) -> Value {
    let facts = context.adapter_facts();
    json!({
        "backend": format!("{:?}", facts.backend()),
        "class": format!("{:?}", facts.class()),
        "software": format!("{:?}", facts.software()),
        "fallback": format!("{:?}", facts.fallback()),
        "name": facts.diagnostic_name(),
        "driver": facts.driver(),
        "driver_info": facts.driver_info(),
        "vendor": facts.vendor(),
        "device": facts.device(),
    })
}

fn progress_to_readbacks(
    context: &GpuContext,
    submission: &GpuSubmission,
    ids: &[GpuReadbackId],
) -> Vec<GpuReadbackBytes> {
    let handles = ids
        .iter()
        .map(|id| submission.readback(*id).unwrap().clone())
        .collect::<Vec<_>>();
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        context.progress();
        let mut all_ready = true;
        for handle in &handles {
            match handle.status() {
                GpuReadbackStatus::Ready(_) => {}
                GpuReadbackStatus::Failed(failure) => {
                    panic!("reaction-diffusion comparison readback failed: {failure:?}")
                }
                GpuReadbackStatus::Pending => all_ready = false,
            }
        }
        match submission.status() {
            GpuSubmissionStatus::Failed(failure) => {
                panic!("reaction-diffusion comparison submission failed: {failure:?}")
            }
            GpuSubmissionStatus::Completed if all_ready => break,
            GpuSubmissionStatus::Accepted | GpuSubmissionStatus::Completed => {}
        }
        assert!(
            Instant::now() < deadline,
            "reaction-diffusion comparison timed out"
        );
        std::thread::yield_now();
    }
    handles
        .into_iter()
        .map(|handle| match handle.status() {
            GpuReadbackStatus::Ready(bytes) => bytes,
            other => panic!("terminal reaction-diffusion readback must be ready, got {other:?}"),
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

fn validate_frame_bytes(bytes: &[u8], envelope: retained::Envelope) -> u64 {
    assert_eq!(
        bytes.len(),
        usize::try_from(envelope.width * envelope.height * BYTES_PER_PIXEL).unwrap()
    );
    let mut pixels = bytes.chunks_exact(4);
    assert!(pixels.remainder().is_empty());
    assert!(pixels.by_ref().all(|pixel| pixel[3] == 255));
    let mut pixels = bytes.chunks_exact(4);
    let first = pixels.next().unwrap();
    assert!(
        pixels.any(|pixel| pixel != first),
        "reaction-diffusion frame must contain spatially varying output"
    );
    fnv1a64(bytes)
}

fn validate_runengpu_frames(frames: &[GpuReadbackBytes], envelope: retained::Envelope) {
    assert_eq!(frames.len(), usize::try_from(envelope.frames).unwrap());
    let hashes = frames
        .iter()
        .map(|frame| {
            assert_eq!(frame.texture_format(), Some(GpuTextureFormat::Rgba8Unorm));
            validate_frame_bytes(frame.as_bytes(), envelope)
        })
        .collect::<Vec<_>>();
    assert_ne!(
        hashes.first(),
        hashes.last(),
        "retained sequence must evolve"
    );
}

fn validate_direct_frames(frames: &[Vec<u8>], envelope: retained::Envelope) {
    assert_eq!(frames.len(), usize::try_from(envelope.frames).unwrap());
    let hashes = frames
        .iter()
        .map(|frame| validate_frame_bytes(frame, envelope))
        .collect::<Vec<_>>();
    assert_ne!(hashes.first(), hashes.last(), "direct sequence must evolve");
}

fn sum_phases(
    first: BTreeMap<String, f64>,
    second: BTreeMap<String, f64>,
) -> BTreeMap<String, f64> {
    let mut result = first;
    for (phase, value) in second {
        *result.entry(phase).or_insert(0.0) += value;
    }
    result
}

fn runengpu_envelope_sample(
    context: &GpuContext,
    sources: &retained::ProgramSources,
    envelope: retained::Envelope,
) -> BTreeMap<String, f64> {
    let total_start = Instant::now();

    let author_start = Instant::now();
    let (graph_label, fragment, ids) = retained::offscreen_work(sources, envelope);
    let authoring_us = micros(author_start.elapsed());

    let graph_start = Instant::now();
    let graph = GpuPreparedWorkGraph::prepare(label(&graph_label), [fragment]).unwrap();
    let graph_prepare_us = micros(graph_start.elapsed());

    let backend_start = Instant::now();
    let prepared = pollster::block_on(context.prepare_submission(graph)).unwrap();
    let backend_prepare_us = micros(backend_start.elapsed());

    let submit_start = Instant::now();
    let submission = context.submit_prepared(prepared).unwrap();
    let submit_encode_and_queue_us = micros(submit_start.elapsed());

    let completion_start = Instant::now();
    let frames = progress_to_readbacks(context, &submission, &ids);
    let completion_readback_us = micros(completion_start.elapsed());
    validate_runengpu_frames(&frames, envelope);

    BTreeMap::from([
        ("authoring".to_owned(), authoring_us),
        ("graph_prepare".to_owned(), graph_prepare_us),
        ("backend_prepare".to_owned(), backend_prepare_us),
        (
            "boundary_prepare_or_record".to_owned(),
            graph_prepare_us + backend_prepare_us,
        ),
        (
            "submit_encode_and_queue".to_owned(),
            submit_encode_and_queue_us,
        ),
        (
            "boundary_prepare_record_submit".to_owned(),
            graph_prepare_us + backend_prepare_us + submit_encode_and_queue_us,
        ),
        ("completion_readback".to_owned(), completion_readback_us),
        ("total".to_owned(), micros(total_start.elapsed())),
    ])
}

fn runengpu_sample(
    context: &GpuContext,
    sources: &retained::ProgramSources,
) -> BTreeMap<String, f64> {
    let total_start = Instant::now();
    let first = runengpu_envelope_sample(context, sources, retained::ENVELOPES[0]);
    let second = runengpu_envelope_sample(context, sources, retained::ENVELOPES[1]);
    let mut phases = sum_phases(first, second);
    phases.insert("total".to_owned(), micros(total_start.elapsed()));
    phases
}

struct DirectPipelines {
    compute: wgpu::ComputePipeline,
    render: wgpu::RenderPipeline,
    cold_pipeline_us: f64,
}

fn direct_pipelines(context: &DirectWgpuContext) -> DirectPipelines {
    let start = Instant::now();
    let compute_shader = context
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("G6-P01 retained reaction-diffusion compute shader"),
            source: wgpu::ShaderSource::Wgsl(COMPUTE_WGSL.into()),
        });
    let render_shader = context
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("G6-P01 retained reaction-diffusion render shader"),
            source: wgpu::ShaderSource::Wgsl(RENDER_WGSL.into()),
        });
    let compute = context
        .device
        .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("G6-P01 reaction-diffusion compute pipeline"),
            layout: None,
            module: &compute_shader,
            entry_point: Some("cs_main"),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        });
    let targets = [Some(wgpu::ColorTargetState {
        format: wgpu::TextureFormat::Rgba8Unorm,
        blend: Some(wgpu::BlendState::REPLACE),
        write_mask: wgpu::ColorWrites::ALL,
    })];
    let render = context
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("G6-P01 reaction-diffusion render pipeline"),
            layout: None,
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &targets,
            }),
            multiview_mask: None,
            cache: None,
        });
    DirectPipelines {
        compute,
        render,
        cold_pipeline_us: micros(start.elapsed()),
    }
}

struct DirectResources {
    state_a: wgpu::Buffer,
    state_b: wgpu::Buffer,
    params: wgpu::Buffer,
    seed_upload_a: wgpu::Buffer,
    seed_upload_b: wgpu::Buffer,
    params_upload: wgpu::Buffer,
    target: wgpu::Texture,
    target_view: wgpu::TextureView,
    compute_ab: wgpu::BindGroup,
    compute_ba: wgpu::BindGroup,
    render_a: wgpu::BindGroup,
    render_b: wgpu::BindGroup,
    readbacks: Vec<wgpu::Buffer>,
}

fn state_bytes(envelope: retained::Envelope) -> u64 {
    u64::from(envelope.width)
        * u64::from(envelope.height)
        * u64::try_from(std::mem::size_of::<ReactionCell>()).unwrap()
}

fn params_bytes() -> u64 {
    u64::try_from(std::mem::size_of::<ReactionParams>()).unwrap()
}

fn logical_row_bytes(envelope: retained::Envelope) -> u32 {
    envelope.width * BYTES_PER_PIXEL
}

fn readback_staging_bytes(envelope: retained::Envelope) -> u64 {
    let logical_row = logical_row_bytes(envelope);
    let physical_row = padded_bytes_per_row(logical_row);
    u64::from(physical_row) * u64::from(envelope.height - 1) + u64::from(logical_row)
}

fn storage_buffer(context: &DirectWgpuContext, label: &str, size: u64) -> wgpu::Buffer {
    context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn direct_resources(
    context: &DirectWgpuContext,
    pipelines: &DirectPipelines,
    envelope: retained::Envelope,
) -> DirectResources {
    let seed = fixed_seed(envelope.width, envelope.height);
    let params_value = reaction_params(envelope.width, envelope.height);
    let seed_bytes = bytemuck::cast_slice(&seed);
    let params_host_bytes = bytemuck::bytes_of(&params_value);

    let state_a = storage_buffer(context, "G6-P01 reaction state a", state_bytes(envelope));
    let state_b = storage_buffer(context, "G6-P01 reaction state b", state_bytes(envelope));
    let params = storage_buffer(context, "G6-P01 reaction params", params_bytes());
    let seed_upload_a = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("G6-P01 reaction seed upload a"),
            contents: seed_bytes,
            usage: wgpu::BufferUsages::COPY_SRC,
        });
    let seed_upload_b = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("G6-P01 reaction seed upload b"),
            contents: seed_bytes,
            usage: wgpu::BufferUsages::COPY_SRC,
        });
    let params_upload = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("G6-P01 reaction params upload"),
            contents: params_host_bytes,
            usage: wgpu::BufferUsages::COPY_SRC,
        });

    let target = context.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("G6-P01 reaction-diffusion target"),
        size: wgpu::Extent3d {
            width: envelope.width,
            height: envelope.height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let target_view = target.create_view(&wgpu::TextureViewDescriptor::default());

    let compute_layout = pipelines.compute.get_bind_group_layout(0);
    let compute_ab = context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("G6-P01 reaction compute A to B"),
            layout: &compute_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: state_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params.as_entire_binding(),
                },
            ],
        });
    let compute_ba = context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("G6-P01 reaction compute B to A"),
            layout: &compute_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: state_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: params.as_entire_binding(),
                },
            ],
        });

    let render_layout = pipelines.render.get_bind_group_layout(0);
    let render_a = context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("G6-P01 reaction render state A"),
            layout: &render_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state_a.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params.as_entire_binding(),
                },
            ],
        });
    let render_b = context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("G6-P01 reaction render state B"),
            layout: &render_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state_b.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: params.as_entire_binding(),
                },
            ],
        });

    let readback_size = readback_staging_bytes(envelope);
    let readbacks = (0..envelope.frames)
        .map(|frame| {
            context.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(&format!("G6-P01 reaction frame {frame:03} readback")),
                size: readback_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        })
        .collect();

    DirectResources {
        state_a,
        state_b,
        params,
        seed_upload_a,
        seed_upload_b,
        params_upload,
        target,
        target_view,
        compute_ab,
        compute_ba,
        render_a,
        render_b,
        readbacks,
    }
}

fn direct_compute_dispatch(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    binding: &wgpu::BindGroup,
    envelope: retained::Envelope,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: Some("G6-P01 reaction-diffusion compute"),
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, binding, &[]);
    pass.dispatch_workgroups(
        envelope.width.div_ceil(retained::WORKGROUP),
        envelope.height.div_ceil(retained::WORKGROUP),
        1,
    );
}

fn direct_render(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::RenderPipeline,
    binding: &wgpu::BindGroup,
    view: &wgpu::TextureView,
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
    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some("G6-P01 reaction-diffusion render"),
        color_attachments: &color_attachments,
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, binding, &[]);
    pass.draw(0..3, 0..1);
}

fn direct_envelope_sample(
    context: &DirectWgpuContext,
    pipelines: &DirectPipelines,
    envelope: retained::Envelope,
) -> BTreeMap<String, f64> {
    let total_start = Instant::now();

    let resource_start = Instant::now();
    let resources = direct_resources(context, pipelines, envelope);
    let resource_setup_us = micros(resource_start.elapsed());

    let record_start = Instant::now();
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("G6-P01 reaction-diffusion direct encoder"),
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
    for frame in 0..envelope.frames {
        for _ in 0..envelope.iterations_per_frame {
            let binding = if current_is_a {
                &resources.compute_ab
            } else {
                &resources.compute_ba
            };
            direct_compute_dispatch(&mut encoder, &pipelines.compute, binding, envelope);
            current_is_a = !current_is_a;
        }
        let render_binding = if current_is_a {
            &resources.render_a
        } else {
            &resources.render_b
        };
        direct_render(
            &mut encoder,
            &pipelines.render,
            render_binding,
            &resources.target_view,
        );
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
    let command_buffer = encoder.finish();
    let command_record_us = micros(record_start.elapsed());

    let readback_refs = resources.readbacks.iter().collect::<Vec<_>>();
    let submitted = submit_and_map(context, command_buffer, &readback_refs);
    let unpack_start = Instant::now();
    let frames = submitted
        .mapped
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
    let row_unpack_us = micros(unpack_start.elapsed());
    let completion_readback_us = submitted.completion_readback_us + row_unpack_us;
    validate_direct_frames(&frames, envelope);

    BTreeMap::from([
        ("resource_setup".to_owned(), resource_setup_us),
        (
            "boundary_prepare_or_record".to_owned(),
            resource_setup_us + command_record_us,
        ),
        ("command_record".to_owned(), command_record_us),
        (
            "readback_registration".to_owned(),
            submitted.readback_registration_us,
        ),
        ("queue_submit".to_owned(), submitted.submit_call_us),
        (
            "boundary_prepare_record_submit".to_owned(),
            resource_setup_us + command_record_us + submitted.boundary_submit_us(),
        ),
        ("readback_row_unpack".to_owned(), row_unpack_us),
        ("completion_readback".to_owned(), completion_readback_us),
        ("total".to_owned(), micros(total_start.elapsed())),
    ])
}

fn direct_sample(
    context: &DirectWgpuContext,
    pipelines: &DirectPipelines,
) -> BTreeMap<String, f64> {
    let total_start = Instant::now();
    let first = direct_envelope_sample(context, pipelines, retained::ENVELOPES[0]);
    let second = direct_envelope_sample(context, pipelines, retained::ENVELOPES[1]);
    let mut phases = sum_phases(first, second);
    phases.insert("total".to_owned(), micros(total_start.elapsed()));
    phases
}

fn envelope_json(envelope: retained::Envelope) -> Value {
    let state = state_bytes(envelope);
    let params = params_bytes();
    let target =
        u64::from(envelope.width) * u64::from(envelope.height) * u64::from(BYTES_PER_PIXEL);
    let readback = readback_staging_bytes(envelope);
    json!({
        "name": envelope.name,
        "width": envelope.width,
        "height": envelope.height,
        "frames": envelope.frames,
        "iterations_per_frame": envelope.iterations_per_frame,
        "total_compute_dispatches": envelope.frames * envelope.iterations_per_frame,
        "render_draws": envelope.frames,
        "readbacks": envelope.frames,
        "workgroup": [retained::WORKGROUP, retained::WORKGROUP, 1],
        "logical_resource_bytes": state * 2 + params + target,
        "upload_staging_bytes": state * 2 + params,
        "readback_logical_bytes": target * u64::from(envelope.frames),
        "readback_staging_bytes": readback * u64::from(envelope.frames),
    })
}

pub(crate) fn compare() -> Value {
    let direct_context = DirectWgpuContext::request("G6-P01 reaction-diffusion direct WGPU");
    let direct_pipelines = direct_pipelines(&direct_context);

    let runengpu_context_start = Instant::now();
    let runengpu_context = runengpu_context();
    let runengpu_context_us = micros(runengpu_context_start.elapsed());
    assert_equivalent_adapter_selection(&runengpu_context, &direct_context);

    let source_start = Instant::now();
    let sources = retained::admitted_sources();
    let runengpu_source_admission_us = micros(source_start.elapsed());

    let mut runengpu_cold = None;
    let mut direct_cold = None;
    for _ in 0..WARMUP_SAMPLES {
        runengpu_cold = Some(runengpu_sample(&runengpu_context, &sources));
        direct_cold = Some(direct_sample(&direct_context, &direct_pipelines));
    }
    let runengpu_cold = runengpu_cold.expect("one RunenGPU warm-up sample is required");
    let direct_cold = direct_cold.expect("one direct-WGPU warm-up sample is required");
    let runengpu_cold_end_to_end_us = runengpu_context_us
        + runengpu_source_admission_us
        + runengpu_cold.get("total").copied().unwrap();
    let direct_cold_end_to_end_us = direct_context.setup_us
        + direct_pipelines.cold_pipeline_us
        + direct_cold.get("total").copied().unwrap();
    assert!(direct_cold_end_to_end_us > 0.0);

    let mut runengpu = Measurements::default();
    let mut direct = Measurements::default();
    for _ in 0..MEASURED_SAMPLES {
        runengpu.push(runengpu_sample(&runengpu_context, &sources));
        direct.push(direct_sample(&direct_context, &direct_pipelines));
    }

    json!({
        "workload": "G6-I01-reaction-diffusion",
        "comparison_envelope": {
            "model": "Gray-Scott",
            "seed": "A=1,B=0 with deterministic centered square A=0,B=1",
            "parameters": {
                "dt": 1.0,
                "feed": 0.055,
                "kill": 0.062,
                "diffusion_a": 1.0,
                "diffusion_b": 0.5,
            },
            "compute_shader": "gpu_reaction_diffusion_native/compute.wgsl",
            "render_shader": "gpu_reaction_diffusion_native/render.wgsl",
            "compute_entry": "cs_main",
            "render_entries": ["vs_main", "fs_main"],
            "format": "Rgba8Unorm",
            "render_load": "Clear black",
            "render_store": "Store",
            "submissions_per_sample_per_path": 2,
            "envelopes": retained::ENVELOPES.map(envelope_json),
        },
        "adapter_equivalence": {
            "criteria": "Vulkan forced-fallback selection with equal vendor/device identity",
            "runengpu": runengpu_adapter_facts_json(&runengpu_context),
            "direct_wgpu": {
                "facts": direct_context.facts_json(),
                "vendor": direct_context.adapter_info.vendor,
                "device": direct_context.adapter_info.device,
            },
        },
        "cold": {
            "scope": "fresh path context/device plus canonical shader/pipeline setup plus first complete two-envelope retained workload sample",
            "construction_order": ["direct_wgpu", "runengpu"],
            "normalized_end_to_end_us": {
                "runengpu": runengpu_cold_end_to_end_us,
                "direct_wgpu": direct_cold_end_to_end_us,
                "runengpu_over_direct_ratio": runengpu_cold_end_to_end_us / direct_cold_end_to_end_us,
            },
            "component_observations": {
                "runengpu_context_us": runengpu_context_us,
                "runengpu_source_admission_us": runengpu_source_admission_us,
                "runengpu_first_workload_phases_us": runengpu_cold,
                "direct_context_us": direct_context.setup_us,
                "direct_physical_pipeline_us": direct_pipelines.cold_pipeline_us,
                "direct_first_workload_phases_us": direct_cold,
                "note": "RunenGPU physical pipeline and binding realization occurs inside backend_prepare and submit_prepared also owns physical encoding/submission. Direct WGPU exposes resource/bind-group setup, command recording, readback callback registration, and queue submission separately. Compare normalized boundary/total fields, not unlike component fields pairwise.",
            },
        },
        "warm_lifecycle": {
            "warmup_samples": WARMUP_SAMPLES,
            "measured_samples": MEASURED_SAMPLES,
            "runengpu_context_and_physical_pipeline_cache_reused": true,
            "direct_wgpu_pipeline_reused": true,
            "per_sample_resources_recreated": true,
            "ping_pong_resources_persist_across_all_frames_inside_each_envelope": true,
            "per_sample_two_envelope_submissions": true,
        },
        "phase_comparability": {
            "ratio_phases": ["boundary_prepare_record_submit", "completion_readback", "total"],
            "runengpu_submit_component": "submit_prepared combines acceptance, physical encoding, readback callback registration, and queue submission and is not separable through the public boundary",
            "direct_readback_registration_component": "map_buffer_on_submit plus submitted-work-done callback registration",
            "direct_queue_submit_component": "queue.submit call only",
            "direct_completion_component": "submission completion + eight map callbacks per envelope + mapped host-byte copies + row normalization",
            "queue_submit_ratio_status": "unavailable because the RunenGPU public submit boundary intentionally combines additional execution work",
        },
        "runengpu": runengpu.to_json(),
        "direct_wgpu": direct.to_json(),
        "runengpu_over_direct_ratio": ratio_summary(
            &runengpu,
            &direct,
            &["boundary_prepare_record_submit", "completion_readback", "total"],
        ),
        "timestamp_evidence": {
            "supported_by_direct_adapter": direct_context.timestamp_supported,
            "status": "not-yet-instrumented",
        },
        "correctness": {
            "runengpu": "retained per-frame format/size/alpha/spatial-variation and sequence-evolution oracle passed on every sample",
            "direct_wgpu": "equivalent per-frame size/alpha/spatial-variation and sequence-evolution oracle passed on every sample",
        },
        "allocation_bytes": null,
        "allocation_bytes_status": "unavailable without new backend allocator instrumentation",
    })
}
