use super::common::{
    DirectWgpuContext, MEASURED_SAMPLES, Measurements, WARMUP_SAMPLES, micros,
    padded_bytes_per_row, ratio_summary, submit_and_map, summarize_u64, tightly_pack_texture_rows,
};
use runen_gpu::*;
use serde_json::{Value, json};
use std::collections::BTreeMap;
use std::time::{Duration, Instant};
use wgpu::util::DeviceExt;

const WIDTH: u32 = 8;
const HEIGHT: u32 = 8;
const CLEAR_PIXEL: [u8; 4] = [0, 0, 0, 255];
const DRAW_PIXEL: [u8; 4] = [255, 0, 0, 255];
const INDICES: [u32; 3] = [0, 1, 2];
const TIMESTAMP_BYTES: u64 = 16;
const KNOWN_PATTERN_WGSL: &str = include_str!("../gpu_offscreen_indexed_native/known_pattern.wgsl");

fn label(value: &str) -> GpuResourceLabel {
    GpuResourceLabel::new(value).unwrap()
}

fn provenance(value: &str) -> GpuResourceProvenance {
    GpuResourceProvenance::new(label(value), None, None)
}

fn common(value: &str) -> GpuResourceCommon {
    GpuResourceCommon::owned(
        label(value),
        GpuResourceLifetime::Transient,
        GpuMemoryIntent::Device,
        GpuReconstruction::SourceBacked,
        provenance(value),
    )
    .unwrap()
}

fn readback_staging_size() -> u64 {
    let logical_row = WIDTH * 4;
    let physical_row = padded_bytes_per_row(logical_row);
    u64::from(physical_row) * u64::from(HEIGHT - 1) + u64::from(logical_row)
}

fn runengpu_context(timestamp: bool) -> GpuContext {
    let mut requirements = GpuCapabilityProfile::OffscreenGraphicsBaseline.requirements();
    if timestamp {
        requirements
            .insert(GpuCapabilityRequirement::Required(
                GpuCapabilityFeature::TimestampQuery,
            ))
            .unwrap();
    }
    let descriptor = GpuContextDescriptor::new(requirements)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::ColorAttachment)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::CopySource)
        .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
        .with_allowed_backends([GpuBackendFamily::Vulkan])
        .with_label(if timestamp {
            "G6-P01 known-pattern RunenGPU timestamp evidence"
        } else {
            "G6-P01 known-pattern RunenGPU comparison"
        });
    let context = pollster::block_on(GpuContext::request(descriptor)).unwrap();
    assert_eq!(context.adapter_facts().backend(), GpuBackendFamily::Vulkan);
    assert_eq!(
        context.adapter_facts().fallback(),
        GpuFallbackStatus::ConfirmedFallback
    );
    assert_eq!(
        context
            .device_facts()
            .is_enabled(GpuCapabilityFeature::TimestampQuery),
        timestamp
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
    assert_eq!(
        runengpu.vendor(),
        Some(direct_context.adapter_info.vendor),
        "RunenGPU and direct-WGPU comparison paths must select the same adapter vendor"
    );
    assert_eq!(
        runengpu.device(),
        Some(direct_context.adapter_info.device),
        "RunenGPU and direct-WGPU comparison paths must select the same adapter device"
    );
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

fn admitted_render_source() -> GpuAdmittedProgramSource {
    let identity = GpuProgramSourceIdentity::new(
        GpuProgramSourceOwnerId::allocate().unwrap(),
        GpuProgramSourceKey::new("proof.direct-cost.offscreen.indexed-known-pattern").unwrap(),
        GpuProgramSourceRevision::try_from_raw(1).unwrap(),
    );
    let mut sources = GpuProgramSourceRegistry::new(4, 16 * 1024).unwrap();
    sources
        .admit_wgsl(
            identity,
            KNOWN_PATTERN_WGSL,
            GpuProgramSourceProvenance::new("direct-cost-indexed-offscreen", None).unwrap(),
        )
        .unwrap()
}

fn render_pipeline() -> GpuRenderPipelineDescriptor {
    let vertex = GpuEntryPointName::new("vs_main").unwrap();
    let fragment = GpuEntryPointName::new("fs_main").unwrap();
    let program = GpuProgramDescriptor::new(
        admitted_render_source(),
        [vertex.clone(), fragment.clone()],
        std::iter::empty::<GpuBindingLayoutRefinement>(),
    )
    .unwrap();
    let color_target = GpuColorTargetStateDescriptor::new(
        GpuTextureFormat::Rgba8Unorm,
        GpuBlendMode::Replace,
        GpuColorWriteMask::ALL,
    )
    .unwrap();
    let state = GpuRenderPipelineStateDescriptor::new(
        GpuVertexInputStateDescriptor::new([]).unwrap(),
        Some(GpuFragmentOutputStateDescriptor::new([color_target])),
        GpuPrimitiveStateDescriptor::default(),
        None,
        GpuMultisampleStateDescriptor::default(),
    )
    .unwrap();
    GpuRenderPipelineDescriptor::new(
        program,
        GpuRenderEntryPoints::new(vertex, Some(fragment)),
        state,
        GpuPipelineConfiguration::default(),
    )
    .unwrap()
}

struct RunenGpuFragment {
    fragment: GpuWorkFragment,
    image_readback: GpuReadbackId,
    timestamp_readback: Option<GpuReadbackId>,
}

fn runengpu_fragment(pipeline: &GpuRenderPipelineDescriptor, timestamp: bool) -> RunenGpuFragment {
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let texture_label = label("direct-cost indexed offscreen color target");
    let texture = allocator
        .allocate_texture_handle(
            GpuTextureDescriptor::new(
                common("direct-cost indexed offscreen color target"),
                GpuTextureDimension::D2,
                GpuTextureExtent::new(&texture_label, GpuTextureDimension::D2, WIDTH, HEIGHT, 1)
                    .unwrap(),
                1,
                1,
                GpuTextureFormat::Rgba8Unorm,
                GpuTextureUsages::new(
                    &texture_label,
                    [
                        GpuTextureUsage::ColorAttachment,
                        GpuTextureUsage::CopySource,
                    ],
                )
                .unwrap(),
                GpuTextureInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();
    let subresources = GpuTextureSubresourceRange::new(
        texture.descriptor().common().label(),
        0,
        1,
        0,
        1,
        GpuTextureAspect::Color,
    )
    .unwrap();
    let view = allocator
        .allocate_texture_view_handle(
            GpuTextureViewDescriptor::new(
                common("direct-cost indexed offscreen color target view"),
                &texture,
                None,
                GpuTextureDimension::D2,
                subresources,
            )
            .unwrap(),
        )
        .unwrap();
    let index_label = label("direct-cost indexed offscreen index buffer");
    let index_data = PreparedGpuData::<TransferData>::from_pod_transfer(
        "direct-cost indexed offscreen indices",
        &INDICES,
        provenance("direct-cost indexed offscreen indices"),
    )
    .unwrap();
    let index_buffer = allocator
        .allocate_buffer_handle(
            GpuBufferDescriptor::new(
                common("direct-cost indexed offscreen index buffer"),
                u64::try_from(core::mem::size_of_val(&INDICES)).unwrap(),
                GpuBufferUsages::new(
                    &index_label,
                    [GpuBufferUsage::Index, GpuBufferUsage::CopyDestination],
                )
                .unwrap(),
                GpuBufferInitialization::Prepared(index_data),
            )
            .unwrap(),
        )
        .unwrap();

    let timestamp_resources = timestamp.then(|| {
        let query_set = allocator
            .allocate_query_set_handle(
                GpuQuerySetDescriptor::new(
                    common("direct-cost indexed offscreen timestamp query set"),
                    GpuQueryKind::Timestamp,
                    2,
                )
                .unwrap(),
            )
            .unwrap();
        let resolve_label = label("direct-cost indexed offscreen timestamp resolve");
        let resolve = allocator
            .allocate_buffer_handle(
                GpuBufferDescriptor::new(
                    common("direct-cost indexed offscreen timestamp resolve"),
                    TIMESTAMP_BYTES,
                    GpuBufferUsages::new(
                        &resolve_label,
                        [GpuBufferUsage::QueryResolve, GpuBufferUsage::CopySource],
                    )
                    .unwrap(),
                    GpuBufferInitialization::Uninitialized,
                )
                .unwrap(),
            )
            .unwrap();
        (query_set, resolve)
    });

    let bindings = GpuRuntimeBindingSet::new(pipeline.layout().clone(), []).unwrap();
    let index_binding = GpuIndexBufferBinding::new(
        &index_buffer,
        GpuBufferRange::whole(&index_buffer).unwrap(),
        GpuIndexFormat::Uint32,
    )
    .unwrap();
    let draw = GpuRenderDraw::new(
        pipeline.clone(),
        bindings,
        [],
        Some(index_binding),
        GpuDrawIntent::indexed(
            GpuDrawRange::new(0, u32::try_from(INDICES.len()).unwrap()).unwrap(),
            0,
            GpuDrawRange::new(0, 1).unwrap(),
        ),
        GpuViewport::new(0.0, 0.0, WIDTH as f32, HEIGHT as f32, 0.0, 1.0).unwrap(),
        GpuScissorRect::new(0, 0, WIDTH / 2, HEIGHT).unwrap(),
        GpuBlendConstant::new(0.0, 0.0, 0.0, 0.0).unwrap(),
        0,
    )
    .unwrap();
    let attachment = GpuRenderColorAttachment::new(
        view.clone(),
        GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 1.0).unwrap()),
        GpuAttachmentStore::Store,
        None,
    )
    .unwrap();
    let timestamp_writes = timestamp_resources
        .as_ref()
        .map(|(query_set, _)| GpuTimestampWrites::new(query_set, Some(0), Some(1)).unwrap());
    let render = GpuRenderOperation::new([attachment], None, [draw], timestamp_writes).unwrap();

    let image_region = GpuTextureCopyRegion::new(
        &texture,
        0,
        GpuTextureOrigin::new(0, 0, 0),
        GpuTextureAspect::Color,
        GpuCopyExtent::new(WIDTH, HEIGHT, 1).unwrap(),
    )
    .unwrap();
    let image_readback = GpuReadbackId::allocate().unwrap();
    let image_operation = GpuReadbackOperation::new(image_region.into(), image_readback).unwrap();

    let timestamp_readback = timestamp_resources.as_ref().map(|(_, resolve)| {
        let id = GpuReadbackId::allocate().unwrap();
        (
            id,
            GpuReadbackOperation::new(GpuBufferRegion::whole(resolve).unwrap().into(), id).unwrap(),
        )
    });

    let mut builder = GpuWorkFragmentBuilder::new(
        label("direct-cost indexed offscreen fragment"),
        provenance("direct-cost indexed offscreen fragment"),
    );
    builder.declare_resource(texture.into()).unwrap();
    builder.declare_resource(view.into()).unwrap();
    builder.declare_resource(index_buffer.into()).unwrap();
    if let Some((query_set, resolve)) = &timestamp_resources {
        builder.declare_resource(query_set.clone().into()).unwrap();
        builder.declare_resource(resolve.clone().into()).unwrap();
    }
    builder
        .add_node(
            label("direct-cost indexed offscreen draw"),
            GpuWorkOperation::Render(render),
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::Automatic,
            provenance("direct-cost indexed offscreen draw"),
        )
        .unwrap();
    if let Some((query_set, resolve)) = &timestamp_resources {
        let resolve_operation = GpuQueryResolveOperation::new(
            query_set,
            GpuQueryRange::new(query_set, 0, 2).unwrap(),
            resolve,
            0,
        )
        .unwrap();
        builder
            .add_node(
                label("direct-cost indexed offscreen timestamp resolve"),
                GpuWorkOperation::Resolve(resolve_operation),
                [],
                GpuCapabilityRequirements::new(),
                GpuExecutionPreference::Automatic,
                provenance("direct-cost indexed offscreen timestamp resolve"),
            )
            .unwrap();
    }
    builder
        .add_node(
            label("direct-cost indexed offscreen readback"),
            GpuWorkOperation::Readback(image_operation),
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::Automatic,
            provenance("direct-cost indexed offscreen readback"),
        )
        .unwrap();
    if let Some((_, operation)) = timestamp_readback.as_ref() {
        builder
            .add_node(
                label("direct-cost indexed offscreen timestamp readback"),
                GpuWorkOperation::Readback(operation.clone()),
                [],
                GpuCapabilityRequirements::new(),
                GpuExecutionPreference::Automatic,
                provenance("direct-cost indexed offscreen timestamp readback"),
            )
            .unwrap();
    }

    RunenGpuFragment {
        fragment: builder.finish().unwrap(),
        image_readback,
        timestamp_readback: timestamp_readback.map(|(id, _)| id),
    }
}

fn progress_runengpu(
    context: &GpuContext,
    submission: &GpuSubmission,
    readback_ids: &[GpuReadbackId],
) -> Vec<GpuReadbackBytes> {
    let readbacks = readback_ids
        .iter()
        .map(|id| submission.readback(*id).unwrap().clone())
        .collect::<Vec<_>>();
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        context.progress();
        let mut all_ready = true;
        for readback in &readbacks {
            match readback.status() {
                GpuReadbackStatus::Ready(_) => {}
                GpuReadbackStatus::Failed(failure) => {
                    panic!("RunenGPU known-pattern readback failed: {failure:?}")
                }
                GpuReadbackStatus::Pending => all_ready = false,
            }
        }
        match submission.status() {
            GpuSubmissionStatus::Failed(failure) => {
                panic!("RunenGPU known-pattern submission failed: {failure:?}")
            }
            GpuSubmissionStatus::Completed if all_ready => break,
            GpuSubmissionStatus::Accepted | GpuSubmissionStatus::Completed => {}
        }
        assert!(
            Instant::now() < deadline,
            "RunenGPU known-pattern completion/readback timed out"
        );
        std::thread::yield_now();
    }
    readbacks
        .into_iter()
        .map(|readback| match readback.status() {
            GpuReadbackStatus::Ready(bytes) => bytes,
            other => panic!("terminal RunenGPU known-pattern readback must be ready: {other:?}"),
        })
        .collect()
}

fn pixel_at(bytes: &[u8], x: u32, y: u32) -> [u8; 4] {
    let offset = usize::try_from((y * WIDTH + x) * 4).unwrap();
    bytes[offset..offset + 4].try_into().unwrap()
}

fn assert_known_pattern(bytes: &[u8]) {
    assert_eq!(bytes.len(), usize::try_from(WIDTH * HEIGHT * 4).unwrap());
    for (x, y) in [(1, 1), (2, 6)] {
        assert_eq!(pixel_at(bytes, x, y), DRAW_PIXEL);
    }
    for (x, y) in [(5, 1), (6, 6)] {
        assert_eq!(pixel_at(bytes, x, y), CLEAR_PIXEL);
    }
}

fn timestamp_delta(bytes: &[u8]) -> u64 {
    let mut timestamps = bytes.chunks_exact(8);
    assert!(timestamps.remainder().is_empty());
    let beginning = u64::from_ne_bytes(timestamps.next().unwrap().try_into().unwrap());
    let end = u64::from_ne_bytes(timestamps.next().unwrap().try_into().unwrap());
    assert!(timestamps.next().is_none());
    assert!(end >= beginning, "timestamp end must not precede beginning");
    end - beginning
}

fn runengpu_sample(
    context: &GpuContext,
    pipeline: &GpuRenderPipelineDescriptor,
) -> BTreeMap<String, f64> {
    let total_start = Instant::now();
    let author_start = Instant::now();
    let authored = runengpu_fragment(pipeline, false);
    let author_us = micros(author_start.elapsed());

    let prepare_start = Instant::now();
    let graph = GpuPreparedWorkGraph::prepare(
        label("G6-P01 known-pattern RunenGPU graph"),
        [authored.fragment],
    )
    .unwrap();
    let graph_prepare_us = micros(prepare_start.elapsed());

    let backend_prepare_start = Instant::now();
    let prepared = pollster::block_on(context.prepare_submission(graph)).unwrap();
    let backend_prepare_us = micros(backend_prepare_start.elapsed());

    let submit_start = Instant::now();
    let submission = context.submit_prepared(prepared).unwrap();
    let submit_encode_and_queue_us = micros(submit_start.elapsed());

    let completion_start = Instant::now();
    let readbacks = progress_runengpu(context, &submission, &[authored.image_readback]);
    let completion_readback_us = micros(completion_start.elapsed());
    assert_eq!(
        readbacks[0].texture_format(),
        Some(GpuTextureFormat::Rgba8Unorm)
    );
    assert_known_pattern(readbacks[0].as_bytes());

    let mut phases = BTreeMap::new();
    phases.insert("authoring".to_owned(), author_us);
    phases.insert("graph_prepare".to_owned(), graph_prepare_us);
    phases.insert("backend_prepare".to_owned(), backend_prepare_us);
    phases.insert(
        "boundary_prepare_or_record".to_owned(),
        graph_prepare_us + backend_prepare_us,
    );
    phases.insert(
        "submit_encode_and_queue".to_owned(),
        submit_encode_and_queue_us,
    );
    phases.insert(
        "boundary_prepare_record_submit".to_owned(),
        graph_prepare_us + backend_prepare_us + submit_encode_and_queue_us,
    );
    phases.insert("completion_readback".to_owned(), completion_readback_us);
    phases.insert("total".to_owned(), micros(total_start.elapsed()));
    phases
}

fn runengpu_timestamp_sample(context: &GpuContext, pipeline: &GpuRenderPipelineDescriptor) -> u64 {
    let authored = runengpu_fragment(pipeline, true);
    let timestamp_id = authored
        .timestamp_readback
        .expect("timestamp evidence fragment must expose timestamp readback");
    let graph = GpuPreparedWorkGraph::prepare(
        label("G6-P01 known-pattern RunenGPU timestamp graph"),
        [authored.fragment],
    )
    .unwrap();
    let prepared = pollster::block_on(context.prepare_submission(graph)).unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    let readbacks = progress_runengpu(
        context,
        &submission,
        &[authored.image_readback, timestamp_id],
    );
    assert_eq!(
        readbacks[0].texture_format(),
        Some(GpuTextureFormat::Rgba8Unorm)
    );
    assert_known_pattern(readbacks[0].as_bytes());
    timestamp_delta(readbacks[1].as_bytes())
}

struct DirectPipeline {
    pipeline: wgpu::RenderPipeline,
    cold_pipeline_us: f64,
}

fn direct_pipeline(context: &DirectWgpuContext) -> DirectPipeline {
    let start = Instant::now();
    let shader = context
        .device
        .create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("G6-P01 known-pattern shader"),
            source: wgpu::ShaderSource::Wgsl(KNOWN_PATTERN_WGSL.into()),
        });
    let layout = context
        .device
        .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("G6-P01 known-pattern pipeline layout"),
            bind_group_layouts: &[],
            immediate_size: 0,
        });
    let targets = [Some(wgpu::ColorTargetState {
        format: wgpu::TextureFormat::Rgba8Unorm,
        blend: Some(wgpu::BlendState::REPLACE),
        write_mask: wgpu::ColorWrites::ALL,
    })];
    let pipeline = context
        .device
        .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("G6-P01 known-pattern pipeline"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                buffers: &[],
            },
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                targets: &targets,
            }),
            multiview_mask: None,
            cache: None,
        });
    DirectPipeline {
        pipeline,
        cold_pipeline_us: micros(start.elapsed()),
    }
}

struct DirectResources {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    index_buffer: wgpu::Buffer,
    index_upload: wgpu::Buffer,
    image_readback: wgpu::Buffer,
}

fn direct_resources(context: &DirectWgpuContext) -> DirectResources {
    let texture = context.device.create_texture(&wgpu::TextureDescriptor {
        label: Some("G6-P01 known-pattern target"),
        size: wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
    let index_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("G6-P01 known-pattern index buffer"),
        size: u64::try_from(core::mem::size_of_val(&INDICES)).unwrap(),
        usage: wgpu::BufferUsages::INDEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    let index_upload = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("G6-P01 known-pattern index upload"),
            contents: bytemuck::cast_slice(&INDICES),
            usage: wgpu::BufferUsages::COPY_SRC,
        });
    let image_readback = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("G6-P01 known-pattern readback"),
        size: readback_staging_size(),
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    DirectResources {
        texture,
        view,
        index_buffer,
        index_upload,
        image_readback,
    }
}

fn record_known_pattern(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::RenderPipeline,
    resources: &DirectResources,
    timestamps: Option<(&wgpu::QuerySet, u32, u32)>,
) {
    encoder.copy_buffer_to_buffer(
        &resources.index_upload,
        0,
        &resources.index_buffer,
        0,
        u64::try_from(core::mem::size_of_val(&INDICES)).unwrap(),
    );
    {
        let color_attachments = [Some(wgpu::RenderPassColorAttachment {
            view: &resources.view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                store: wgpu::StoreOp::Store,
            },
        })];
        let timestamp_writes =
            timestamps.map(
                |(query_set, beginning, end)| wgpu::RenderPassTimestampWrites {
                    query_set,
                    beginning_of_pass_write_index: Some(beginning),
                    end_of_pass_write_index: Some(end),
                },
            );
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("G6-P01 known-pattern pass"),
            color_attachments: &color_attachments,
            depth_stencil_attachment: None,
            timestamp_writes,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_index_buffer(resources.index_buffer.slice(..), wgpu::IndexFormat::Uint32);
        pass.set_viewport(0.0, 0.0, WIDTH as f32, HEIGHT as f32, 0.0, 1.0);
        pass.set_scissor_rect(0, 0, WIDTH / 2, HEIGHT);
        pass.set_blend_constant(wgpu::Color::TRANSPARENT);
        pass.set_stencil_reference(0);
        pass.draw_indexed(0..u32::try_from(INDICES.len()).unwrap(), 0, 0..1);
    }
    let physical_row = padded_bytes_per_row(WIDTH * 4);
    encoder.copy_texture_to_buffer(
        wgpu::TexelCopyTextureInfo {
            texture: &resources.texture,
            mip_level: 0,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyBufferInfo {
            buffer: &resources.image_readback,
            layout: wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(physical_row),
                rows_per_image: None,
            },
        },
        wgpu::Extent3d {
            width: WIDTH,
            height: HEIGHT,
            depth_or_array_layers: 1,
        },
    );
}

fn direct_sample(
    context: &DirectWgpuContext,
    pipeline: &wgpu::RenderPipeline,
) -> BTreeMap<String, f64> {
    let total_start = Instant::now();
    let resource_start = Instant::now();
    let resources = direct_resources(context);
    let resource_setup_us = micros(resource_start.elapsed());

    let record_start = Instant::now();
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("G6-P01 known-pattern encoder"),
        });
    record_known_pattern(&mut encoder, pipeline, &resources, None);
    let command_buffer = encoder.finish();
    let command_record_us = micros(record_start.elapsed());

    let submitted = submit_and_map(context, command_buffer, &[&resources.image_readback]);
    let physical_row = padded_bytes_per_row(WIDTH * 4);
    let row_unpack_start = Instant::now();
    let bytes = tightly_pack_texture_rows(&submitted.mapped[0], WIDTH, HEIGHT, 4, physical_row);
    let row_unpack_us = micros(row_unpack_start.elapsed());
    let completion_readback_us = submitted.completion_readback_us + row_unpack_us;
    assert_known_pattern(&bytes);

    let mut phases = BTreeMap::new();
    phases.insert("resource_setup".to_owned(), resource_setup_us);
    phases.insert(
        "boundary_prepare_or_record".to_owned(),
        resource_setup_us + command_record_us,
    );
    phases.insert("command_record".to_owned(), command_record_us);
    phases.insert(
        "readback_registration".to_owned(),
        submitted.readback_registration_us,
    );
    phases.insert("queue_submit".to_owned(), submitted.submit_call_us);
    phases.insert(
        "boundary_prepare_record_submit".to_owned(),
        resource_setup_us + command_record_us + submitted.boundary_submit_us(),
    );
    phases.insert("readback_row_unpack".to_owned(), row_unpack_us);
    phases.insert("completion_readback".to_owned(), completion_readback_us);
    phases.insert("total".to_owned(), micros(total_start.elapsed()));
    phases
}

fn direct_timestamp_sample(context: &DirectWgpuContext, pipeline: &wgpu::RenderPipeline) -> u64 {
    let resources = direct_resources(context);
    let query_set = context.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("G6-P01 known-pattern direct timestamp query set"),
        ty: wgpu::QueryType::Timestamp,
        count: 2,
    });
    let resolve = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("G6-P01 known-pattern direct timestamp resolve"),
        size: TIMESTAMP_BYTES,
        usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });
    let timestamp_readback = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("G6-P01 known-pattern direct timestamp readback"),
        size: TIMESTAMP_BYTES,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("G6-P01 known-pattern direct timestamp encoder"),
        });
    record_known_pattern(&mut encoder, pipeline, &resources, Some((&query_set, 0, 1)));
    encoder.resolve_query_set(&query_set, 0..2, &resolve, 0);
    encoder.copy_buffer_to_buffer(&resolve, 0, &timestamp_readback, 0, TIMESTAMP_BYTES);
    let command_buffer = encoder.finish();
    let submitted = submit_and_map(
        context,
        command_buffer,
        &[&resources.image_readback, &timestamp_readback],
    );
    let physical_row = padded_bytes_per_row(WIDTH * 4);
    let image = tightly_pack_texture_rows(&submitted.mapped[0], WIDTH, HEIGHT, 4, physical_row);
    assert_known_pattern(&image);
    timestamp_delta(&submitted.mapped[1])
}

fn timestamp_evidence(wall_runengpu: &GpuContext, wall_direct: &DirectWgpuContext) -> Value {
    assert!(
        wall_direct.timestamp_supported,
        "accepted G6-P01 Lavapipe adapter reports timestamp-query support"
    );
    let runengpu_context = runengpu_context(true);
    let direct_context =
        DirectWgpuContext::request_timestamp("G6-P01 known-pattern direct WGPU timestamp evidence");
    assert_equivalent_adapter_selection(&runengpu_context, &direct_context);
    assert_eq!(
        wall_runengpu.adapter_facts().vendor(),
        runengpu_context.adapter_facts().vendor()
    );
    assert_eq!(
        wall_runengpu.adapter_facts().device(),
        runengpu_context.adapter_facts().device()
    );
    assert_eq!(
        wall_direct.adapter_info.vendor,
        direct_context.adapter_info.vendor
    );
    assert_eq!(
        wall_direct.adapter_info.device,
        direct_context.adapter_info.device
    );

    let runengpu_pipeline = render_pipeline();
    let direct_pipeline = direct_pipeline(&direct_context);
    for _ in 0..WARMUP_SAMPLES {
        let _ = runengpu_timestamp_sample(&runengpu_context, &runengpu_pipeline);
        let _ = direct_timestamp_sample(&direct_context, &direct_pipeline.pipeline);
    }
    let mut runengpu_ticks = Vec::with_capacity(MEASURED_SAMPLES);
    let mut direct_ticks = Vec::with_capacity(MEASURED_SAMPLES);
    for _ in 0..MEASURED_SAMPLES {
        runengpu_ticks.push(runengpu_timestamp_sample(
            &runengpu_context,
            &runengpu_pipeline,
        ));
        direct_ticks.push(direct_timestamp_sample(
            &direct_context,
            &direct_pipeline.pipeline,
        ));
    }
    let runengpu_period_ns = f64::from(
        runengpu_context
            .timestamp_period_ns()
            .expect("timestamp-query-admitted RunenGPU context must expose timestamp period"),
    );
    let direct_period_ns = f64::from(direct_context.queue.get_timestamp_period());
    assert!(runengpu_period_ns.is_finite() && runengpu_period_ns > 0.0);
    assert!(direct_period_ns.is_finite() && direct_period_ns > 0.0);
    let runengpu_ns = runengpu_ticks
        .iter()
        .map(|ticks| *ticks as f64 * runengpu_period_ns)
        .collect::<Vec<_>>();
    let direct_ns = direct_ticks
        .iter()
        .map(|ticks| *ticks as f64 * direct_period_ns)
        .collect::<Vec<_>>();

    json!({
        "status": "measured",
        "separate_from_wall_clock_samples": true,
        "pass_boundaries": "beginning/end of the one retained indexed render pass",
        "warmup_samples": WARMUP_SAMPLES,
        "measured_samples": MEASURED_SAMPLES,
        "runengpu": {
            "raw_delta_ticks": runengpu_ticks,
            "summary_ticks": summarize_u64(&runengpu_ticks),
            "timestamp_period_ns": runengpu_period_ns,
            "timestamp_period_status": "reported by GpuContext::timestamp_period_ns()",
            "delta_ns": runengpu_ns,
        },
        "direct_wgpu": {
            "raw_delta_ticks": direct_ticks,
            "summary_ticks": summarize_u64(&direct_ticks),
            "timestamp_period_ns": direct_period_ns,
            "delta_ns": direct_ns,
        },
        "interpretation": "Both paths write timestamps at equivalent retained render-pass boundaries on the same Vulkan fallback adapter. Raw ticks remain retained for both paths; each path converts its own ticks with its own reported timestamp period, keeping GPU timestamp evidence separate from wall-clock samples.",
    })
}

pub(crate) fn compare() -> Value {
    let direct_context = DirectWgpuContext::request("G6-P01 known-pattern direct WGPU");
    let direct_pipeline = direct_pipeline(&direct_context);

    let runengpu_context_start = Instant::now();
    let runengpu_context = runengpu_context(false);
    let runengpu_context_us = micros(runengpu_context_start.elapsed());
    assert_equivalent_adapter_selection(&runengpu_context, &direct_context);

    let runengpu_pipeline_start = Instant::now();
    let runengpu_pipeline = render_pipeline();
    let runengpu_pipeline_descriptor_us = micros(runengpu_pipeline_start.elapsed());

    let mut runengpu_cold = None;
    let mut direct_cold = None;
    for _ in 0..WARMUP_SAMPLES {
        runengpu_cold = Some(runengpu_sample(&runengpu_context, &runengpu_pipeline));
        direct_cold = Some(direct_sample(&direct_context, &direct_pipeline.pipeline));
    }
    let runengpu_cold = runengpu_cold.expect("one RunenGPU first-use sample is required");
    let direct_cold = direct_cold.expect("one direct-WGPU first-use sample is required");
    let runengpu_cold_end_to_end_us = runengpu_context_us
        + runengpu_pipeline_descriptor_us
        + runengpu_cold.get("total").copied().unwrap();
    let direct_cold_end_to_end_us = direct_context.setup_us
        + direct_pipeline.cold_pipeline_us
        + direct_cold.get("total").copied().unwrap();
    assert!(direct_cold_end_to_end_us > 0.0);

    let mut runengpu = Measurements::default();
    let mut direct = Measurements::default();
    for _ in 0..MEASURED_SAMPLES {
        runengpu.push(runengpu_sample(&runengpu_context, &runengpu_pipeline));
        direct.push(direct_sample(&direct_context, &direct_pipeline.pipeline));
    }
    assert_eq!(runengpu.len(), MEASURED_SAMPLES);
    assert_eq!(direct.len(), MEASURED_SAMPLES);
    let timestamps = timestamp_evidence(&runengpu_context, &direct_context);

    json!({
        "workload": "G6-C01-known-pattern-offscreen-draw",
        "comparison_envelope": {
            "width": WIDTH,
            "height": HEIGHT,
            "format": "Rgba8Unorm",
            "clear": CLEAR_PIXEL,
            "indices": INDICES,
            "draw": "one indexed triangle",
            "viewport": [0, 0, WIDTH, HEIGHT],
            "scissor": [0, 0, WIDTH / 2, HEIGHT],
            "shader_source": "shared retained G6-C01 fixture",
            "readback_logical_bytes": WIDTH * HEIGHT * 4,
            "readback_staging_bytes_per_path": readback_staging_size(),
            "index_upload_bytes": core::mem::size_of_val(&INDICES),
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
            "scope": "fresh path context/device plus path-specific program/pipeline setup plus first complete submission/readback within one test process",
            "construction_order": ["direct_wgpu", "runengpu"],
            "normalized_end_to_end_us": {
                "runengpu": runengpu_cold_end_to_end_us,
                "direct_wgpu": direct_cold_end_to_end_us,
                "runengpu_over_direct_ratio": runengpu_cold_end_to_end_us / direct_cold_end_to_end_us,
            },
            "component_observations": {
                "runengpu_context_us": runengpu_context_us,
                "runengpu_program_pipeline_descriptor_us": runengpu_pipeline_descriptor_us,
                "runengpu_first_submission_phases_us": runengpu_cold,
                "direct_context_us": direct_context.setup_us,
                "direct_physical_pipeline_us": direct_pipeline.cold_pipeline_us,
                "direct_first_submission_phases_us": direct_cold,
                "note": "RunenGPU physical pipeline realization occurs during first backend_prepare and submit_prepared also owns physical encoding/readback registration/submission. Direct WGPU exposes resource creation, command recording, readback callback registration, and queue submission separately. Direct completion/readback includes mapped host-byte materialization and texture row-unpadding. Compare normalized boundary/total fields, not unlike component fields pairwise.",
            },
        },
        "warm_lifecycle": {
            "warmup_samples": WARMUP_SAMPLES,
            "measured_samples": MEASURED_SAMPLES,
            "runengpu_program_pipeline_identity_reused": true,
            "direct_wgpu_pipeline_reused": true,
            "per_sample_resources_recreated": true,
            "per_sample_logical_graph_or_command_recording": true,
        },
        "phase_comparability": {
            "ratio_phases": ["boundary_prepare_record_submit", "completion_readback", "total"],
            "runengpu_submit_component": "submit_prepared combines acceptance, physical encoding, readback callback registration, and queue submission and is not separable through the public boundary",
            "direct_readback_registration_component": "map_buffer_on_submit plus submitted-work-done callback registration",
            "direct_queue_submit_component": "queue.submit call only",
            "direct_completion_component": "submission completion + map callback + mapped host-byte copy + texture row-unpadding",
            "queue_submit_ratio_status": "unavailable because the RunenGPU public submit boundary intentionally combines additional execution work",
        },
        "runengpu": runengpu.to_json(),
        "direct_wgpu": direct.to_json(),
        "runengpu_over_direct_ratio": ratio_summary(
            &runengpu,
            &direct,
            &["boundary_prepare_record_submit", "completion_readback", "total"],
        ),
        "timestamp_evidence": timestamps,
        "correctness": "passed exact retained selected-pixel oracle on every wall-clock and timestamp sample",
        "allocation_bytes": null,
        "allocation_bytes_status": "unavailable without new backend allocator instrumentation",
    })
}
