use runen_gpu::*;
use std::time::{Duration, Instant};

const DEPTH_FORMATS: [GpuTextureFormat; 2] = [
    GpuTextureFormat::Depth16Unorm,
    GpuTextureFormat::Depth24Plus,
];

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

fn requirements() -> GpuCapabilityRequirements {
    let mut requirements = GpuCapabilityRequirements::new();
    for feature in [
        GpuCapabilityFeature::RenderPipeline,
        GpuCapabilityFeature::DepthAttachment,
        GpuCapabilityFeature::Copy,
    ] {
        requirements
            .insert(GpuCapabilityRequirement::Required(feature))
            .unwrap();
    }
    requirements
}

fn census_context() -> GpuContext {
    let descriptor = GpuContextDescriptor::new(requirements())
        .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
        .with_allowed_backends([GpuBackendFamily::Vulkan])
        .with_label("R1 native baseline depth census");
    pollster::block_on(GpuContext::request(descriptor))
        .expect("native conformance must provide the declared Vulkan software baseline")
}

fn depth_context_for(format: GpuTextureFormat) -> GpuContext {
    let descriptor = GpuContextDescriptor::new(requirements())
        .require_format_role(format, GpuFormatRole::DepthStencil)
        .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
        .with_allowed_backends([GpuBackendFamily::Vulkan])
        .with_label(format!("R1 {format:?} native depth proof"));
    pollster::block_on(GpuContext::request(descriptor))
        .expect("observed native depth role must admit a Vulkan software context")
}

fn copy_context_for(format: GpuTextureFormat) -> GpuContext {
    let descriptor = GpuContextDescriptor::new(requirements())
        .require_format_role(format, GpuFormatRole::CopySource)
        .require_format_role(format, GpuFormatRole::CopyDestination)
        .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
        .with_allowed_backends([GpuBackendFamily::Vulkan])
        .with_label(format!("R1 {format:?} native copy proof"));
    pollster::block_on(GpuContext::request(descriptor))
        .expect("observed native copy roles must admit a Vulkan software context")
}

fn depth_texture(
    allocator: &mut GpuWorkResourceIdAllocator,
    name: &str,
    format: GpuTextureFormat,
    usages: impl IntoIterator<Item = GpuTextureUsage>,
    initialization: GpuTextureInitialization,
    width: u32,
    height: u32,
) -> GpuTextureHandle {
    let resource_label = label(name);
    allocator
        .allocate_texture_handle(
            GpuTextureDescriptor::new(
                common(name),
                GpuTextureDimension::D2,
                GpuTextureExtent::new(&resource_label, GpuTextureDimension::D2, width, height, 1)
                    .unwrap(),
                1,
                1,
                format,
                GpuTextureUsages::new(&resource_label, usages).unwrap(),
                initialization,
            )
            .unwrap(),
        )
        .unwrap()
}

fn run_native_supported_usage_realization(
    format: GpuTextureFormat,
    sampled: bool,
    depth_stencil: bool,
    copy_source: bool,
    copy_destination: bool,
) {
    let mut descriptor = GpuContextDescriptor::new(requirements())
        .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
        .with_allowed_backends([GpuBackendFamily::Vulkan])
        .with_label(format!("{format:?} native supported-usage realization"));
    let mut usages = Vec::new();

    if sampled {
        descriptor = descriptor.require_format_role(format, GpuFormatRole::Sampled);
        usages.push(GpuTextureUsage::Sampled);
    }
    if depth_stencil {
        descriptor = descriptor.require_format_role(format, GpuFormatRole::DepthStencil);
        usages.push(GpuTextureUsage::DepthStencilAttachment);
    }
    if copy_source {
        descriptor = descriptor.require_format_role(format, GpuFormatRole::CopySource);
        usages.push(GpuTextureUsage::CopySource);
    }
    if copy_destination {
        descriptor = descriptor.require_format_role(format, GpuFormatRole::CopyDestination);
        usages.push(GpuTextureUsage::CopyDestination);
    }

    if usages.is_empty() {
        println!(
            "{format:?}: SKIPPED supported-usage D2 realization (no sampled/depth/copy roles advertised)"
        );
        return;
    }

    let context = pollster::block_on(GpuContext::request(descriptor))
        .expect("advertised native depth-format roles must admit the selected format");
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let name = format!("{format:?} native supported-usage realization");
    let texture = depth_texture(
        &mut allocator,
        &name,
        format,
        usages,
        GpuTextureInitialization::Uninitialized,
        32,
        16,
    );
    let _realized = context.realize_texture(&texture).unwrap();

    if sampled {
        println!("{format:?} Sampled: EXERCISED D2 texture realization");
    } else {
        println!("{format:?} Sampled: SKIPPED (sampled role not advertised)");
    }
}

fn depth_view(
    allocator: &mut GpuWorkResourceIdAllocator,
    texture: &GpuTextureHandle,
    name: &str,
) -> GpuTextureViewHandle {
    let range = GpuTextureSubresourceRange::new(
        texture.descriptor().common().label(),
        0,
        1,
        0,
        1,
        GpuTextureAspect::DepthOnly,
    )
    .unwrap();
    allocator
        .allocate_texture_view_handle(
            GpuTextureViewDescriptor::new(
                common(name),
                texture,
                None,
                GpuTextureViewDimension::D2,
                range,
            )
            .unwrap(),
        )
        .unwrap()
}

fn add_node(builder: &mut GpuWorkFragmentBuilder, name: &str, operation: GpuWorkOperation) {
    builder
        .add_node(
            label(name),
            operation,
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::Automatic,
            provenance(name),
        )
        .unwrap();
}

fn wait_submission(context: &GpuContext, submission: &GpuSubmission, name: &str) {
    let deadline = Instant::now() + Duration::from_secs(15);
    loop {
        context.progress();
        match submission.status() {
            GpuSubmissionStatus::Completed => break,
            GpuSubmissionStatus::Failed(error) => panic!("{name} failed: {error:?}"),
            GpuSubmissionStatus::Accepted => {}
        }
        assert!(Instant::now() < deadline, "{name} timed out");
        std::thread::yield_now();
    }
}

fn clear_operation(view: GpuTextureViewHandle) -> GpuRenderOperation {
    let attachment = GpuRenderDepthStencilAttachment::new(
        view,
        GpuDepthStencilAccess::ReadWrite,
        GpuDepthAttachmentLoad::Clear(GpuDepthClearValue::new(0.5).unwrap()),
        GpuAttachmentStore::Store,
    )
    .unwrap();
    GpuRenderOperation::new([], Some(attachment), [], None).unwrap()
}

fn run_native_clear(format: GpuTextureFormat) {
    let context = depth_context_for(format);
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let name = format!("R1 native {format:?} clear");
    let texture = depth_texture(
        &mut allocator,
        &name,
        format,
        [GpuTextureUsage::DepthStencilAttachment],
        GpuTextureInitialization::Uninitialized,
        32,
        16,
    );
    let view = depth_view(&mut allocator, &texture, &format!("{name} view"));
    let realized = context.realize_texture(&texture).unwrap();
    let _realized_view = context.realize_texture_view(&view, &realized).unwrap();

    let mut builder = GpuWorkFragmentBuilder::new(label(&name), provenance(&name));
    builder.declare_resource(texture.into()).unwrap();
    builder.declare_resource(view.clone().into()).unwrap();
    add_node(
        &mut builder,
        &format!("{name} render"),
        GpuWorkOperation::Render(clear_operation(view)),
    );
    let graph = GpuPreparedWorkGraph::prepare(label(&name), [builder.finish().unwrap()]).unwrap();
    let prepared = pollster::block_on(context.prepare_submission(graph)).unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    wait_submission(&context, &submission, &name);
    println!("{format:?}: EXERCISED depth-attachment clear");
}

fn run_native_texture_copy(format: GpuTextureFormat) {
    let context = copy_context_for(format);
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let name = format!("R1 native {format:?} texture copy");
    let source = depth_texture(
        &mut allocator,
        &format!("{name} source"),
        format,
        [GpuTextureUsage::CopySource],
        GpuTextureInitialization::Zeroed,
        32,
        16,
    );
    let destination = depth_texture(
        &mut allocator,
        &format!("{name} destination"),
        format,
        [GpuTextureUsage::CopyDestination],
        GpuTextureInitialization::Uninitialized,
        32,
        16,
    );
    context.realize_texture(&source).unwrap();
    context.realize_texture(&destination).unwrap();

    let extent = GpuCopyExtent::new(32, 16, 1).unwrap();
    let source_region = GpuTextureCopyRegion::new(
        &source,
        0,
        GpuTextureOrigin::new(0, 0, 0),
        GpuTextureAspect::DepthOnly,
        extent,
    )
    .unwrap();
    let destination_region = GpuTextureCopyRegion::new(
        &destination,
        0,
        GpuTextureOrigin::new(0, 0, 0),
        GpuTextureAspect::DepthOnly,
        extent,
    )
    .unwrap();
    let copy = GpuCopyOperation::texture_to_texture(source_region, destination_region).unwrap();

    let mut builder = GpuWorkFragmentBuilder::new(label(&name), provenance(&name));
    builder.declare_resource(source.into()).unwrap();
    builder.declare_resource(destination.into()).unwrap();
    add_node(
        &mut builder,
        &format!("{name} copy"),
        GpuWorkOperation::Copy(copy),
    );
    let graph = GpuPreparedWorkGraph::prepare(label(&name), [builder.finish().unwrap()]).unwrap();
    let prepared = pollster::block_on(context.prepare_submission(graph)).unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    wait_submission(&context, &submission, &name);
    println!("{format:?} Copy: EXERCISED full-plane texture-to-texture copy");
}

fn run_depth16_linear_roundtrip(width: u32) {
    let format = GpuTextureFormat::Depth16Unorm;
    let context = copy_context_for(format);
    let height = 2;
    let name = format!("R1 native Depth16Unorm {width}x{height}");
    let expected = (0..width * height * 2)
        .map(|index| (index % 251) as u8)
        .collect::<Vec<_>>();
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let source = depth_texture(
        &mut allocator,
        &format!("{name} source"),
        format,
        [
            GpuTextureUsage::CopySource,
            GpuTextureUsage::CopyDestination,
        ],
        GpuTextureInitialization::Uninitialized,
        width,
        height,
    );
    let destination = depth_texture(
        &mut allocator,
        &format!("{name} destination"),
        format,
        [
            GpuTextureUsage::CopySource,
            GpuTextureUsage::CopyDestination,
        ],
        GpuTextureInitialization::Uninitialized,
        width,
        height,
    );
    let extent = GpuCopyExtent::new(width, height, 1).unwrap();
    let source_region = GpuTextureCopyRegion::new(
        &source,
        0,
        GpuTextureOrigin::new(0, 0, 0),
        GpuTextureAspect::DepthOnly,
        extent,
    )
    .unwrap();
    let destination_region = GpuTextureCopyRegion::new(
        &destination,
        0,
        GpuTextureOrigin::new(0, 0, 0),
        GpuTextureAspect::DepthOnly,
        extent,
    )
    .unwrap();
    let upload = GpuUploadOperation::new(
        source_region.clone().into(),
        PreparedGpuData::<TransferData>::from_pod_transfer(
            &name,
            expected.as_slice(),
            provenance(&name),
        )
        .unwrap(),
    )
    .unwrap();
    let copy =
        GpuCopyOperation::texture_to_texture(source_region, destination_region.clone()).unwrap();
    let readback_id = GpuReadbackId::allocate().unwrap();
    let readback = GpuReadbackOperation::new(destination_region.into(), readback_id).unwrap();

    let mut builder = GpuWorkFragmentBuilder::new(label(&name), provenance(&name));
    builder.declare_resource(source.into()).unwrap();
    builder.declare_resource(destination.into()).unwrap();
    add_node(
        &mut builder,
        &format!("{name} upload"),
        GpuWorkOperation::Upload(upload),
    );
    add_node(
        &mut builder,
        &format!("{name} copy"),
        GpuWorkOperation::Copy(copy),
    );
    add_node(
        &mut builder,
        &format!("{name} readback"),
        GpuWorkOperation::Readback(readback),
    );
    let graph = GpuPreparedWorkGraph::prepare(label(&name), [builder.finish().unwrap()]).unwrap();
    let prepared = pollster::block_on(context.prepare_submission(graph)).unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    let readback = submission.readback(readback_id).unwrap().clone();

    let deadline = Instant::now() + Duration::from_secs(15);
    let bytes = loop {
        context.progress();
        match readback.status() {
            GpuReadbackStatus::Ready(bytes) => break bytes,
            GpuReadbackStatus::Failed(error) => panic!("{name} readback failed: {error:?}"),
            GpuReadbackStatus::Pending => {}
        }
        assert!(Instant::now() < deadline, "{name} readback timed out");
        std::thread::yield_now();
    };
    wait_submission(&context, &submission, &name);
    assert_eq!(bytes.as_bytes(), expected.as_slice());
    assert_eq!(bytes.layout().byte_len(), expected.len() as u64);
    assert_eq!(bytes.texture_format(), Some(format));
    println!(
        "Depth16Unorm: EXERCISED {width}x{height} linear roundtrip ({} logical bytes/row)",
        width * 2
    );
}

#[test]
fn depth24plus_linear_cpu_transfer_paths_fail_closed() {
    assert_eq!(
        GpuTextureFormat::Depth24Plus.copy_block_size(GpuTextureAspect::DepthOnly),
        None
    );
}

#[test]
#[ignore = "requires a Vulkan software adapter; executed by RunenGPU native Conformance CI"]
fn native_baseline_depth_formats_execute_attachment_and_copy_contracts() {
    let census = census_context();
    let mut attachment_exercised = 0;
    for format in DEPTH_FORMATS {
        let facts = census
            .adapter_facts()
            .supported()
            .format(format)
            .expect("baseline depth format must be enumerated");
        run_native_supported_usage_realization(
            format,
            facts.sampled,
            facts.depth_stencil,
            facts.copy_source,
            facts.copy_destination,
        );

        if facts.depth_stencil {
            run_native_clear(format);
            attachment_exercised += 1;
        } else {
            println!("{format:?} DepthStencil: SKIPPED (depth-attachment role not advertised)");
        }

        if facts.copy_source && facts.copy_destination {
            run_native_texture_copy(format);
        } else {
            println!("{format:?} Copy: SKIPPED (CopySource + CopyDestination not both advertised)");
        }
    }
    assert!(
        attachment_exercised > 0,
        "baseline depth native proof NOT QUALIFIED: no depth attachment format executed"
    );

    let facts = census
        .adapter_facts()
        .supported()
        .format(GpuTextureFormat::Depth16Unorm)
        .expect("Depth16Unorm must be enumerated");
    if facts.copy_source && facts.copy_destination {
        for width in [127, 128] {
            run_depth16_linear_roundtrip(width);
        }
    } else {
        println!(
            "Depth16Unorm linear transfer: SKIPPED (CopySource + CopyDestination not both advertised)"
        );
    }
}
