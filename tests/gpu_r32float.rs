use runen_gpu::*;
use std::time::{Duration, Instant};

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

fn native_r32float_context() -> GpuContext {
    let mut requirements = GpuCapabilityRequirements::new();
    requirements
        .insert(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::Copy,
        ))
        .unwrap();
    let descriptor = GpuContextDescriptor::new(requirements)
        .require_format_role(GpuTextureFormat::R32Float, GpuFormatRole::Sampled)
        .require_format_role(GpuTextureFormat::R32Float, GpuFormatRole::CopySource)
        .require_format_role(GpuTextureFormat::R32Float, GpuFormatRole::CopyDestination)
        .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
        .with_allowed_backends([GpuBackendFamily::Vulkan])
        .with_label("R32Float native format proof");
    let context = pollster::block_on(GpuContext::request(descriptor))
        .expect("native conformance Vulkan fallback must admit sampled/copy R32Float roles");
    let facts = context
        .adapter_facts()
        .supported()
        .format(GpuTextureFormat::R32Float)
        .expect("R32Float must be present in normalized adapter format facts");
    assert!(facts.sampled);
    assert!(facts.copy_source);
    assert!(facts.copy_destination);
    context
}

fn add_operation(builder: &mut GpuWorkFragmentBuilder, name: &str, operation: GpuWorkOperation) {
    builder
        .add_node(
            label(name),
            operation,
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::TransferPreferred,
            provenance(name),
        )
        .unwrap();
}

fn copy_buffer(allocator: &mut GpuWorkResourceIdAllocator, byte_len: u64) -> GpuBufferHandle {
    let resource_label = label("R32Float transfer source buffer");
    allocator
        .allocate_buffer_handle(
            GpuBufferDescriptor::new(
                common("R32Float transfer source buffer"),
                byte_len,
                GpuBufferUsages::new(
                    &resource_label,
                    [GpuBufferUsage::CopySource, GpuBufferUsage::CopyDestination],
                )
                .unwrap(),
                GpuBufferInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap()
}

fn progress_submission_and_readback(
    context: &GpuContext,
    submission: &GpuSubmission,
    readback_id: GpuReadbackId,
) -> GpuReadbackBytes {
    let readback = submission
        .readback(readback_id)
        .expect("accepted R32Float readback must remain observable")
        .clone();
    let deadline = Instant::now() + Duration::from_secs(15);
    let bytes = loop {
        context.progress();
        match readback.status() {
            GpuReadbackStatus::Ready(bytes) => break bytes,
            GpuReadbackStatus::Failed(failure) => {
                panic!("R32Float readback failed: {failure:?}")
            }
            GpuReadbackStatus::Pending => {}
        }
        if let GpuSubmissionStatus::Failed(failure) = submission.status() {
            panic!("R32Float submission failed before readback: {failure:?}");
        }
        assert!(Instant::now() < deadline, "R32Float readback timed out");
        std::thread::yield_now();
    };
    loop {
        context.progress();
        match submission.status() {
            GpuSubmissionStatus::Completed => break,
            GpuSubmissionStatus::Failed(failure) => {
                panic!("R32Float submission failed: {failure:?}")
            }
            GpuSubmissionStatus::Accepted => {}
        }
        assert!(
            Instant::now() < deadline,
            "R32Float submission did not terminalize"
        );
        std::thread::yield_now();
    }
    bytes
}

#[test]
fn r32float_is_public_backend_neutral_and_four_bytes_per_texel() {
    let format = GpuTextureFormat::R32Float;

    assert_eq!(format.bytes_per_texel(), 4);
    assert!(!format.is_depth());
    assert!(!format.is_srgb());
}

#[test]
fn r32float_participates_in_public_texture_and_fragment_output_contracts() {
    let texture_label = label("R32Float public texture");
    let extent = GpuTextureExtent::new(&texture_label, GpuTextureDimension::D2, 16, 8, 1).unwrap();
    let descriptor = GpuTextureDescriptor::new(
        common("R32Float public texture"),
        GpuTextureDimension::D2,
        extent,
        1,
        1,
        GpuTextureFormat::R32Float,
        GpuTextureUsages::new(
            &texture_label,
            [GpuTextureUsage::Sampled, GpuTextureUsage::CopyDestination],
        )
        .unwrap(),
        GpuTextureInitialization::Uninitialized,
    )
    .unwrap();
    assert_eq!(descriptor.format(), GpuTextureFormat::R32Float);
    assert_eq!(descriptor.extent(), extent);

    let target = GpuColorTargetStateDescriptor::new(
        GpuTextureFormat::R32Float,
        GpuBlendMode::Replace,
        GpuColorWriteMask::RED,
    )
    .unwrap();
    let signature = GpuFragmentOutputStateDescriptor::new([target])
        .expected_signature(GpuEntryPointName::new("fragment_main").unwrap())
        .unwrap();
    let output = signature.locations().next().unwrap();
    assert_eq!(
        output.value_type().scalar_class(),
        GpuShaderIoScalarClass::Float
    );
    assert_eq!(output.value_type().vector_width().get(), 1);
}

#[test]
#[ignore = "requires a real Vulkan fallback adapter; executed by RunenGPU Native Conformance CI"]
fn r32float_transfer_realization_admission_and_sampled_binding_are_backend_proven() {
    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 2;
    const BYTES_PER_ROW: u32 = WIDTH * 4;

    let context = native_r32float_context();

    let mut rejection_requirements = GpuCapabilityRequirements::new();
    rejection_requirements
        .insert(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::Copy,
        ))
        .unwrap();
    let rejected_descriptor = GpuContextDescriptor::new(rejection_requirements)
        .require_format_role(GpuTextureFormat::R32Float, GpuFormatRole::DepthStencil)
        .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
        .with_allowed_backends([GpuBackendFamily::Vulkan])
        .with_label("R32Float unsupported depth-role proof");
    match pollster::block_on(GpuContext::request(rejected_descriptor)) {
        Err(error) => assert_eq!(
            error.category(),
            GpuContextRequestErrorCategory::NoAdmissibleCandidate,
            "R32Float depth-role rejection must come from normalized candidate admission"
        ),
        Ok(_) => panic!("non-depth R32Float must not admit the depth/stencil role"),
    }

    let values = (0..(WIDTH * HEIGHT))
        .map(|index| index as f32 * 0.25 - 8.0)
        .collect::<Vec<_>>();
    let expected = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    assert_eq!(
        expected.len(),
        usize::try_from(BYTES_PER_ROW * HEIGHT).unwrap()
    );

    let mut allocator = GpuWorkResourceIdAllocator::new();
    let buffer = copy_buffer(&mut allocator, expected.len() as u64);
    let texture_label = label("native R32Float transfer texture");
    let extent =
        GpuTextureExtent::new(&texture_label, GpuTextureDimension::D2, WIDTH, HEIGHT, 1).unwrap();
    let texture = allocator
        .allocate_texture_handle(
            GpuTextureDescriptor::new(
                common("native R32Float transfer texture"),
                GpuTextureDimension::D2,
                extent,
                1,
                1,
                GpuTextureFormat::R32Float,
                GpuTextureUsages::new(
                    &texture_label,
                    [
                        GpuTextureUsage::Sampled,
                        GpuTextureUsage::CopySource,
                        GpuTextureUsage::CopyDestination,
                    ],
                )
                .unwrap(),
                GpuTextureInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();

    let realized_texture = context
        .realize_texture(&texture)
        .expect("admitted R32Float texture must realize");
    let view_common = common("native R32Float sampled view");
    let view_subresources =
        GpuTextureSubresourceRange::new(view_common.label(), 0, 1, 0, 1, GpuTextureAspect::Color)
            .unwrap();
    let view = allocator
        .allocate_texture_view_handle(
            GpuTextureViewDescriptor::new(
                view_common,
                &texture,
                None,
                GpuTextureViewDimension::D2,
                view_subresources,
            )
            .unwrap(),
        )
        .unwrap();
    let realized_view = context
        .realize_texture_view(&view, &realized_texture)
        .expect("R32Float sampled view must realize");

    let binding_key = GpuBindingKey::try_new(0, 0).unwrap();
    let binding = GpuBindingDeclaration::new(
        binding_key,
        GpuShaderStages::one(GpuShaderStage::Fragment),
        GpuBindingKind::sampled_texture(
            GpuTextureSampleClass::FloatUnfilterable,
            GpuTextureViewDimension::D2,
            false,
        )
        .unwrap(),
        None,
        "r32float_texture",
        GpuBindingProvenance::new("R32Float native sampled binding proof", None).unwrap(),
    )
    .unwrap();
    let layout = GpuBindGroupLayoutDescriptor::new(0, [binding]).unwrap();
    let realized_layout = pollster::block_on(context.realize_bind_group_layout(&layout))
        .expect("unfilterable R32Float sampled layout must realize");
    let binding_value = GpuRuntimeBindingValue::new(
        binding_key,
        [GpuRuntimeBindingResource::TextureView(
            GpuRuntimeTextureViewBinding::new(view.clone()),
        )],
    )
    .unwrap();
    let realized_bind_group =
        pollster::block_on(context.realize_bind_group(&realized_layout, [binding_value]))
            .expect("R32Float sampled texture must bind without a filterability requirement");

    let buffer_region =
        GpuBufferRegion::new(&buffer, GpuBufferRange::whole(&buffer).unwrap()).unwrap();
    let upload = GpuUploadOperation::new(
        buffer_region.into(),
        PreparedGpuData::<TransferData>::from_pod_transfer(
            "R32Float transfer source bytes",
            expected.as_slice(),
            provenance("R32Float transfer source bytes"),
        )
        .unwrap(),
    )
    .unwrap();
    let texture_region = GpuTextureCopyRegion::new(
        &texture,
        0,
        GpuTextureOrigin::new(0, 0, 0),
        GpuTextureAspect::Color,
        GpuCopyExtent::new(WIDTH, HEIGHT, 1).unwrap(),
    )
    .unwrap();
    let source_layout = GpuBufferTextureLayout::new(&buffer, 0, BYTES_PER_ROW, HEIGHT).unwrap();
    let copy = GpuCopyOperation::buffer_to_texture(source_layout, texture_region.clone()).unwrap();
    let readback_id = GpuReadbackId::allocate().unwrap();
    let readback = GpuReadbackOperation::new(texture_region.into(), readback_id).unwrap();
    let mut builder = GpuWorkFragmentBuilder::new(
        label("native R32Float transfer proof"),
        provenance("native R32Float transfer proof"),
    );
    builder.declare_resource(buffer.into()).unwrap();
    builder.declare_resource(texture.into()).unwrap();
    add_operation(
        &mut builder,
        "upload R32Float source bytes",
        GpuWorkOperation::Upload(upload),
    );
    add_operation(
        &mut builder,
        "copy bytes into R32Float texture",
        GpuWorkOperation::Copy(copy),
    );
    add_operation(
        &mut builder,
        "read R32Float texture bytes",
        GpuWorkOperation::Readback(readback),
    );
    let graph = GpuPreparedWorkGraph::prepare(
        label("native R32Float transfer graph"),
        [builder.finish().unwrap()],
    )
    .unwrap();

    let prepared = pollster::block_on(context.prepare_submission(graph)).unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    let bytes = progress_submission_and_readback(&context, &submission, readback_id);
    assert_eq!(bytes.as_bytes(), expected.as_slice());
    assert_eq!(bytes.layout().byte_len(), expected.len() as u64);
    assert_eq!(bytes.texture_format(), Some(GpuTextureFormat::R32Float));

    drop(realized_bind_group);
    drop(realized_view);
    drop(realized_texture);
}

#[test]
#[ignore = "requires a real Vulkan fallback adapter; executed by RunenGPU Native Conformance CI"]
fn new_32bit_formats_round_trip_when_adapter_reports_copy_roles() {
    const WIDTH: u32 = 64;
    const HEIGHT: u32 = 2;

    let context = native_r32float_context();
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let mut exercised = 0;
    for format in [
        GpuTextureFormat::R32Sint,
        GpuTextureFormat::Rg32Uint,
        GpuTextureFormat::Rg32Sint,
        GpuTextureFormat::Rg32Float,
        GpuTextureFormat::Rgba32Uint,
        GpuTextureFormat::Rgba32Sint,
        GpuTextureFormat::Rgba32Float,
    ] {
        let facts = context
            .adapter_facts()
            .supported()
            .format(format)
            .expect("every normalized 32-bit format must have enumerated adapter facts");
        if !facts.copy_source || !facts.copy_destination {
            println!("{format:?}: copy roles not both advertised; native round-trip skipped");
            continue;
        }

        let bytes_per_row = WIDTH * format.bytes_per_texel();
        let expected = (0..bytes_per_row * HEIGHT)
            .map(|index| (index % 251) as u8)
            .collect::<Vec<_>>();
        let buffer = copy_buffer(&mut allocator, expected.len() as u64);
        let texture_name = format!("native {format:?} transfer texture");
        let texture_label = label(&texture_name);
        let extent =
            GpuTextureExtent::new(&texture_label, GpuTextureDimension::D2, WIDTH, HEIGHT, 1)
                .unwrap();
        let texture = allocator
            .allocate_texture_handle(
                GpuTextureDescriptor::new(
                    common(&texture_name),
                    GpuTextureDimension::D2,
                    extent,
                    1,
                    1,
                    format,
                    GpuTextureUsages::new(
                        &texture_label,
                        [GpuTextureUsage::CopySource, GpuTextureUsage::CopyDestination],
                    )
                    .unwrap(),
                    GpuTextureInitialization::Uninitialized,
                )
                .unwrap(),
            )
            .unwrap();
        let realized_texture = context
            .realize_texture(&texture)
            .expect("adapter-admitted 32-bit copy texture must realize");

        let source_name = format!("native {format:?} upload bytes");
        let buffer_region =
            GpuBufferRegion::new(&buffer, GpuBufferRange::whole(&buffer).unwrap()).unwrap();
        let upload = GpuUploadOperation::new(
            buffer_region.into(),
            PreparedGpuData::<TransferData>::from_pod_transfer(
                &source_name,
                expected.as_slice(),
                provenance(&source_name),
            )
            .unwrap(),
        )
        .unwrap();
        let texture_region = GpuTextureCopyRegion::new(
            &texture,
            0,
            GpuTextureOrigin::new(0, 0, 0),
            GpuTextureAspect::Color,
            GpuCopyExtent::new(WIDTH, HEIGHT, 1).unwrap(),
        )
        .unwrap();
        let layout = GpuBufferTextureLayout::new(&buffer, 0, bytes_per_row, HEIGHT).unwrap();
        let copy = GpuCopyOperation::buffer_to_texture(layout, texture_region.clone()).unwrap();
        let readback_id = GpuReadbackId::allocate().unwrap();
        let readback = GpuReadbackOperation::new(texture_region.into(), readback_id).unwrap();
        let graph_name = format!("native {format:?} copy round-trip");
        let mut builder = GpuWorkFragmentBuilder::new(label(&graph_name), provenance(&graph_name));
        builder.declare_resource(buffer.into()).unwrap();
        builder.declare_resource(texture.into()).unwrap();
        add_operation(
            &mut builder,
            &format!("upload {format:?} bytes"),
            GpuWorkOperation::Upload(upload),
        );
        add_operation(
            &mut builder,
            &format!("copy bytes into {format:?} texture"),
            GpuWorkOperation::Copy(copy),
        );
        add_operation(
            &mut builder,
            &format!("read {format:?} texture bytes"),
            GpuWorkOperation::Readback(readback),
        );
        let graph = GpuPreparedWorkGraph::prepare(label(&graph_name), [builder.finish().unwrap()])
            .unwrap();
        let prepared = pollster::block_on(context.prepare_submission(graph)).unwrap();
        let submission = context.submit_prepared(prepared).unwrap();
        let bytes = progress_submission_and_readback(&context, &submission, readback_id);
        assert_eq!(bytes.as_bytes(), expected.as_slice(), "{format:?} round-trip bytes");
        assert_eq!(bytes.layout().byte_len(), expected.len() as u64);
        assert_eq!(bytes.texture_format(), Some(format));
        exercised += 1;
        println!("{format:?}: copy round-trip PASS ({bytes_per_row} bytes/row)");
        drop(realized_texture);
    }
    println!("observed new 32-bit format copy round-trips: {exercised}/7");
}
