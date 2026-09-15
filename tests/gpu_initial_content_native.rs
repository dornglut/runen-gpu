use runen_gpu::*;
use std::num::NonZeroUsize;
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

fn native_copy_context() -> GpuContext {
    native_copy_context_with_policies(
        GpuRealizationPolicies::default(),
        GpuExecutionPolicy::default(),
    )
}

fn native_copy_context_with_policies(
    realization_policies: GpuRealizationPolicies,
    execution_policy: GpuExecutionPolicy,
) -> GpuContext {
    let mut requirements = GpuCapabilityRequirements::new();
    requirements
        .insert(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::Copy,
        ))
        .unwrap();
    let descriptor = GpuContextDescriptor::new(requirements)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::CopySource)
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::CopyDestination)
        .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
        .with_allowed_backends([GpuBackendFamily::Vulkan])
        .with_label("G5R native prepared initial-content proof");
    let context = pollster::block_on(GpuContext::request_with_policies(
        descriptor,
        realization_policies,
        execution_policy,
    ))
    .expect("native conformance environment must provide a Vulkan fallback adapter");
    assert_eq!(context.adapter_facts().backend(), GpuBackendFamily::Vulkan);
    assert_eq!(
        context.adapter_facts().fallback(),
        GpuFallbackStatus::ConfirmedFallback
    );
    context
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

fn prepared_buffer(allocator: &mut GpuWorkResourceIdAllocator, initial: &[u8]) -> GpuBufferHandle {
    let resource_label = label("native G5R prepared buffer");
    allocator
        .allocate_buffer_handle(
            GpuBufferDescriptor::new(
                common("native G5R prepared buffer"),
                initial.len() as u64,
                GpuBufferUsages::new(
                    &resource_label,
                    [GpuBufferUsage::CopySource, GpuBufferUsage::CopyDestination],
                )
                .unwrap(),
                GpuBufferInitialization::Prepared(
                    PreparedGpuData::<TransferData>::from_pod_transfer(
                        "native G5R initial bytes",
                        initial,
                        provenance("native G5R initial bytes"),
                    )
                    .unwrap(),
                ),
            )
            .unwrap(),
        )
        .unwrap()
}

fn readback_graph(buffer: &GpuBufferHandle) -> (GpuPreparedWorkGraph, GpuReadbackId) {
    let region = GpuBufferRegion::new(buffer, GpuBufferRange::whole(buffer).unwrap()).unwrap();
    let id = GpuReadbackId::allocate().unwrap();
    let readback = GpuReadbackOperation::new(region.into(), id).unwrap();
    let mut builder = GpuWorkFragmentBuilder::new(
        label("native G5R prepared readback"),
        provenance("native G5R prepared readback"),
    );
    builder.declare_resource(buffer.clone().into()).unwrap();
    add_operation(
        &mut builder,
        "read prepared bytes",
        GpuWorkOperation::Readback(readback),
    );
    (
        GpuPreparedWorkGraph::prepare(
            label("native G5R prepared readback graph"),
            [builder.finish().unwrap()],
        )
        .unwrap(),
        id,
    )
}

fn upload_and_readback_graph(
    buffer: &GpuBufferHandle,
    replacement: &[u8],
) -> (GpuPreparedWorkGraph, GpuReadbackId) {
    let region = GpuBufferRegion::new(buffer, GpuBufferRange::whole(buffer).unwrap()).unwrap();
    let upload = GpuUploadOperation::new(
        region.clone().into(),
        PreparedGpuData::<TransferData>::from_pod_transfer(
            "native G5R replacement bytes",
            replacement,
            provenance("native G5R replacement bytes"),
        )
        .unwrap(),
    )
    .unwrap();
    let id = GpuReadbackId::allocate().unwrap();
    let readback = GpuReadbackOperation::new(region.into(), id).unwrap();
    let mut builder = GpuWorkFragmentBuilder::new(
        label("native G5R explicit replacement"),
        provenance("native G5R explicit replacement"),
    );
    builder.declare_resource(buffer.clone().into()).unwrap();
    add_operation(
        &mut builder,
        "replace prepared bytes",
        GpuWorkOperation::Upload(upload),
    );
    add_operation(
        &mut builder,
        "read replaced bytes",
        GpuWorkOperation::Readback(readback),
    );
    (
        GpuPreparedWorkGraph::prepare(
            label("native G5R explicit replacement graph"),
            [builder.finish().unwrap()],
        )
        .unwrap(),
        id,
    )
}

fn progress_submission_and_readback(
    context: &GpuContext,
    submission: &GpuSubmission,
    readback_id: GpuReadbackId,
) -> GpuReadbackBytes {
    let readback = submission
        .readback(readback_id)
        .expect("accepted G5R readback must remain observable")
        .clone();
    let deadline = Instant::now() + Duration::from_secs(15);
    let bytes = loop {
        context.progress();
        match readback.status() {
            GpuReadbackStatus::Ready(bytes) => break bytes,
            GpuReadbackStatus::Failed(failure) => panic!("G5R readback failed: {failure:?}"),
            GpuReadbackStatus::Pending => {}
        }
        if let GpuSubmissionStatus::Failed(failure) = submission.status() {
            panic!("G5R submission failed before readback: {failure:?}");
        }
        assert!(Instant::now() < deadline, "G5R readback timed out");
        std::thread::yield_now();
    };
    loop {
        context.progress();
        match submission.status() {
            GpuSubmissionStatus::Completed => break,
            GpuSubmissionStatus::Failed(failure) => panic!("G5R submission failed: {failure:?}"),
            GpuSubmissionStatus::Accepted => {}
        }
        assert!(
            Instant::now() < deadline,
            "G5R submission did not terminalize"
        );
        std::thread::yield_now();
    }
    bytes
}

fn submit_and_readback(
    context: &GpuContext,
    graph: GpuPreparedWorkGraph,
    readback_id: GpuReadbackId,
) -> GpuReadbackBytes {
    let prepared = pollster::block_on(context.prepare_submission(graph)).unwrap();
    let submission = context.submit_prepared(prepared).unwrap();
    progress_submission_and_readback(context, &submission, readback_id)
}

#[test]
#[ignore = "requires a real Vulkan fallback adapter; executed by RunenGPU Native Conformance CI"]
fn prepared_buffer_materializes_once_per_physical_realization_and_explicit_upload_wins() {
    let context = native_copy_context();
    let initial = (0_u8..64).collect::<Vec<_>>();
    let replacement = (0_u8..64).map(|value| 255 - value).collect::<Vec<_>>();
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let buffer = prepared_buffer(&mut allocator, &initial);

    let (first_graph, first_id) = readback_graph(&buffer);
    let first = submit_and_readback(&context, first_graph, first_id);
    assert_eq!(first.as_bytes(), initial.as_slice());

    let (replacement_graph, replacement_id) = upload_and_readback_graph(&buffer, &replacement);
    let replaced = submit_and_readback(&context, replacement_graph, replacement_id);
    assert_eq!(
        replaced.as_bytes(),
        replacement.as_slice(),
        "an explicit upload on the same realization must remain unconditional and must not be overwritten by Prepared metadata"
    );

    let (later_graph, later_id) = readback_graph(&buffer);
    let later = submit_and_readback(&context, later_graph, later_id);
    assert_eq!(
        later.as_bytes(),
        replacement.as_slice(),
        "later submissions on the same physical realization must not replay Prepared initial content"
    );
}

#[test]
#[ignore = "requires a real Vulkan fallback adapter; executed by RunenGPU Native Conformance CI"]
fn prepared_submissions_select_seed_once_during_ordered_acceptance() {
    let execution_policy = GpuExecutionPolicy::new(
        NonZeroUsize::new(4).unwrap(),
        NonZeroUsize::new(2).unwrap(),
        64,
        128,
        2,
    );
    let context =
        native_copy_context_with_policies(GpuRealizationPolicies::default(), execution_policy);
    let initial = (0_u8..64).collect::<Vec<_>>();
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let buffer = prepared_buffer(&mut allocator, &initial);
    let (first_graph, first_id) = readback_graph(&buffer);
    let (second_graph, second_id) = readback_graph(&buffer);

    let first_prepared = pollster::block_on(context.prepare_submission(first_graph)).unwrap();
    let second_prepared = pollster::block_on(context.prepare_submission(second_graph)).unwrap();
    assert_eq!(context.execution_stats().prepared_submissions(), 2);
    assert_eq!(context.execution_stats().upload_bytes_in_flight(), 0);

    let first = context.submit_prepared(first_prepared).unwrap();
    assert_eq!(
        context.execution_stats().upload_bytes_in_flight(),
        64,
        "the first ordered acceptance must charge exactly one conditional seed"
    );
    let second = context
        .submit_prepared(second_prepared)
        .expect("the second already-prepared submission must observe the queued seed and skip duplicate pressure");
    assert_eq!(
        context.execution_stats().upload_bytes_in_flight(),
        64,
        "the second ordered acceptance must not charge the same physical seed twice"
    );

    let first_bytes = progress_submission_and_readback(&context, &first, first_id);
    let second_bytes = progress_submission_and_readback(&context, &second, second_id);
    assert_eq!(first_bytes.as_bytes(), initial.as_slice());
    assert_eq!(second_bytes.as_bytes(), initial.as_slice());
}

#[test]
#[ignore = "requires a real Vulkan fallback adapter; executed by RunenGPU Native Conformance CI"]
fn conditional_seed_pressure_is_charged_only_at_acceptance_and_rejection_keeps_seed_required() {
    let execution_policy = GpuExecutionPolicy::new(
        NonZeroUsize::new(2).unwrap(),
        NonZeroUsize::new(1).unwrap(),
        32,
        64,
        1,
    );
    let context =
        native_copy_context_with_policies(GpuRealizationPolicies::default(), execution_policy);
    let initial = (0_u8..64).collect::<Vec<_>>();
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let buffer = prepared_buffer(&mut allocator, &initial);

    for attempt in 0..2 {
        let (graph, _) = readback_graph(&buffer);
        let prepared = pollster::block_on(context.prepare_submission(graph)).expect(
            "conditional Prepared bytes must not consume upload pressure during preparation",
        );
        assert_eq!(context.execution_stats().prepared_submissions(), 1);
        assert_eq!(context.execution_stats().upload_bytes_in_flight(), 0);

        let rejected = context.submit_prepared(prepared).expect_err(
            "ordered acceptance must reject when the required seed exceeds upload pressure",
        );
        assert_eq!(
            rejected.reason().kind(),
            GpuSubmissionRejectionKind::UploadBytesInFlightExceeded,
            "attempt {attempt} must reject through the existing upload-pressure authority"
        );
        drop(rejected);
        assert_eq!(context.execution_stats().prepared_submissions(), 0);
        assert_eq!(context.execution_stats().upload_bytes_in_flight(), 0);
    }
}

#[test]
#[ignore = "requires a real Vulkan fallback adapter; executed by RunenGPU Native Conformance CI"]
fn recreated_physical_realization_receives_prepared_seed_again() {
    let realization_policies = GpuRealizationPolicies::new(
        GpuResourceRealizationPolicy::new(NonZeroUsize::new(1).unwrap()),
        Default::default(),
    );
    let context =
        native_copy_context_with_policies(realization_policies, GpuExecutionPolicy::default());
    let initial = (0_u8..64).collect::<Vec<_>>();
    let replacement = (0_u8..64).map(|value| 255 - value).collect::<Vec<_>>();
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let buffer = prepared_buffer(&mut allocator, &initial);

    let (first_graph, first_id) = readback_graph(&buffer);
    assert_eq!(
        submit_and_readback(&context, first_graph, first_id).as_bytes(),
        initial.as_slice()
    );
    let (replacement_graph, replacement_id) = upload_and_readback_graph(&buffer, &replacement);
    assert_eq!(
        submit_and_readback(&context, replacement_graph, replacement_id).as_bytes(),
        replacement.as_slice()
    );

    let eviction_label = label("native G5R eviction buffer");
    let eviction = allocator
        .allocate_buffer_handle(
            GpuBufferDescriptor::new(
                common("native G5R eviction buffer"),
                64,
                GpuBufferUsages::new(&eviction_label, [GpuBufferUsage::CopyDestination]).unwrap(),
                GpuBufferInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();
    let realized_eviction = context
        .realize_buffer(&eviction)
        .expect("one-record pressure must reclaim the now-unretained prepared realization");
    assert_eq!(context.resource_realization_stats().retained_records(), 1);
    drop(realized_eviction);

    let (recreated_graph, recreated_id) = readback_graph(&buffer);
    let recreated = submit_and_readback(&context, recreated_graph, recreated_id);
    assert_eq!(
        recreated.as_bytes(),
        initial.as_slice(),
        "a recreated physical record for the same logical identity must receive its Prepared seed exactly once"
    );
}

#[test]
#[ignore = "requires a real Vulkan fallback adapter; executed by RunenGPU Native Conformance CI"]
fn padded_prepared_texture_materializes_through_canonical_texture_upload() {
    const WIDTH: u32 = 3;
    const HEIGHT: u32 = 2;
    const LAYERS: u32 = 2;
    const BYTES_PER_ROW: u32 = 16;
    const ROWS_PER_IMAGE: u32 = 3;

    let context = native_copy_context();
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let texture_label = label("native G5R prepared texture");
    let extent = GpuTextureExtent::new(
        &texture_label,
        GpuTextureDimension::D2,
        WIDTH,
        HEIGHT,
        LAYERS,
    )
    .unwrap();
    let source_len = usize::try_from(BYTES_PER_ROW * ROWS_PER_IMAGE * LAYERS).unwrap();
    let logical_row = usize::try_from(WIDTH * 4).unwrap();
    let mut source = vec![0xD7_u8; source_len];
    let mut expected = Vec::new();
    for image in 0..LAYERS {
        for row in 0..HEIGHT {
            let row_bytes = (0..logical_row)
                .map(|column| {
                    u8::try_from((image * 79 + row * 31 + u32::try_from(column).unwrap() * 7) % 251)
                        .unwrap()
                })
                .collect::<Vec<_>>();
            let start =
                usize::try_from(image * BYTES_PER_ROW * ROWS_PER_IMAGE + row * BYTES_PER_ROW)
                    .unwrap();
            source[start..start + logical_row].copy_from_slice(&row_bytes);
            expected.extend_from_slice(&row_bytes);
        }
    }
    let prepared_data = GpuPreparedTextureData::new(
        &texture_label,
        PreparedGpuData::<TransferData>::from_pod_transfer(
            "native G5R padded prepared texture bytes",
            source.as_slice(),
            provenance("native G5R padded prepared texture bytes"),
        )
        .unwrap(),
        GpuTextureFormat::Rgba8Unorm,
        extent,
        BYTES_PER_ROW,
        ROWS_PER_IMAGE,
    )
    .unwrap();
    let texture = allocator
        .allocate_texture_handle(
            GpuTextureDescriptor::new(
                common("native G5R prepared texture"),
                GpuTextureDimension::D2,
                extent,
                1,
                1,
                GpuTextureFormat::Rgba8Unorm,
                GpuTextureUsages::new(
                    &texture_label,
                    [
                        GpuTextureUsage::CopySource,
                        GpuTextureUsage::CopyDestination,
                    ],
                )
                .unwrap(),
                GpuTextureInitialization::Prepared(prepared_data),
            )
            .unwrap(),
        )
        .unwrap();
    let region = GpuTextureCopyRegion::new(
        &texture,
        0,
        GpuTextureOrigin::new(0, 0, 0),
        GpuTextureAspect::Color,
        GpuCopyExtent::new(WIDTH, HEIGHT, LAYERS).unwrap(),
    )
    .unwrap();
    let readback_id = GpuReadbackId::allocate().unwrap();
    let readback = GpuReadbackOperation::new(region.into(), readback_id).unwrap();
    let mut builder = GpuWorkFragmentBuilder::new(
        label("native G5R prepared texture readback"),
        provenance("native G5R prepared texture readback"),
    );
    builder.declare_resource(texture.into()).unwrap();
    add_operation(
        &mut builder,
        "read prepared texture bytes",
        GpuWorkOperation::Readback(readback),
    );
    let graph = GpuPreparedWorkGraph::prepare(
        label("native G5R prepared texture graph"),
        [builder.finish().unwrap()],
    )
    .unwrap();

    let bytes = submit_and_readback(&context, graph, readback_id);
    assert_eq!(bytes.as_bytes(), expected.as_slice());
    assert_eq!(bytes.layout().byte_len(), expected.len() as u64);
    assert_eq!(bytes.texture_format(), Some(GpuTextureFormat::Rgba8Unorm));
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
    assert_eq!(expected.len(), usize::try_from(BYTES_PER_ROW * HEIGHT).unwrap());

    let mut allocator = GpuWorkResourceIdAllocator::new();
    let buffer = prepared_buffer(&mut allocator, &expected);
    let texture_label = label("native R32Float transfer texture");
    let extent = GpuTextureExtent::new(
        &texture_label,
        GpuTextureDimension::D2,
        WIDTH,
        HEIGHT,
        1,
    )
    .unwrap();
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
    let view_subresources = GpuTextureSubresourceRange::new(
        view_common.label(),
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
                view_common,
                &texture,
                None,
                GpuTextureDimension::D2,
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
            GpuRuntimeTextureViewBinding::new(view.clone(), GpuTextureViewDimension::D2),
        )],
    )
    .unwrap();
    let realized_bind_group = pollster::block_on(
        context.realize_bind_group(&realized_layout, [binding_value]),
    )
    .expect("R32Float sampled texture must bind without a filterability requirement");
    assert_eq!(realized_bind_group.layout_descriptor(), &layout);

    let region = GpuTextureCopyRegion::new(
        &texture,
        0,
        GpuTextureOrigin::new(0, 0, 0),
        GpuTextureAspect::Color,
        GpuCopyExtent::new(WIDTH, HEIGHT, 1).unwrap(),
    )
    .unwrap();
    let source_layout = GpuBufferTextureLayout::new(&buffer, 0, BYTES_PER_ROW, HEIGHT).unwrap();
    let copy = GpuCopyOperation::buffer_to_texture(source_layout, region.clone()).unwrap();
    let readback_id = GpuReadbackId::allocate().unwrap();
    let readback = GpuReadbackOperation::new(region.into(), readback_id).unwrap();
    let mut builder = GpuWorkFragmentBuilder::new(
        label("native R32Float transfer proof"),
        provenance("native R32Float transfer proof"),
    );
    builder.declare_resource(buffer.into()).unwrap();
    builder.declare_resource(texture.into()).unwrap();
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

    let bytes = submit_and_readback(&context, graph, readback_id);
    assert_eq!(bytes.as_bytes(), expected.as_slice());
    assert_eq!(bytes.layout().byte_len(), expected.len() as u64);
    assert_eq!(bytes.texture_format(), Some(GpuTextureFormat::R32Float));

    drop(realized_bind_group);
    drop(realized_view);
    drop(realized_texture);
}
