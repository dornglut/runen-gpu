use runen_gpu::*;
use std::time::{Duration, Instant};

const R8_NEW_FORMATS: [(GpuTextureFormat, GpuShaderIoScalarClass); 3] = [
    (GpuTextureFormat::R8Snorm, GpuShaderIoScalarClass::Float),
    (GpuTextureFormat::R8Uint, GpuShaderIoScalarClass::Uint),
    (GpuTextureFormat::R8Sint, GpuShaderIoScalarClass::Sint),
];

const RG8_FORMATS: [(GpuTextureFormat, GpuShaderIoScalarClass); 4] = [
    (GpuTextureFormat::Rg8Unorm, GpuShaderIoScalarClass::Float),
    (GpuTextureFormat::Rg8Snorm, GpuShaderIoScalarClass::Float),
    (GpuTextureFormat::Rg8Uint, GpuShaderIoScalarClass::Uint),
    (GpuTextureFormat::Rg8Sint, GpuShaderIoScalarClass::Sint),
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

fn test_limits() -> GpuLimits {
    GpuLimits::new(
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        256 * 1024 * 1024,
        8192,
        2048,
        256,
        16,
        2048,
    )
    .unwrap()
}

fn assert_public_metadata(
    formats: &[(GpuTextureFormat, GpuShaderIoScalarClass)],
    copy_block_bytes: u32,
) {
    for (format, _) in formats.iter().copied() {
        assert_eq!(format.block_dimensions(), (1, 1));
        assert_eq!(
            format.copy_block_size(GpuTextureAspect::All),
            Some(copy_block_bytes)
        );
        assert_eq!(
            format.copy_block_size(GpuTextureAspect::Color),
            Some(copy_block_bytes)
        );
        assert_eq!(format.copy_block_size(GpuTextureAspect::DepthOnly), None);
        assert!(!format.is_depth());
        assert!(!format.is_srgb());
        let normalized = GpuCapabilities::from_normalized_facts(
            [],
            test_limits(),
            [(format, GpuTextureFormatCapabilities::none())],
        );
        let facts = normalized.format(format).unwrap();
        assert_eq!(facts.block_dimensions, Some((1, 1)));
        assert_eq!(facts.block_copy_size, Some(copy_block_bytes));
        assert!(!facts.sampled);
        assert!(!facts.filterable);
        assert!(!facts.storage_read);
        assert!(!facts.storage_write);
        assert!(!facts.color_attachment);
        assert!(!facts.copy_source);
        assert!(!facts.copy_destination);
    }
}

#[test]
fn r8_new_public_metadata_and_copy_geometry_are_exact() {
    assert_public_metadata(&R8_NEW_FORMATS, 1);
}

#[test]
fn rg8_public_metadata_and_copy_geometry_are_exact() {
    assert_public_metadata(&RG8_FORMATS, 2);
}

fn assert_structural_normalization_preserves_observed_roles(
    formats: &[(GpuTextureFormat, GpuShaderIoScalarClass)],
    copy_block_bytes: u32,
) {
    let supplied = GpuTextureFormatCapabilities {
        sampled: true,
        filterable: false,
        storage_read: false,
        storage_write: true,
        color_attachment: false,
        depth_stencil: false,
        copy_source: true,
        copy_destination: false,
        block_dimensions: Some((17, 19)),
        block_copy_size: Some(999),
    };
    for (format, _) in formats.iter().copied() {
        let normalized =
            GpuCapabilities::from_normalized_facts([], test_limits(), [(format, supplied)]);
        let facts = normalized.format(format).unwrap();
        assert_eq!(facts.block_dimensions, Some((1, 1)));
        assert_eq!(facts.block_copy_size, Some(copy_block_bytes));
        assert_eq!(facts.sampled, supplied.sampled);
        assert_eq!(facts.filterable, supplied.filterable);
        assert_eq!(facts.storage_read, supplied.storage_read);
        assert_eq!(facts.storage_write, supplied.storage_write);
        assert_eq!(facts.color_attachment, supplied.color_attachment);
        assert_eq!(facts.depth_stencil, supplied.depth_stencil);
        assert_eq!(facts.copy_source, supplied.copy_source);
        assert_eq!(facts.copy_destination, supplied.copy_destination);
    }
}

#[test]
fn r8_new_structural_normalization_preserves_observed_roles() {
    assert_structural_normalization_preserves_observed_roles(&R8_NEW_FORMATS, 1);
}

#[test]
fn rg8_structural_normalization_preserves_observed_roles() {
    assert_structural_normalization_preserves_observed_roles(&RG8_FORMATS, 2);
}

fn admitted_sampled_class(
    scalar: &str,
    float_policy: Option<GpuTextureSampleClass>,
) -> GpuTextureSampleClass {
    let source_text = format!(
        "@group(0) @binding(0) var image: texture_2d<{scalar}>;\n\
         @compute @workgroup_size(1) fn inspect() {{\n\
             let dimensions = textureDimensions(image);\n\
         }}\n"
    );
    let owner = GpuProgramSourceOwnerId::allocate().unwrap();
    let identity = GpuProgramSourceIdentity::new(
        owner,
        GpuProgramSourceKey::new("r1.r-rg8.sampled").unwrap(),
        GpuProgramSourceRevision::try_from_raw(1).unwrap(),
    );
    let mut registry = GpuProgramSourceRegistry::new(2, 16 * 1024).unwrap();
    let source = registry
        .admit_wgsl(
            identity,
            &source_text,
            GpuProgramSourceProvenance::new("r1-r-rg8-format-test", None).unwrap(),
        )
        .unwrap();
    let key = GpuBindingKey::try_new(0, 0).unwrap();
    let refinements = float_policy
        .into_iter()
        .map(|class| GpuBindingLayoutRefinement::new(key).with_texture_sample_class(class));
    let program = GpuProgramDescriptor::new(
        source,
        [GpuEntryPointName::new("inspect").unwrap()],
        refinements,
    )
    .unwrap();
    program
        .interface()
        .binding(key)
        .unwrap()
        .kind()
        .texture_sample_class()
        .unwrap()
}

#[test]
fn canonical_wgsl_derives_r_rg8_sampled_scalar_classes_without_storage_formats() {
    assert_eq!(
        admitted_sampled_class("f32", Some(GpuTextureSampleClass::FloatUnfilterable)),
        GpuTextureSampleClass::FloatUnfilterable
    );
    assert_eq!(
        admitted_sampled_class("u32", None),
        GpuTextureSampleClass::Uint
    );
    assert_eq!(
        admitted_sampled_class("i32", None),
        GpuTextureSampleClass::Sint
    );
}

#[test]
fn r_rg8_storage_texels_fail_closed_without_normalized_tier1() {
    for storage_format in [
        "r8unorm", "r8snorm", "r8uint", "r8sint", "rg8unorm", "rg8snorm", "rg8uint", "rg8sint",
    ] {
        let source_text = format!(
            "@group(0) @binding(0) var image: texture_storage_2d<{storage_format}, write>;\n\
             @compute @workgroup_size(1) fn inspect() {{\n\
                 let dimensions = textureDimensions(image);\n\
             }}\n"
        );
        let owner = GpuProgramSourceOwnerId::allocate().unwrap();
        let identity = GpuProgramSourceIdentity::new(
            owner,
            GpuProgramSourceKey::new("r1.r-rg8.storage-tier").unwrap(),
            GpuProgramSourceRevision::try_from_raw(1).unwrap(),
        );
        let mut registry = GpuProgramSourceRegistry::new(2, 16 * 1024).unwrap();
        let source = registry
            .admit_wgsl(
                identity,
                &source_text,
                GpuProgramSourceProvenance::new("r1-r-rg8-storage-tier-test", None).unwrap(),
            )
            .unwrap();
        let error = GpuProgramDescriptor::new(
            source,
            [GpuEntryPointName::new("inspect").unwrap()],
            std::iter::empty::<GpuBindingLayoutRefinement>(),
        )
        .expect_err("R/RG8 storage texels must not be admitted without normalized tier1 authority");
        if storage_format == "r8unorm" {
            assert!(error.to_string().contains("texture_formats_tier1"));
        }
    }
}

fn assert_fragment_io_and_integer_blending_follow_scalar_class(
    formats: &[(GpuTextureFormat, GpuShaderIoScalarClass)],
    width: u8,
) {
    for (format, class) in formats.iter().copied() {
        let target =
            GpuColorTargetStateDescriptor::new(format, None, GpuColorWriteMask::ALL).unwrap();
        let signature = GpuFragmentOutputStateDescriptor::new([target])
            .expected_signature(GpuEntryPointName::new("fragment_main").unwrap())
            .unwrap();
        let output = signature.locations().next().unwrap();
        assert_eq!(output.value_type().scalar_class(), class);
        assert_eq!(output.value_type().vector_width().get(), width);
        assert!(!target.has_blendable_alpha_channel());
        if matches!(
            class,
            GpuShaderIoScalarClass::Uint | GpuShaderIoScalarClass::Sint
        ) {
            assert!(
                GpuColorTargetStateDescriptor::new(
                    format,
                    Some(GpuBlendState::new(
                        GpuBlendComponent::new(
                            GpuBlendFactor::SrcAlpha,
                            GpuBlendFactor::OneMinusSrcAlpha,
                            GpuBlendOperation::Add,
                        ),
                        GpuBlendComponent::new(
                            GpuBlendFactor::One,
                            GpuBlendFactor::OneMinusSrcAlpha,
                            GpuBlendOperation::Add,
                        ),
                    )),
                    GpuColorWriteMask::ALL
                )
                .is_err()
            );
        }
    }
}

#[test]
fn r8_new_fragment_io_and_integer_blending_follow_scalar_class() {
    assert_fragment_io_and_integer_blending_follow_scalar_class(&R8_NEW_FORMATS, 1);
}

#[test]
fn rg8_fragment_io_and_integer_blending_follow_scalar_class() {
    assert_fragment_io_and_integer_blending_follow_scalar_class(&RG8_FORMATS, 2);
}

fn assert_prepared_texture_rows(
    formats: &[(GpuTextureFormat, GpuShaderIoScalarClass)],
    copy_block_bytes: u32,
    family: &str,
) {
    for (format, _) in formats.iter().copied() {
        let name = format!("prepared {family} {format:?}");
        let texture_label = label(&name);
        let extent =
            GpuTextureExtent::new(&texture_label, GpuTextureDimension::D2, 3, 2, 1).unwrap();
        let row_bytes = 3 * copy_block_bytes;
        let byte_len = (row_bytes * 2) as usize;
        let data = PreparedGpuData::<TransferData>::from_pod_transfer(
            &name,
            vec![0_u8; byte_len].as_slice(),
            provenance(&name),
        )
        .unwrap();
        let prepared =
            GpuPreparedTextureData::new(&texture_label, data, format, extent, row_bytes, 0)
                .unwrap();
        assert_eq!(prepared.bytes_per_row(), row_bytes);
        assert_eq!(prepared.data().layout().byte_len(), byte_len as u64);
        let descriptor = GpuTextureDescriptor::new(
            common(&name),
            GpuTextureDimension::D2,
            extent,
            1,
            1,
            format,
            GpuTextureUsages::new(&texture_label, [GpuTextureUsage::CopyDestination]).unwrap(),
            GpuTextureInitialization::Prepared(prepared),
        )
        .unwrap();
        assert_eq!(descriptor.format(), format);
    }
}

#[test]
fn r8_new_prepared_texture_uses_one_byte_rows() {
    assert_prepared_texture_rows(&R8_NEW_FORMATS, 1, "R8-new");
}

#[test]
fn rg8_prepared_texture_uses_two_byte_rows() {
    assert_prepared_texture_rows(&RG8_FORMATS, 2, "RG8");
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

fn wait_for_readback(
    context: &GpuContext,
    submission: &GpuSubmission,
    id: GpuReadbackId,
    family: &str,
) -> GpuReadbackBytes {
    let readback = submission.readback(id).unwrap().clone();
    let deadline = Instant::now() + Duration::from_secs(15);
    let bytes = loop {
        context.progress();
        match readback.status() {
            GpuReadbackStatus::Ready(bytes) => break bytes,
            GpuReadbackStatus::Failed(error) => panic!("{family} readback failed: {error:?}"),
            GpuReadbackStatus::Pending => {}
        }
        if let GpuSubmissionStatus::Failed(error) = submission.status() {
            panic!("{family} submission failed: {error:?}");
        }
        assert!(Instant::now() < deadline, "{family} readback timed out");
        std::thread::yield_now();
    };
    loop {
        context.progress();
        match submission.status() {
            GpuSubmissionStatus::Completed => break,
            GpuSubmissionStatus::Failed(error) => panic!("{family} submission failed: {error:?}"),
            GpuSubmissionStatus::Accepted => {}
        }
        assert!(
            Instant::now() < deadline,
            "{family} submission did not finish"
        );
        std::thread::yield_now();
    }
    bytes
}

fn run_native_copy_family(
    formats: &[GpuTextureFormat],
    widths: &[u32],
    copy_block_bytes: u32,
    family: &str,
) -> usize {
    let mut requirements = GpuCapabilityRequirements::new();
    requirements
        .insert(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::Copy,
        ))
        .unwrap();
    let census = pollster::block_on(GpuContext::request(
        GpuContextDescriptor::new(requirements.clone())
            .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
            .with_allowed_backends([GpuBackendFamily::Vulkan])
            .with_label(format!("{family} native format census")),
    ))
    .expect("native Conformance must provide a Vulkan software adapter");

    let mut exercised = 0;
    for format in formats.iter().copied() {
        let facts = census
            .adapter_facts()
            .supported()
            .format(format)
            .expect("tested format must be enumerated");
        let rejected_depth = pollster::block_on(GpuContext::request(
            GpuContextDescriptor::new(requirements.clone())
                .require_format_role(format, GpuFormatRole::DepthStencil)
                .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
                .with_allowed_backends([GpuBackendFamily::Vulkan])
                .with_label(format!("{family} non-depth role rejection proof")),
        ));
        match rejected_depth {
            Err(error) => assert_eq!(
                error.category(),
                GpuContextRequestErrorCategory::NoAdmissibleCandidate,
                "{format:?} must reject the depth/stencil role at candidate admission"
            ),
            Ok(_) => panic!("{format:?} must not admit the depth/stencil role"),
        }
        if !facts.copy_source || !facts.copy_destination {
            println!("{format:?}: SKIPPED (copy roles not both advertised)");
            continue;
        }
        let context = pollster::block_on(GpuContext::request(
            GpuContextDescriptor::new(requirements.clone())
                .require_format_role(format, GpuFormatRole::CopySource)
                .require_format_role(format, GpuFormatRole::CopyDestination)
                .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
                .with_allowed_backends([GpuBackendFamily::Vulkan])
                .with_label(format!("{family} native format copy proof")),
        ))
        .expect("observed copy roles must be admitted for the chosen format");
        let admitted = context.adapter_facts().supported().format(format).unwrap();
        assert!(admitted.copy_source && admitted.copy_destination);
        for width in widths.iter().copied() {
            let height = 2;
            let name = format!("{family} {format:?} {width}x{height}");
            let expected = (0..width * height * copy_block_bytes)
                .map(|index| (index % 251) as u8)
                .collect::<Vec<_>>();
            let mut allocator = GpuWorkResourceIdAllocator::new();
            let mut texture = |suffix: &str| {
                let resource_name = format!("{name} {suffix}");
                let resource_label = label(&resource_name);
                let extent = GpuTextureExtent::new(
                    &resource_label,
                    GpuTextureDimension::D2,
                    width,
                    height,
                    1,
                )
                .unwrap();
                allocator
                    .allocate_texture_handle(
                        GpuTextureDescriptor::new(
                            common(&resource_name),
                            GpuTextureDimension::D2,
                            extent,
                            1,
                            1,
                            format,
                            GpuTextureUsages::new(
                                &resource_label,
                                [
                                    GpuTextureUsage::CopySource,
                                    GpuTextureUsage::CopyDestination,
                                ],
                            )
                            .unwrap(),
                            GpuTextureInitialization::Uninitialized,
                        )
                        .unwrap(),
                    )
                    .unwrap()
            };
            let source = texture("source");
            let destination = texture("destination");
            let realized_source = context.realize_texture(&source).unwrap();
            let realized_destination = context.realize_texture(&destination).unwrap();
            let source_region = GpuTextureCopyRegion::new(
                &source,
                0,
                GpuTextureOrigin::new(0, 0, 0),
                GpuTextureAspect::Color,
                GpuCopyExtent::new(width, height, 1).unwrap(),
            )
            .unwrap();
            let destination_region = GpuTextureCopyRegion::new(
                &destination,
                0,
                GpuTextureOrigin::new(0, 0, 0),
                GpuTextureAspect::Color,
                GpuCopyExtent::new(width, height, 1).unwrap(),
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
                GpuCopyOperation::texture_to_texture(source_region, destination_region.clone())
                    .unwrap();
            let readback_id = GpuReadbackId::allocate().unwrap();
            let readback =
                GpuReadbackOperation::new(destination_region.into(), readback_id).unwrap();
            let mut builder = GpuWorkFragmentBuilder::new(label(&name), provenance(&name));
            builder.declare_resource(source.into()).unwrap();
            builder.declare_resource(destination.into()).unwrap();
            add_operation(
                &mut builder,
                &format!("upload {name}"),
                GpuWorkOperation::Upload(upload),
            );
            add_operation(
                &mut builder,
                &format!("copy {name}"),
                GpuWorkOperation::Copy(copy),
            );
            add_operation(
                &mut builder,
                &format!("readback {name}"),
                GpuWorkOperation::Readback(readback),
            );
            let graph =
                GpuPreparedWorkGraph::prepare(label(&name), [builder.finish().unwrap()]).unwrap();
            let prepared = pollster::block_on(context.prepare_submission(graph)).unwrap();
            let submission = context.submit_prepared(prepared).unwrap();
            let bytes = wait_for_readback(&context, &submission, readback_id, family);
            assert_eq!(
                bytes.as_bytes(),
                expected.as_slice(),
                "{format:?} {width}px content"
            );
            assert_eq!(bytes.layout().byte_len(), expected.len() as u64);
            assert_eq!(bytes.texture_format(), Some(format));
            println!(
                "{format:?}: PASS {width}x{height}, {} bytes per logical row",
                width * copy_block_bytes
            );
            drop(realized_destination);
            drop(realized_source);
        }
        println!("{format:?}: EXERCISED (all {} widths)", widths.len());
        exercised += 1;
    }
    exercised
}

fn assert_native_copy_family_qualified(exercised: usize, family: &str) {
    assert!(
        exercised > 0,
        "{family} native copy proof NOT QUALIFIED: all formats skipped"
    );
}

#[test]
fn native_copy_family_qualification_fails_closed_without_gpu() {
    for (family, full) in [("R8-new", 3), ("RG8", 4)] {
        assert!(
            std::panic::catch_unwind(|| assert_native_copy_family_qualified(0, family)).is_err()
        );
        assert_native_copy_family_qualified(1, family);
        assert_native_copy_family_qualified(full, family);
    }
}

#[test]
#[ignore = "requires a Vulkan software adapter; executed by RunenGPU native Conformance CI"]
fn r8_new_native_copy_round_trips_per_observed_format() {
    let formats = R8_NEW_FORMATS.map(|entry| entry.0);
    let exercised = run_native_copy_family(&formats, &[255, 256], 1, "R8-new");
    println!(
        "R8-new native copy proofs: {exercised}/3 formats exercised; all-skipped is NOT qualified"
    );
    assert_native_copy_family_qualified(exercised, "R8-new");
}

#[test]
#[ignore = "requires a Vulkan software adapter; executed by RunenGPU native Conformance CI"]
fn rg8_native_copy_round_trips_per_observed_format() {
    let formats = RG8_FORMATS.map(|entry| entry.0);
    let exercised = run_native_copy_family(&formats, &[127, 128], 2, "RG8");
    println!(
        "RG8 native copy proofs: {exercised}/4 formats exercised; all-skipped is NOT qualified"
    );
    assert_native_copy_family_qualified(exercised, "RG8");
}
