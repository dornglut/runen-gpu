use runen_gpu::*;
use std::time::{Duration, Instant};

const PACKED_FORMATS: [(GpuTextureFormat, GpuShaderIoScalarClass, u8, bool); 4] = [
    (
        GpuTextureFormat::Rgb9e5Ufloat,
        GpuShaderIoScalarClass::Float,
        3,
        false,
    ),
    (
        GpuTextureFormat::Rgb10a2Uint,
        GpuShaderIoScalarClass::Uint,
        4,
        true,
    ),
    (
        GpuTextureFormat::Rgb10a2Unorm,
        GpuShaderIoScalarClass::Float,
        4,
        true,
    ),
    (
        GpuTextureFormat::Rg11b10Ufloat,
        GpuShaderIoScalarClass::Float,
        3,
        false,
    ),
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
    requirements
        .insert(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::Copy,
        ))
        .unwrap();
    requirements
}

#[test]
fn packed32_public_semantics_and_program_typing_are_exact() {
    for (format, class, width, alpha) in PACKED_FORMATS {
        assert_eq!(format.block_dimensions(), (1, 1));
        assert_eq!(format.copy_block_size(GpuTextureAspect::All), Some(4));
        assert_eq!(format.copy_block_size(GpuTextureAspect::Color), Some(4));
        assert_eq!(format.copy_block_size(GpuTextureAspect::DepthOnly), None);
        assert_eq!(format.copy_block_size(GpuTextureAspect::StencilOnly), None);
        assert!(!format.is_depth());
        assert!(!format.is_stencil());
        assert!(!format.is_srgb());

        let target = GpuColorTargetStateDescriptor::new(
            format,
            GpuBlendMode::Replace,
            GpuColorWriteMask::ALL,
        )
        .unwrap();
        let signature = GpuFragmentOutputStateDescriptor::new([target])
            .expected_signature(GpuEntryPointName::new("fragment_main").unwrap())
            .unwrap();
        let output = signature.locations().next().unwrap();
        assert_eq!(output.value_type().scalar_class(), class);
        assert_eq!(output.value_type().vector_width().get(), width);
        assert_eq!(target.has_blendable_alpha_channel(), alpha);

        let alpha_result =
            GpuColorTargetStateDescriptor::new(format, GpuBlendMode::Alpha, GpuColorWriteMask::ALL);
        if class == GpuShaderIoScalarClass::Uint {
            assert!(alpha_result.is_err());
        } else {
            assert!(alpha_result.is_ok());
        }
    }
}

#[test]
fn packed_tier1_storage_texels_remain_outside_the_normalized_wgsl_contract() {
    for storage_format in ["rgb10a2uint", "rgb10a2unorm", "rg11b10ufloat"] {
        let source_text = format!(
            "@group(0) @binding(0) var image: texture_storage_2d<{storage_format}, write>;\n\
             @compute @workgroup_size(1) fn inspect() {{\n\
                 let dimensions = textureDimensions(image);\n\
             }}\n"
        );
        let identity = GpuProgramSourceIdentity::new(
            GpuProgramSourceOwnerId::allocate().unwrap(),
            GpuProgramSourceKey::new("r1.packed32.storage-tier").unwrap(),
            GpuProgramSourceRevision::try_from_raw(1).unwrap(),
        );
        let mut registry = GpuProgramSourceRegistry::new(2, 16 * 1024).unwrap();
        let source = registry
            .admit_wgsl(
                identity,
                &source_text,
                GpuProgramSourceProvenance::new("r1-packed32-storage-tier", None).unwrap(),
            )
            .unwrap();
        GpuProgramDescriptor::new(
            source,
            [GpuEntryPointName::new("inspect").unwrap()],
            std::iter::empty::<GpuBindingLayoutRefinement>(),
        )
        .expect_err("tier1 packed storage texels must remain fail-closed on WGPU 30");
    }
}

fn texture(
    allocator: &mut GpuWorkResourceIdAllocator,
    format: GpuTextureFormat,
    name: &str,
    width: u32,
    usages: impl IntoIterator<Item = GpuTextureUsage>,
) -> GpuTextureHandle {
    let resource_label = label(name);
    allocator
        .allocate_texture_handle(
            GpuTextureDescriptor::new(
                common(name),
                GpuTextureDimension::D2,
                GpuTextureExtent::new(&resource_label, GpuTextureDimension::D2, width, 2, 1)
                    .unwrap(),
                1,
                1,
                format,
                GpuTextureUsages::new(&resource_label, usages).unwrap(),
                GpuTextureInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap()
}

fn wait_for_readback(
    context: &GpuContext,
    submission: &GpuSubmission,
    id: GpuReadbackId,
    name: &str,
) -> GpuReadbackBytes {
    let readback = submission.readback(id).unwrap().clone();
    let deadline = Instant::now() + Duration::from_secs(15);
    let bytes = loop {
        context.progress();
        match readback.status() {
            GpuReadbackStatus::Ready(bytes) => break bytes,
            GpuReadbackStatus::Failed(error) => panic!("{name} readback failed: {error:?}"),
            GpuReadbackStatus::Pending => {}
        }
        if let GpuSubmissionStatus::Failed(error) = submission.status() {
            panic!("{name} submission failed: {error:?}");
        }
        assert!(Instant::now() < deadline, "{name} readback timed out");
        std::thread::yield_now();
    };
    loop {
        context.progress();
        match submission.status() {
            GpuSubmissionStatus::Completed => break,
            GpuSubmissionStatus::Failed(error) => panic!("{name} submission failed: {error:?}"),
            GpuSubmissionStatus::Accepted => {}
        }
        assert!(
            Instant::now() < deadline,
            "{name} submission did not finish"
        );
        std::thread::yield_now();
    }
    bytes
}

fn exercise_role_realization(
    format: GpuTextureFormat,
    role: GpuFormatRole,
    usage: GpuTextureUsage,
    label_text: &str,
) {
    let mut role_requirements = GpuCapabilityRequirements::new();
    if role == GpuFormatRole::ColorAttachment {
        role_requirements
            .insert(GpuCapabilityRequirement::Required(
                GpuCapabilityFeature::RenderPipeline,
            ))
            .unwrap();
    }
    let context = pollster::block_on(GpuContext::request(
        GpuContextDescriptor::new(role_requirements)
            .require_format_role(format, role)
            .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
            .with_allowed_backends([GpuBackendFamily::Vulkan])
            .with_label(label_text),
    ))
    .expect("advertised packed role must admit a Vulkan fallback context");
    let mut allocator = GpuWorkResourceIdAllocator::new();
    let texture = texture(&mut allocator, format, label_text, 4, [usage]);
    let _realized = context.realize_texture(&texture).unwrap();
}

fn run_native_packed32() -> usize {
    let census = pollster::block_on(GpuContext::request(
        GpuContextDescriptor::new(requirements())
            .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
            .with_allowed_backends([GpuBackendFamily::Vulkan])
            .with_label("Packed32 native census"),
    ))
    .expect("native Conformance must provide the retained Vulkan software adapter");

    let mut exercised = 0;
    for (format, _, _, _) in PACKED_FORMATS {
        let facts = census
            .adapter_facts()
            .supported()
            .format(format)
            .expect("packed format must be present in the normalized census");

        if facts.sampled {
            exercise_role_realization(
                format,
                GpuFormatRole::Sampled,
                GpuTextureUsage::Sampled,
                &format!("{format:?} native sampled realization"),
            );
            println!("{format:?} Sampled: EXERCISED");
        } else {
            println!("{format:?} Sampled: SKIPPED (role not advertised)");
        }

        if facts.color_attachment {
            exercise_role_realization(
                format,
                GpuFormatRole::ColorAttachment,
                GpuTextureUsage::ColorAttachment,
                &format!("{format:?} native color-attachment realization"),
            );
            println!("{format:?} ColorAttachment: EXERCISED");
        } else {
            println!("{format:?} ColorAttachment: SKIPPED (role not advertised)");
        }

        if !facts.copy_source || !facts.copy_destination {
            println!("{format:?} Copy: SKIPPED (copy roles not both advertised)");
            continue;
        }

        let context = pollster::block_on(GpuContext::request(
            GpuContextDescriptor::new(requirements())
                .require_format_role(format, GpuFormatRole::CopySource)
                .require_format_role(format, GpuFormatRole::CopyDestination)
                .with_fallback_policy(GpuSoftwareFallbackPolicy::Require)
                .with_allowed_backends([GpuBackendFamily::Vulkan])
                .with_label(format!("{format:?} native packed copy proof")),
        ))
        .expect("advertised packed copy roles must admit a context");

        for width in [63_u32, 64_u32] {
            let name = format!("{format:?} packed copy {width}x2");
            let expected = vec![0_u8; usize::try_from(width * 2 * 4).unwrap()];
            let mut allocator = GpuWorkResourceIdAllocator::new();
            let source = texture(
                &mut allocator,
                format,
                &format!("{name} source"),
                width,
                [
                    GpuTextureUsage::CopySource,
                    GpuTextureUsage::CopyDestination,
                ],
            );
            let destination = texture(
                &mut allocator,
                format,
                &format!("{name} destination"),
                width,
                [
                    GpuTextureUsage::CopySource,
                    GpuTextureUsage::CopyDestination,
                ],
            );
            let extent = GpuCopyExtent::new(width, 2, 1).unwrap();
            let source_region = GpuTextureCopyRegion::new(
                &source,
                0,
                GpuTextureOrigin::new(0, 0, 0),
                GpuTextureAspect::Color,
                extent,
            )
            .unwrap();
            let destination_region = GpuTextureCopyRegion::new(
                &destination,
                0,
                GpuTextureOrigin::new(0, 0, 0),
                GpuTextureAspect::Color,
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
                GpuCopyOperation::texture_to_texture(source_region, destination_region.clone())
                    .unwrap();
            let readback_id = GpuReadbackId::allocate().unwrap();
            let readback =
                GpuReadbackOperation::new(destination_region.into(), readback_id).unwrap();
            let mut builder = GpuWorkFragmentBuilder::new(label(&name), provenance(&name));
            builder.declare_resource(source.into()).unwrap();
            builder.declare_resource(destination.into()).unwrap();
            for (node, operation) in [
                ("packed upload", GpuWorkOperation::Upload(upload)),
                ("packed copy", GpuWorkOperation::Copy(copy)),
                ("packed readback", GpuWorkOperation::Readback(readback)),
            ] {
                builder
                    .add_node(
                        label(node),
                        operation,
                        [],
                        GpuCapabilityRequirements::new(),
                        GpuExecutionPreference::TransferPreferred,
                        provenance(node),
                    )
                    .unwrap();
            }
            let graph =
                GpuPreparedWorkGraph::prepare(label(&name), [builder.finish().unwrap()]).unwrap();
            let prepared = pollster::block_on(context.prepare_submission(graph)).unwrap();
            let submission = context.submit_prepared(prepared).unwrap();
            let bytes = wait_for_readback(&context, &submission, readback_id, &name);
            assert_eq!(bytes.as_bytes(), expected.as_slice());
            assert_eq!(bytes.layout().byte_len(), expected.len() as u64);
            assert_eq!(bytes.texture_format(), Some(format));
        }
        println!("{format:?} Copy: EXERCISED zero-valued 63px/64px round trips");
        exercised += 1;
    }
    exercised
}

#[test]
#[ignore = "requires the retained Vulkan software adapter"]
fn packed32_native_roles_and_zero_copy_round_trips() {
    let exercised = run_native_packed32();
    assert!(
        exercised > 0,
        "Packed32 native copy proof NOT QUALIFIED: all formats skipped"
    );
    println!("Packed32 native copy proofs: {exercised}/4 formats exercised");
}
