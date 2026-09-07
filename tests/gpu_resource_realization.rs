use runen_gpu::{
    GpuAddressMode, GpuBufferDescriptor, GpuBufferInitialization, GpuBufferUsage, GpuBufferUsages,
    GpuCapabilityFeature, GpuCapabilityRequirement, GpuCapabilityRequirements, GpuContext,
    GpuContextDescriptor, GpuContextRequestErrorCategory, GpuFilterMode, GpuFormatRole,
    GpuMemoryIntent, GpuQueryKind, GpuQuerySetDescriptor, GpuRealizationPolicies,
    GpuReconstruction, GpuResourceCommon, GpuResourceLabel, GpuResourceLifetime,
    GpuResourceOwnership, GpuResourceProvenance, GpuResourceRealizationErrorCategory,
    GpuResourceRealizationPolicy, GpuSamplerDescriptor, GpuTextureDescriptor, GpuTextureDimension,
    GpuTextureExtent, GpuTextureFormat, GpuTextureInitialization, GpuTextureSubresourceRange,
    GpuTextureUsage, GpuTextureUsages, GpuTextureViewDescriptor, GpuWorkResourceIdAllocator,
};
use std::num::NonZeroUsize;

fn owned_common(label: &str) -> GpuResourceCommon {
    let label = GpuResourceLabel::new(label).unwrap();
    let provenance = GpuResourceProvenance::new(label.clone(), None, None);
    GpuResourceCommon::owned(
        label,
        GpuResourceLifetime::Retained,
        GpuMemoryIntent::Device,
        GpuReconstruction::SourceBacked,
        provenance,
    )
    .unwrap()
}

fn buffer_descriptor(label: &str) -> GpuBufferDescriptor {
    let common = owned_common(label);
    let usages = GpuBufferUsages::new(common.label(), [GpuBufferUsage::Storage]).unwrap();
    GpuBufferDescriptor::new(common, 64, usages, GpuBufferInitialization::Uninitialized).unwrap()
}

fn imported_buffer_descriptor(label: &str) -> GpuBufferDescriptor {
    let label = GpuResourceLabel::new(label).unwrap();
    let provenance = GpuResourceProvenance::new(label.clone(), Some(7), None);
    let common = GpuResourceCommon::imported(label, GpuResourceLifetime::Retained, provenance);
    let usages = GpuBufferUsages::new(common.label(), [GpuBufferUsage::Storage]).unwrap();
    GpuBufferDescriptor::new(common, 64, usages, GpuBufferInitialization::Uninitialized).unwrap()
}

fn admitted_context() -> Result<GpuContext, runen_gpu::GpuContextRequestError> {
    let mut requirements = GpuCapabilityRequirements::new();
    requirements
        .insert(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::TimestampQuery,
        ))
        .unwrap();
    let descriptor = GpuContextDescriptor::new(requirements)
        .with_label("G4C1 representative resource realization")
        .require_format_role(GpuTextureFormat::Rgba8Unorm, GpuFormatRole::Sampled);
    pollster::block_on(GpuContext::request_with_realization_policies(
        descriptor,
        GpuRealizationPolicies::new(
            GpuResourceRealizationPolicy::new(NonZeroUsize::new(5).unwrap()),
            Default::default(),
        ),
    ))
}

#[test]
fn representative_resources_realize_transactionally_or_report_environment_absence() {
    let context = match admitted_context() {
        Ok(context) => context,
        Err(error)
            if matches!(
                error.category(),
                GpuContextRequestErrorCategory::NoAdapterAvailable
                    | GpuContextRequestErrorCategory::NoAdmissibleCandidate
                    | GpuContextRequestErrorCategory::MandatoryFeatureMissing
            ) =>
        {
            eprintln!("G4C1 representative environment unavailable: {error}");
            return;
        }
        Err(error) => panic!("unexpected G4C1 context admission failure: {error}"),
    };
    assert_eq!(context.resource_realization_policy().max_records().get(), 5);

    let mut allocator = GpuWorkResourceIdAllocator::new();
    let buffer = allocator
        .allocate_buffer_handle(buffer_descriptor("representative buffer"))
        .unwrap();

    let texture_common = owned_common("representative texture");
    let texture_extent =
        GpuTextureExtent::new(texture_common.label(), GpuTextureDimension::D2, 4, 4, 1).unwrap();
    let texture_usages =
        GpuTextureUsages::new(texture_common.label(), [GpuTextureUsage::Sampled]).unwrap();
    let texture = allocator
        .allocate_texture_handle(
            GpuTextureDescriptor::new(
                texture_common,
                GpuTextureDimension::D2,
                texture_extent,
                1,
                1,
                GpuTextureFormat::Rgba8Unorm,
                texture_usages,
                GpuTextureInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();

    let view_common = owned_common("representative texture view");
    let subresources = GpuTextureSubresourceRange::new(
        view_common.label(),
        0,
        1,
        0,
        1,
        runen_gpu::GpuTextureAspect::Color,
    )
    .unwrap();
    let view = allocator
        .allocate_texture_view_handle(
            GpuTextureViewDescriptor::new(
                view_common,
                &texture,
                None,
                GpuTextureDimension::D2,
                subresources,
            )
            .unwrap(),
        )
        .unwrap();

    let sampler = allocator
        .allocate_sampler_handle(
            GpuSamplerDescriptor::new(
                owned_common("representative sampler"),
                GpuAddressMode::ClampToEdge,
                GpuAddressMode::ClampToEdge,
                GpuAddressMode::ClampToEdge,
                GpuFilterMode::Nearest,
                GpuFilterMode::Nearest,
                GpuFilterMode::Nearest,
                0.0,
                16.0,
                None,
            )
            .unwrap(),
        )
        .unwrap();
    let query_set = allocator
        .allocate_query_set_handle(
            GpuQuerySetDescriptor::new(
                owned_common("representative query set"),
                GpuQueryKind::Timestamp,
                2,
            )
            .unwrap(),
        )
        .unwrap();

    let realized_buffer = context.realize_buffer(&buffer).unwrap();
    let duplicate_buffer = context.realize_buffer(&buffer).unwrap();
    assert!(realized_buffer.is_same_record(&duplicate_buffer));
    let concurrent_buffers = std::thread::scope(|scope| {
        (0..4)
            .map(|_| scope.spawn(|| context.realize_buffer(&buffer).unwrap()))
            .collect::<Vec<_>>()
            .into_iter()
            .map(|worker| worker.join().expect("realization worker must not panic"))
            .collect::<Vec<_>>()
    });
    assert!(
        concurrent_buffers
            .iter()
            .all(|realized| realized_buffer.is_same_record(realized)),
        "concurrent realization of one identity must publish exactly one record"
    );
    let realized_texture = context.realize_texture(&texture).unwrap();
    let realized_view = context
        .realize_texture_view(&view, &realized_texture)
        .unwrap();
    assert_eq!(
        realized_view.parent_texture_identity(),
        realized_texture.logical_identity()
    );
    let realized_sampler = context.realize_sampler(&sampler).unwrap();
    let realized_query_set = context.realize_query_set(&query_set).unwrap();
    assert_eq!(context.resource_realization_stats().retained_records(), 5);

    let second_context = admitted_context()
        .expect("a second context should remain admissible after the first succeeded");
    let isolated_buffer = second_context.realize_buffer(&buffer).unwrap();
    assert!(!realized_buffer.is_same_record(&isolated_buffer));
    assert_ne!(realized_buffer.affinity(), isolated_buffer.affinity());
    let foreign_parent = second_context.realize_texture(&texture).unwrap();
    let foreign = context
        .realize_texture_view(&view, &foreign_parent)
        .unwrap_err();
    assert_eq!(
        foreign.category(),
        GpuResourceRealizationErrorCategory::ForeignContext
    );

    let oversized_common = owned_common("oversized deterministic rejection");
    let oversized_usages =
        GpuBufferUsages::new(oversized_common.label(), [GpuBufferUsage::Storage]).unwrap();
    let oversized = allocator
        .allocate_buffer_handle(
            GpuBufferDescriptor::new(
                oversized_common,
                u64::MAX,
                oversized_usages,
                GpuBufferInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();
    let incompatible = context.realize_buffer(&oversized).unwrap_err();
    assert_eq!(
        incompatible.category(),
        GpuResourceRealizationErrorCategory::FormatOrAlignmentNotAdmitted
    );
    assert_eq!(context.resource_realization_stats().retained_records(), 5);

    let copy_common = owned_common("unadmitted copy usage");
    let copy_usages =
        GpuBufferUsages::new(copy_common.label(), [GpuBufferUsage::CopyDestination]).unwrap();
    let copy_buffer = allocator
        .allocate_buffer_handle(
            GpuBufferDescriptor::new(
                copy_common,
                64,
                copy_usages,
                GpuBufferInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();
    let unadmitted = context.realize_buffer(&copy_buffer).unwrap_err();
    assert_eq!(
        unadmitted.category(),
        GpuResourceRealizationErrorCategory::RequirementNotAdmitted
    );
    assert_eq!(context.resource_realization_stats().retained_records(), 5);

    let attempted = allocator
        .allocate_buffer_handle(buffer_descriptor("live capacity attempt"))
        .unwrap();
    let capacity = context.realize_buffer(&attempted).unwrap_err();
    assert_eq!(
        capacity.category(),
        GpuResourceRealizationErrorCategory::RegistryCapacityExceeded
    );
    assert_eq!(context.resource_realization_stats().retained_records(), 5);

    let imported = allocator
        .allocate_buffer_handle(imported_buffer_descriptor("unresolved import"))
        .unwrap();
    let unresolved = context.realize_buffer(&imported).unwrap_err();
    assert_eq!(
        unresolved.category(),
        GpuResourceRealizationErrorCategory::ImportSourceUnavailable
    );
    assert_eq!(
        imported.descriptor().common().ownership(),
        GpuResourceOwnership::Imported
    );
    assert_eq!(context.resource_realization_stats().retained_records(), 5);

    drop(duplicate_buffer);
    drop(concurrent_buffers);
    drop(realized_buffer);
    drop(realized_view);
    drop(realized_texture);
    drop(realized_sampler);
    drop(realized_query_set);

    let replacement = context.realize_buffer(&attempted).unwrap();
    assert_eq!(
        replacement.logical_identity(),
        attempted.diagnostic_identity()
    );
    assert_eq!(
        context.resource_realization_stats().retained_records(),
        1,
        "pressure collection must use realization-record liveness, including view-parent retention"
    );
}
