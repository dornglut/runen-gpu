use super::support::*;

fn retained_buffer(allocator: &mut GpuWorkResourceIdAllocator, name: &str) -> GpuBufferHandle {
    retained_buffer_with_size(allocator, name, 64)
}

fn retained_buffer_with_size(
    allocator: &mut GpuWorkResourceIdAllocator,
    name: &str,
    size_bytes: u64,
) -> GpuBufferHandle {
    let resource_label = label(name);
    let common = GpuResourceCommon::owned(
        resource_label.clone(),
        GpuResourceLifetime::Retained,
        GpuMemoryIntent::Device,
        GpuReconstruction::SourceBacked,
        provenance(name),
    )
    .unwrap();
    allocator
        .allocate_buffer_handle(
            GpuBufferDescriptor::new(
                common,
                size_bytes,
                GpuBufferUsages::new(
                    &resource_label,
                    [GpuBufferUsage::Storage, GpuBufferUsage::CopyDestination],
                )
                .unwrap(),
                GpuBufferInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap()
}

fn retained_texture(allocator: &mut GpuWorkResourceIdAllocator, name: &str) -> GpuTextureHandle {
    let resource_label = label(name);
    let common = GpuResourceCommon::owned(
        resource_label.clone(),
        GpuResourceLifetime::Retained,
        GpuMemoryIntent::Device,
        GpuReconstruction::SourceBacked,
        provenance(name),
    )
    .unwrap();
    allocator
        .allocate_texture_handle(
            GpuTextureDescriptor::new(
                common,
                GpuTextureDimension::D2,
                GpuTextureExtent::new(&resource_label, GpuTextureDimension::D2, 8, 8, 1).unwrap(),
                1,
                1,
                GpuTextureFormat::Rgba8Unorm,
                GpuTextureUsages::new(
                    &resource_label,
                    [GpuTextureUsage::ColorAttachment, GpuTextureUsage::Sampled],
                )
                .unwrap(),
                GpuTextureInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap()
}

fn retained_seed(coverage: GpuInitialCoverage) -> GpuRetainedInitializationSeed {
    GpuRetainedInitializationSeed::new(coverage.resource().clone(), Some(coverage))
}

#[test]
fn retained_coverage_seeds_canonical_read_validation_and_initial_summary() {
    let mut allocator = allocator();
    let buffer = retained_buffer(&mut allocator, "retained history");
    let initialized = GpuBufferRange::new(&buffer, 0, 16).unwrap();
    let retained =
        GpuInitialCoverage::buffer(&buffer, [GpuBufferCoverage::dense(initialized)]).unwrap();

    let mut fragment = builder("retained reader");
    fragment
        .declare_resource(GpuResourceRef::Buffer(buffer.clone()))
        .unwrap();
    add_compute(
        &mut fragment,
        "read retained history",
        [buffer_access(
            &buffer,
            initialized,
            GpuBufferAccessKind::StorageRead,
        )],
    );
    let fragment = fragment.finish().unwrap();

    let unseeded =
        GpuPreparedWorkGraph::prepare(label("unseeded retained read"), [fragment.clone()])
            .unwrap_err();
    assert_eq!(
        unseeded.cause(),
        GpuWorkGraphCause::ReadBeforeInitialization
    );

    let seed = retained_seed(retained.clone());
    let prepared = GpuPreparedWorkGraph::prepare_with_retained_coverage(
        label("seeded retained read"),
        [fragment],
        std::slice::from_ref(&seed),
    )
    .unwrap();
    let summary = prepared
        .initialization()
        .iter()
        .find(|summary| summary.resource().diagnostic_identity() == buffer.diagnostic_identity())
        .unwrap();
    assert_eq!(summary.initial(), Some(&retained));
    assert_eq!(summary.final_coverage(), Some(&retained));
}

#[test]
fn exact_clear_expands_retained_initialized_coverage_through_canonical_simulation() {
    let mut allocator = allocator();
    let buffer = retained_buffer(&mut allocator, "retained exact expansion");
    let retained_range = GpuBufferRange::new(&buffer, 0, 16).unwrap();
    let cleared_range = GpuBufferRange::new(&buffer, 16, 16).unwrap();
    let combined_read = GpuBufferRange::new(&buffer, 0, 32).unwrap();
    let retained =
        GpuInitialCoverage::buffer(&buffer, [GpuBufferCoverage::dense(retained_range)]).unwrap();

    let mut fragment = builder("retained exact expansion");
    fragment
        .declare_resource(GpuResourceRef::Buffer(buffer.clone()))
        .unwrap();
    fragment
        .operation(
            "zero exact retained tail",
            GpuClearOperation::buffer_zero(GpuBufferRegion::new(&buffer, cleared_range).unwrap())
                .unwrap(),
        )
        .unwrap();
    add_compute(
        &mut fragment,
        "read retained plus exact effect",
        [buffer_access(
            &buffer,
            combined_read,
            GpuBufferAccessKind::StorageRead,
        )],
    );

    let prepared = GpuPreparedWorkGraph::prepare_with_retained_coverage(
        label("retained exact expansion graph"),
        [fragment.finish().unwrap()],
        &[retained_seed(retained)],
    )
    .unwrap();
    let final_coverage = prepared
        .initialization()
        .iter()
        .find(|summary| summary.resource().diagnostic_identity() == buffer.diagnostic_identity())
        .unwrap()
        .final_coverage()
        .unwrap()
        .buffer_values()
        .unwrap();
    assert_eq!(
        final_coverage,
        [GpuBufferCoverage::dense(
            GpuBufferRange::new(&buffer, 0, 32).unwrap()
        )]
    );
}

#[test]
fn possible_discard_does_not_remain_failure_preserved_initialization_evidence() {
    let mut allocator = allocator();
    let texture = retained_texture(&mut allocator, "retained discard");
    let range = GpuTextureSubresourceRange::whole(&texture).unwrap();
    let view = texture_view(&mut allocator, &texture, "retained discard view", range);
    let retained = GpuInitialCoverage::texture_subresources(
        &GpuTextureAccessResource::Texture(texture.clone()),
        [range],
    )
    .unwrap();
    let render = GpuRenderOperation::new(
        [GpuRenderColorAttachment::new(
            view.clone(),
            GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 1.0).unwrap()),
            GpuAttachmentStore::Discard,
            None,
        )
        .unwrap()],
        None,
        [],
        None,
    )
    .unwrap();

    let mut fragment = builder("retained discard");
    fragment
        .declare_resource(GpuResourceRef::Texture(texture.clone()))
        .unwrap();
    fragment
        .declare_resource(GpuResourceRef::TextureView(view))
        .unwrap();
    fragment
        .operation("discard retained attachment", render)
        .unwrap();

    let seed = retained_seed(retained.clone());
    let prepared = GpuPreparedWorkGraph::prepare_with_retained_coverage(
        label("retained discard graph"),
        [fragment.finish().unwrap()],
        std::slice::from_ref(&seed),
    )
    .unwrap();
    let summary = prepared
        .initialization()
        .iter()
        .find(|summary| summary.resource().diagnostic_identity() == texture.diagnostic_identity())
        .unwrap();
    assert_eq!(summary.initial(), Some(&retained));
    assert!(summary.final_coverage().is_none());
    assert!(
        prepared
            .failure_preserved_coverage(texture.diagnostic_identity())
            .is_none()
    );
}

#[test]
fn retained_seed_cannot_initialize_current_transient_storage_with_equal_identity() {
    let mut transient_allocator = allocator();
    let transient = buffer(
        &mut transient_allocator,
        "current transient storage",
        GpuBufferInitialization::Uninitialized,
        [GpuBufferUsage::Storage],
    );
    let mut retained_allocator = allocator();
    let retained_source = retained_buffer(&mut retained_allocator, "foreign retained descriptor");
    assert_eq!(
        transient.diagnostic_identity(),
        retained_source.diagnostic_identity(),
        "the proof requires equal process-local diagnostic identity with different descriptors"
    );

    let current_range = GpuBufferRange::new(&transient, 0, 16).unwrap();
    let retained_range = GpuBufferRange::new(&retained_source, 0, 16).unwrap();
    let invalid_seed =
        GpuInitialCoverage::buffer(&retained_source, [GpuBufferCoverage::dense(retained_range)])
            .unwrap();

    let mut fragment = builder("transient reader");
    fragment
        .declare_resource(GpuResourceRef::Buffer(transient.clone()))
        .unwrap();
    add_compute(
        &mut fragment,
        "read transient storage",
        [buffer_access(
            &transient,
            current_range,
            GpuBufferAccessKind::StorageRead,
        )],
    );

    let error = GpuPreparedWorkGraph::prepare_with_retained_coverage(
        label("equal identity retained seed versus transient storage"),
        [fragment.finish().unwrap()],
        &[retained_seed(invalid_seed)],
    )
    .unwrap_err();
    assert_eq!(error.cause(), GpuWorkGraphCause::ReadBeforeInitialization);
}

#[test]
fn retained_seed_with_changed_descriptor_is_ignored_before_canonical_simulation() {
    let mut current_allocator = allocator();
    let current =
        retained_buffer_with_size(&mut current_allocator, "current retained descriptor", 32);
    let mut stale_allocator = allocator();
    let stale = retained_buffer_with_size(&mut stale_allocator, "stale retained descriptor", 64);
    assert_eq!(current.diagnostic_identity(), stale.diagnostic_identity());
    assert_ne!(current.descriptor(), stale.descriptor());

    let stale_range = GpuBufferRange::new(&stale, 0, 64).unwrap();
    let stale_seed =
        GpuInitialCoverage::buffer(&stale, [GpuBufferCoverage::dense(stale_range)]).unwrap();
    let current_read = GpuBufferRange::new(&current, 0, 16).unwrap();

    let mut fragment = builder("changed retained descriptor");
    fragment
        .declare_resource(GpuResourceRef::Buffer(current.clone()))
        .unwrap();
    add_compute(
        &mut fragment,
        "read current retained descriptor",
        [buffer_access(
            &current,
            current_read,
            GpuBufferAccessKind::StorageRead,
        )],
    );

    let error = GpuPreparedWorkGraph::prepare_with_retained_coverage(
        label("changed retained descriptor graph"),
        [fragment.finish().unwrap()],
        &[retained_seed(stale_seed)],
    )
    .unwrap_err();
    assert_eq!(error.cause(), GpuWorkGraphCause::ReadBeforeInitialization);
    assert_eq!(error.resource(), Some(current.diagnostic_identity()));
}

#[test]
fn generic_shader_write_does_not_expand_retained_initialized_coverage() {
    let mut allocator = allocator();
    let buffer = retained_buffer(&mut allocator, "partially initialized history");
    let retained_range = GpuBufferRange::new(&buffer, 0, 16).unwrap();
    let shader_written = GpuBufferRange::new(&buffer, 16, 16).unwrap();
    let combined_read = GpuBufferRange::new(&buffer, 0, 32).unwrap();
    let retained =
        GpuInitialCoverage::buffer(&buffer, [GpuBufferCoverage::dense(retained_range)]).unwrap();

    let mut fragment = builder("generic retained write");
    fragment
        .declare_resource(GpuResourceRef::Buffer(buffer.clone()))
        .unwrap();
    add_compute(
        &mut fragment,
        "generic write outside retained coverage",
        [buffer_access(
            &buffer,
            shader_written,
            GpuBufferAccessKind::StorageWrite,
        )],
    );
    add_compute(
        &mut fragment,
        "read retained plus generic write",
        [buffer_access(
            &buffer,
            combined_read,
            GpuBufferAccessKind::StorageRead,
        )],
    );

    let error = GpuPreparedWorkGraph::prepare_with_retained_coverage(
        label("generic write retained coverage"),
        [fragment.finish().unwrap()],
        &[retained_seed(retained)],
    )
    .unwrap_err();
    assert_eq!(error.cause(), GpuWorkGraphCause::ReadBeforeInitialization);
    assert_eq!(error.resource(), Some(buffer.diagnostic_identity()));
}
