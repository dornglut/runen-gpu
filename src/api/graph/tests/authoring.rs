use super::super::authoring::{normalize_node_accesses, preserve_caller_readable_accesses};
use super::super::coverage::buffer_coverage_contains;
use super::support::*;

fn naive_buffer_coverage_bytes(values: &[GpuBufferCoverage]) -> std::collections::BTreeSet<u64> {
    let mut bytes = std::collections::BTreeSet::new();
    for value in values {
        match value {
            GpuBufferCoverage::Dense(range) => {
                bytes.extend(range.offset()..range.end());
            }
            GpuBufferCoverage::Strided(coverage) => {
                for group in 0..u64::from(coverage.group_count()) {
                    for segment in 0..u64::from(coverage.segment_count()) {
                        let start = coverage.first()
                            + group * coverage.group_stride()
                            + segment * coverage.segment_stride();
                        bytes.extend(start..start + coverage.segment_size());
                    }
                }
            }
        }
    }
    bytes
}

#[test]
fn initial_coverage_is_checked_normalized_and_kind_preserving() {
    let mut allocator = allocator();
    let buffer = buffer(
        &mut allocator,
        "coverage",
        GpuBufferInitialization::Uninitialized,
        [GpuBufferUsage::Storage],
    );
    let coverage = GpuInitialCoverage::buffer(
        &buffer,
        [
            GpuBufferCoverage::dense(GpuBufferRange::new(&buffer, 16, 16).unwrap()),
            GpuBufferCoverage::dense(GpuBufferRange::new(&buffer, 0, 16).unwrap()),
            GpuBufferCoverage::dense(GpuBufferRange::new(&buffer, 8, 8).unwrap()),
        ],
    )
    .unwrap();
    assert_eq!(coverage.kind(), GpuInitialCoverageKind::Buffer);
    assert_eq!(coverage.buffer_values().unwrap().len(), 1);
    assert_eq!(
        coverage.buffer_values().unwrap()[0],
        GpuBufferCoverage::dense(GpuBufferRange::new(&buffer, 0, 32).unwrap())
    );

    let queries = allocator
        .allocate_query_set_handle(
            GpuQuerySetDescriptor::new(common("queries"), GpuQueryKind::Timestamp, 4).unwrap(),
        )
        .unwrap();
    assert!(
        GpuInitialCoverage::descriptor_initialization(GpuResourceRef::QuerySet(queries)).is_err()
    );
}

#[test]
fn compact_buffer_coverage_compares_by_exact_semantics() {
    let mut allocator = allocator();
    let buffer = buffer(
        &mut allocator,
        "semantic coverage",
        GpuBufferInitialization::Uninitialized,
        [GpuBufferUsage::Storage],
    );
    let dense = GpuInitialCoverage::buffer(
        &buffer,
        [GpuBufferCoverage::dense(
            GpuBufferRange::new(&buffer, 0, 64).unwrap(),
        )],
    )
    .unwrap();
    let strided = GpuInitialCoverage::buffer(
        &buffer,
        [GpuBufferCoverage::strided(
            GpuBufferStridedCoverage::new(&buffer, 0, 32, 32, 2, 0, 1).unwrap(),
        )],
    )
    .unwrap();
    assert_eq!(dense, strided);
}

#[test]
fn compact_buffer_coverage_matches_an_independent_explicit_byte_oracle() {
    let mut allocator = allocator();
    let buffer = buffer(
        &mut allocator,
        "coverage oracle",
        GpuBufferInitialization::Uninitialized,
        [GpuBufferUsage::Storage],
    );
    let terms = vec![
        GpuBufferCoverage::dense(GpuBufferRange::new(&buffer, 0, 1).unwrap()),
        GpuBufferCoverage::dense(GpuBufferRange::new(&buffer, 1, 3).unwrap()),
        GpuBufferCoverage::dense(GpuBufferRange::new(&buffer, 4, 4).unwrap()),
        GpuBufferCoverage::dense(GpuBufferRange::new(&buffer, 8, 8).unwrap()),
        GpuBufferCoverage::strided(
            GpuBufferStridedCoverage::new(&buffer, 0, 1, 2, 4, 0, 1).unwrap(),
        ),
        GpuBufferCoverage::strided(
            GpuBufferStridedCoverage::new(&buffer, 1, 1, 2, 4, 0, 1).unwrap(),
        ),
        GpuBufferCoverage::strided(
            GpuBufferStridedCoverage::new(&buffer, 0, 2, 4, 2, 8, 2).unwrap(),
        ),
        GpuBufferCoverage::strided(
            GpuBufferStridedCoverage::new(&buffer, 1, 1, 3, 2, 8, 2).unwrap(),
        ),
    ];
    let mut unions = vec![Vec::new()];
    for term in &terms {
        unions.push(vec![term.clone()]);
    }
    for (index, first) in terms.iter().enumerate() {
        for second in &terms[index..] {
            unions.push(vec![first.clone(), second.clone()]);
        }
    }
    unions.push(vec![terms[0].clone(), terms[4].clone(), terms[6].clone()]);
    unions.push(vec![terms[1].clone(), terms[5].clone(), terms[7].clone()]);

    for have in &unions {
        let have_bytes = naive_buffer_coverage_bytes(have);
        for required in &unions {
            let required_bytes = naive_buffer_coverage_bytes(required);
            assert_eq!(
                buffer_coverage_contains(have, required),
                required_bytes.is_subset(&have_bytes),
                "compact containment disagreed with the explicit-byte oracle: have={have:?}, required={required:?}",
            );
            if !have.is_empty() && !required.is_empty() {
                let have_coverage = GpuInitialCoverage::buffer(&buffer, have.clone()).unwrap();
                let required_coverage =
                    GpuInitialCoverage::buffer(&buffer, required.clone()).unwrap();
                assert_eq!(
                    have_coverage == required_coverage,
                    have_bytes == required_bytes,
                    "compact equality disagreed with the explicit-byte oracle: left={have:?}, right={required:?}",
                );
            }
        }
    }
}

#[test]
fn large_strided_buffer_coverage_stays_compact() {
    let mut allocator = allocator();
    let resource_label = label("large compact coverage");
    let buffer = allocator
        .allocate_buffer_handle(
            GpuBufferDescriptor::new(
                common("large compact coverage"),
                64_000_000,
                GpuBufferUsages::new(&resource_label, [GpuBufferUsage::Storage]).unwrap(),
                GpuBufferInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();
    let coverage = GpuInitialCoverage::buffer(
        &buffer,
        [GpuBufferCoverage::strided(
            GpuBufferStridedCoverage::new(&buffer, 0, 32, 64, 1_000_000, 0, 1).unwrap(),
        )],
    )
    .unwrap();
    assert_eq!(coverage.buffer_values().unwrap().len(), 1);
    assert_eq!(
        match &coverage.buffer_values().unwrap()[0] {
            GpuBufferCoverage::Strided(coverage) => coverage.segment_count(),
            GpuBufferCoverage::Dense(_) => 0,
        },
        1_000_000
    );
}

#[test]
fn differently_shaped_large_strided_coverage_compares_exactly_and_canonically() {
    let mut allocator = allocator();
    let resource_label = label("large equivalent coverage");
    let buffer = allocator
        .allocate_buffer_handle(
            GpuBufferDescriptor::new(
                common("large equivalent coverage"),
                6_400_000,
                GpuBufferUsages::new(&resource_label, [GpuBufferUsage::Storage]).unwrap(),
                GpuBufferInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();
    let flat = GpuInitialCoverage::buffer(
        &buffer,
        [GpuBufferCoverage::strided(
            GpuBufferStridedCoverage::new(&buffer, 0, 32, 64, 100_000, 0, 1).unwrap(),
        )],
    )
    .unwrap();
    let grouped = GpuInitialCoverage::buffer(
        &buffer,
        [GpuBufferCoverage::strided(
            GpuBufferStridedCoverage::new(&buffer, 0, 32, 64, 100, 6_400, 1_000).unwrap(),
        )],
    )
    .unwrap();
    assert_eq!(flat, grouped);
    assert_eq!(flat.buffer_values().unwrap().len(), 1);
    assert_eq!(grouped.buffer_values().unwrap().len(), 1);

    let canonical = GpuInitialCoverage::buffer(
        &buffer,
        [GpuBufferCoverage::strided(
            GpuBufferStridedCoverage::new(&buffer, 0, 32, 32, 2, 0, 1).unwrap(),
        )],
    )
    .unwrap();
    assert_eq!(
        canonical.buffer_values().unwrap(),
        [GpuBufferCoverage::dense(
            GpuBufferRange::new(&buffer, 0, 64).unwrap()
        )]
    );
    let one_segment = GpuInitialCoverage::buffer(
        &buffer,
        [GpuBufferCoverage::strided(
            GpuBufferStridedCoverage::new(&buffer, 64, 32, 48, 1, 96, 1).unwrap(),
        )],
    )
    .unwrap();
    assert_eq!(
        one_segment.buffer_values().unwrap(),
        [GpuBufferCoverage::dense(
            GpuBufferRange::new(&buffer, 64, 32).unwrap()
        )]
    );
}

#[test]
fn compact_buffer_coverage_removes_all_cheaply_subsumed_terms_deterministically() {
    let mut allocator = allocator();
    let buffer = buffer(
        &mut allocator,
        "subsumed compact coverage",
        GpuBufferInitialization::Uninitialized,
        [GpuBufferUsage::Storage],
    );
    let superset = GpuBufferCoverage::strided(
        GpuBufferStridedCoverage::new(&buffer, 0, 8, 16, 4, 0, 1).unwrap(),
    );
    let subset = GpuBufferCoverage::strided(
        GpuBufferStridedCoverage::new(&buffer, 16, 8, 16, 2, 0, 1).unwrap(),
    );
    let dense_segment = GpuBufferCoverage::dense(GpuBufferRange::new(&buffer, 32, 8).unwrap());

    let first = GpuInitialCoverage::buffer(
        &buffer,
        [subset.clone(), dense_segment.clone(), superset.clone()],
    )
    .unwrap();
    let second =
        GpuInitialCoverage::buffer(&buffer, [superset.clone(), dense_segment, subset]).unwrap();
    assert_eq!(first, second);
    assert_eq!(first.buffer_values().unwrap(), [superset]);
    assert_eq!(
        second.buffer_values().unwrap(),
        first.buffer_values().unwrap()
    );
}

#[test]
fn strided_buffer_coverage_subsumes_only_dense_intervals_in_covered_runs() {
    let mut allocator = allocator();
    let buffer = buffer(
        &mut allocator,
        "strided dense containment",
        GpuBufferInitialization::Uninitialized,
        [GpuBufferUsage::Storage],
    );
    let strided = GpuBufferCoverage::strided(
        GpuBufferStridedCoverage::new(&buffer, 0, 8, 16, 2, 32, 2).unwrap(),
    );

    let fully_inside = GpuInitialCoverage::buffer(
        &buffer,
        [
            strided.clone(),
            GpuBufferCoverage::dense(GpuBufferRange::new(&buffer, 17, 4).unwrap()),
        ],
    )
    .unwrap();
    assert_eq!(
        fully_inside.buffer_values().unwrap(),
        std::slice::from_ref(&strided)
    );

    let group_boundary = GpuInitialCoverage::buffer(
        &buffer,
        [
            strided.clone(),
            GpuBufferCoverage::dense(GpuBufferRange::new(&buffer, 32, 8).unwrap()),
        ],
    )
    .unwrap();
    assert_eq!(
        group_boundary.buffer_values().unwrap(),
        std::slice::from_ref(&strided)
    );

    for range in [
        GpuBufferRange::new(&buffer, 6, 10).unwrap(),
        GpuBufferRange::new(&buffer, 24, 8).unwrap(),
    ] {
        let coverage =
            GpuInitialCoverage::buffer(&buffer, [strided.clone(), GpuBufferCoverage::dense(range)])
                .unwrap();
        assert_eq!(coverage.buffer_values().unwrap().len(), 2);
        assert!(
            coverage
                .buffer_values()
                .unwrap()
                .contains(&GpuBufferCoverage::dense(range))
        );
        assert!(coverage.buffer_values().unwrap().contains(&strided));
    }
}

#[test]
fn same_node_access_deduplicates_merges_and_rejects_incompatible_roles() {
    let mut allocator = allocator();
    let buffer = buffer(
        &mut allocator,
        "storage",
        GpuBufferInitialization::Uninitialized,
        [GpuBufferUsage::Storage, GpuBufferUsage::Uniform],
    );
    let range = GpuBufferRange::whole(&buffer).unwrap();
    let read = buffer_access(&buffer, range, GpuBufferAccessKind::StorageRead);
    let write = buffer_access(&buffer, range, GpuBufferAccessKind::StorageWrite);
    let mut fragment = builder("normalize");
    fragment
        .declare_resource(GpuResourceRef::Buffer(buffer.clone()))
        .unwrap();
    add_compute(
        &mut fragment,
        "read write",
        [read.clone(), read.clone(), write],
    );
    let fragment = fragment.finish().unwrap();
    assert_eq!(fragment.nodes()[0].accesses().len(), 1);
    assert!(matches!(
        fragment.nodes()[0].accesses()[0],
        GpuResourceAccess::Buffer(ref access)
            if access.kind() == GpuBufferAccessKind::StorageReadWrite
    ));
    assert_eq!(fragment.nodes()[0].caller_readable_accesses(), [read]);

    let mut invalid = builder("invalid normalize");
    invalid
        .declare_resource(GpuResourceRef::Buffer(buffer.clone()))
        .unwrap();
    let error = invalid
        .add_node(
            label("contradiction"),
            compute_operation(),
            [
                buffer_access(&buffer, range, GpuBufferAccessKind::UniformRead),
                buffer_access(&buffer, range, GpuBufferAccessKind::StorageWrite),
            ],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::Automatic,
            provenance("contradiction"),
        )
        .unwrap_err();
    assert_eq!(
        error.cause(),
        GpuWorkAuthoringCause::IncompatibleSameNodeAccess
    );
}

#[test]
fn caller_readable_initialization_truth_survives_derived_normalization() {
    let mut allocator = allocator();
    let buffer = buffer(
        &mut allocator,
        "preserved caller read",
        GpuBufferInitialization::Uninitialized,
        [GpuBufferUsage::Storage],
    );
    let range = GpuBufferRange::whole(&buffer).unwrap();
    let derived = buffer_access(&buffer, range, GpuBufferAccessKind::StorageReadWrite);
    let caller = buffer_access(&buffer, range, GpuBufferAccessKind::StorageRead);
    let normalized = normalize_node_accesses(
        &label("fragment"),
        &label("node"),
        &provenance("node"),
        vec![derived.clone()],
        vec![caller.clone()],
    )
    .unwrap();
    assert_eq!(normalized, [derived]);
    assert_eq!(
        preserve_caller_readable_accesses(std::slice::from_ref(&caller)),
        [caller]
    );
}

#[test]
fn duplicate_resource_declaration_is_transactional_and_authoring_can_continue() {
    let mut allocator = allocator();
    let buffer = buffer(
        &mut allocator,
        "transactional resource",
        GpuBufferInitialization::Zeroed,
        [GpuBufferUsage::Storage],
    );
    let original = GpuResourceRef::Buffer(buffer.clone());
    let replacement_label = label("replacement resource");
    let replacement = GpuResourceRef::Buffer(GpuBufferHandle::from_descriptor(
        buffer.diagnostic_identity(),
        GpuBufferDescriptor::new(
            common("replacement resource"),
            32,
            GpuBufferUsages::new(&replacement_label, [GpuBufferUsage::Storage]).unwrap(),
            GpuBufferInitialization::Zeroed,
        )
        .unwrap(),
    ));
    let mut fragment = builder("transactional declaration");
    fragment.declare_resource(original.clone()).unwrap();

    let error = fragment.declare_resource(replacement).unwrap_err();
    assert_eq!(error.cause(), GpuWorkAuthoringCause::DuplicateResource);

    add_compute(
        &mut fragment,
        "continued authoring",
        [buffer_access(
            &buffer,
            GpuBufferRange::whole(&buffer).unwrap(),
            GpuBufferAccessKind::StorageWrite,
        )],
    );
    let fragment = fragment.finish().unwrap();
    assert!(matches!(
        fragment.resources(),
        [GpuResourceRef::Buffer(retained)]
            if retained.descriptor().common().label().as_str() == "transactional resource"
                && retained.descriptor().size_bytes() == 64
    ));
    assert_eq!(fragment.nodes().len(), 1);
}

#[test]
fn canonical_compute_builder_accepts_only_execution_complete_operation_state() {
    let operation = compute_operation_with_dispatch(GpuDispatchSize::new(4, 1, 1));
    let GpuWorkOperation::Compute(compute) = operation.clone() else {
        panic!("test fixture must create compute work");
    };
    let mut fragment = builder("canonical compute");
    let node_id = fragment.compute("integrate", compute).unwrap();
    assert_eq!(node_id.diagnostic_local(), 1);

    let fragment = fragment.finish().unwrap();
    let node = &fragment.nodes()[0];
    assert_eq!(node.operation(), &operation);
    assert!(matches!(
        node.operation(),
        GpuWorkOperation::Compute(operation)
            if operation.dispatch().direct_size().is_some_and(|size| size.as_array() == [4, 1, 1])
    ));
    assert!(
        node.requirements()
            .get(GpuCapabilityFeature::Compute)
            .is_some()
    );
}

#[test]
fn caller_cannot_repeat_an_operation_derived_access() {
    let mut allocator = allocator();
    let queries = allocator
        .allocate_query_set_handle(
            GpuQuerySetDescriptor::new(common("derived query"), GpuQueryKind::Timestamp, 2)
                .unwrap(),
        )
        .unwrap();
    let target = texture(
        &mut allocator,
        "derived timestamp target",
        GpuTextureInitialization::Uninitialized,
        1,
        1,
        [GpuTextureUsage::ColorAttachment],
    );
    let target_range = GpuTextureSubresourceRange::whole(&target).unwrap();
    let target_view = texture_view(
        &mut allocator,
        &target,
        "derived timestamp target view",
        target_range,
    );
    let access = GpuQueryAccess::new(
        &queries,
        GpuQueryRange::new(&queries, 0, 1).unwrap(),
        GpuQueryAccessKind::WriteTimestamp,
    )
    .unwrap();
    let timestamp_writes = GpuTimestampWrites::new(&queries, Some(0), None).unwrap();
    let attachment = GpuRenderColorAttachment::new(
        target_view.clone(),
        GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 1.0).unwrap()),
        GpuAttachmentStore::Store,
        None,
    )
    .unwrap();
    let operation = GpuWorkOperation::Render(
        GpuRenderOperation::new([attachment], None, [], Some(timestamp_writes)).unwrap(),
    );
    let mut fragment = builder("derived access");
    for resource in [
        GpuResourceRef::QuerySet(queries),
        GpuResourceRef::Texture(target),
        GpuResourceRef::TextureView(target_view),
    ] {
        fragment.declare_resource(resource).unwrap();
    }
    let error = fragment
        .add_node(
            label("timestamp"),
            operation,
            [GpuResourceAccess::Query(access)],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::GraphicsRequired,
            provenance("timestamp"),
        )
        .unwrap_err();
    assert_eq!(
        error.cause(),
        GpuWorkAuthoringCause::OperationAccessContradiction
    );
}
