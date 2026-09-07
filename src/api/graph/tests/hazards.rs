use super::support::*;

fn dependency_reasons(
    graph: &GpuPreparedWorkGraph,
    before: u64,
    after: u64,
) -> &[GpuDependencyReason] {
    graph
        .dependencies()
        .iter()
        .find(|dependency| {
            dependency.before().local_node() == before && dependency.after().local_node() == after
        })
        .map(GpuWorkDependency::reasons)
        .expect("the requested dependency should exist")
}

#[test]
fn inferred_hazards_are_typed_and_disjoint_regions_remain_independent() {
    let mut allocator = allocator();
    let buffer = buffer(
        &mut allocator,
        "hazards",
        GpuBufferInitialization::Zeroed,
        [GpuBufferUsage::Storage],
    );
    let first = GpuBufferRange::new(&buffer, 0, 16).unwrap();
    let second = GpuBufferRange::new(&buffer, 32, 16).unwrap();
    let mut fragment = builder("hazards");
    fragment
        .declare_resource(GpuResourceRef::Buffer(buffer.clone()))
        .unwrap();
    add_compute(
        &mut fragment,
        "read",
        [buffer_access(
            &buffer,
            first,
            GpuBufferAccessKind::StorageRead,
        )],
    );
    add_compute(
        &mut fragment,
        "write",
        [buffer_access(
            &buffer,
            first,
            GpuBufferAccessKind::StorageWrite,
        )],
    );
    add_compute(
        &mut fragment,
        "read again",
        [buffer_access(
            &buffer,
            first,
            GpuBufferAccessKind::StorageRead,
        )],
    );
    add_compute(
        &mut fragment,
        "disjoint write",
        [buffer_access(
            &buffer,
            second,
            GpuBufferAccessKind::StorageWrite,
        )],
    );
    let graph =
        GpuPreparedWorkGraph::prepare(label("hazard graph"), [fragment.finish().unwrap()]).unwrap();
    let reasons = graph
        .dependencies()
        .iter()
        .flat_map(GpuWorkDependency::reasons)
        .collect::<Vec<_>>();
    assert!(
        reasons
            .iter()
            .any(|reason| matches!(reason, GpuDependencyReason::WriteAfterRead { .. }))
    );
    assert!(
        reasons
            .iter()
            .any(|reason| matches!(reason, GpuDependencyReason::ReadAfterWrite { .. }))
    );
    assert!(graph.dependencies().iter().all(|dependency| {
        dependency.before().local_node() != 4 && dependency.after().local_node() != 4
    }));
}

#[test]
fn partially_overlapping_buffers_retain_exact_raw_war_and_waw_regions() {
    let mut allocator = allocator();
    let buffer = buffer(
        &mut allocator,
        "partial buffer hazards",
        GpuBufferInitialization::Zeroed,
        [GpuBufferUsage::Storage],
    );
    let read_first = GpuBufferRange::new(&buffer, 0, 32).unwrap();
    let write_middle = GpuBufferRange::new(&buffer, 16, 32).unwrap();
    let read_later = GpuBufferRange::new(&buffer, 24, 32).unwrap();
    let write_last = GpuBufferRange::new(&buffer, 32, 32).unwrap();
    let mut fragment = builder("partial buffer hazards");
    fragment
        .declare_resource(GpuResourceRef::Buffer(buffer.clone()))
        .unwrap();
    for (name, range, kind) in [
        ("initial read", read_first, GpuBufferAccessKind::StorageRead),
        (
            "middle write",
            write_middle,
            GpuBufferAccessKind::StorageWrite,
        ),
        ("later read", read_later, GpuBufferAccessKind::StorageRead),
        ("last write", write_last, GpuBufferAccessKind::StorageWrite),
    ] {
        add_compute(&mut fragment, name, [buffer_access(&buffer, range, kind)]);
    }
    let graph =
        GpuPreparedWorkGraph::prepare(label("partial buffer graph"), [fragment.finish().unwrap()])
            .unwrap();

    let identity = buffer.diagnostic_identity();
    assert!(
        dependency_reasons(&graph, 1, 2).contains(&GpuDependencyReason::WriteAfterRead {
            resource: identity,
            region: GpuDependencyRegion::Buffer(GpuBufferRange::new(&buffer, 16, 16).unwrap()),
        })
    );
    assert!(
        dependency_reasons(&graph, 2, 3).contains(&GpuDependencyReason::ReadAfterWrite {
            resource: identity,
            region: GpuDependencyRegion::Buffer(GpuBufferRange::new(&buffer, 24, 24).unwrap()),
        })
    );
    assert!(
        dependency_reasons(&graph, 2, 4).contains(&GpuDependencyReason::WriteAfterWrite {
            resource: identity,
            region: GpuDependencyRegion::Buffer(GpuBufferRange::new(&buffer, 32, 16).unwrap()),
        })
    );
}

#[test]
fn buffer_hazard_truth_table_is_lexically_oriented() {
    let mut allocator = allocator();
    let buffer = buffer(
        &mut allocator,
        "truth table",
        GpuBufferInitialization::Zeroed,
        [GpuBufferUsage::Storage],
    );
    let range = GpuBufferRange::whole(&buffer).unwrap();
    let mut fragment = builder("truth table");
    fragment
        .declare_resource(GpuResourceRef::Buffer(buffer.clone()))
        .unwrap();
    for (name, kind) in [
        ("read one", GpuBufferAccessKind::StorageRead),
        ("read two", GpuBufferAccessKind::StorageRead),
        ("write one", GpuBufferAccessKind::StorageWrite),
        ("write two", GpuBufferAccessKind::StorageWrite),
        ("read write", GpuBufferAccessKind::StorageReadWrite),
    ] {
        add_compute(&mut fragment, name, [buffer_access(&buffer, range, kind)]);
    }
    let graph =
        GpuPreparedWorkGraph::prepare(label("truth table graph"), [fragment.finish().unwrap()])
            .unwrap();
    let reasons = |before, after| {
        graph.dependencies().iter().find_map(|dependency| {
            (dependency.before().local_node() == before && dependency.after().local_node() == after)
                .then(|| dependency.reasons())
        })
    };
    assert!(reasons(1, 2).is_none());
    assert!(
        reasons(1, 3)
            .unwrap()
            .iter()
            .any(|reason| { matches!(reason, GpuDependencyReason::WriteAfterRead { .. }) })
    );
    assert!(
        reasons(3, 4)
            .unwrap()
            .iter()
            .any(|reason| { matches!(reason, GpuDependencyReason::WriteAfterWrite { .. }) })
    );
    let read_write = reasons(4, 5).unwrap();
    assert!(
        read_write
            .iter()
            .any(|reason| matches!(reason, GpuDependencyReason::ReadAfterWrite { .. }))
    );
    assert!(
        read_write
            .iter()
            .any(|reason| matches!(reason, GpuDependencyReason::WriteAfterWrite { .. }))
    );
}

#[test]
fn disjoint_query_ranges_remain_independent() {
    let mut allocator = allocator();
    let queries = allocator
        .allocate_query_set_handle(
            GpuQuerySetDescriptor::new(common("disjoint queries"), GpuQueryKind::Timestamp, 4)
                .unwrap(),
        )
        .unwrap();
    let first_target = texture(
        &mut allocator,
        "first query target",
        GpuTextureInitialization::Uninitialized,
        1,
        1,
        [GpuTextureUsage::ColorAttachment],
    );
    let second_target = texture(
        &mut allocator,
        "second query target",
        GpuTextureInitialization::Uninitialized,
        1,
        1,
        [GpuTextureUsage::ColorAttachment],
    );
    let first_view = texture_view(
        &mut allocator,
        &first_target,
        "first query target view",
        GpuTextureSubresourceRange::whole(&first_target).unwrap(),
    );
    let second_view = texture_view(
        &mut allocator,
        &second_target,
        "second query target view",
        GpuTextureSubresourceRange::whole(&second_target).unwrap(),
    );
    let operation = |view: GpuTextureViewHandle, beginning_of_pass, end_of_pass| {
        GpuWorkOperation::Render(
            GpuRenderOperation::new(
                [GpuRenderColorAttachment::new(
                    view,
                    GpuColorAttachmentLoad::Clear(
                        GpuColorClearValue::new(0.0, 0.0, 0.0, 1.0).unwrap(),
                    ),
                    GpuAttachmentStore::Store,
                    None,
                )
                .unwrap()],
                None,
                [],
                Some(
                    GpuTimestampWrites::new(&queries, Some(beginning_of_pass), Some(end_of_pass))
                        .unwrap(),
                ),
            )
            .unwrap(),
        )
    };
    let mut fragment = builder("disjoint queries");
    for resource in [
        GpuResourceRef::QuerySet(queries.clone()),
        GpuResourceRef::Texture(first_target),
        GpuResourceRef::TextureView(first_view.clone()),
        GpuResourceRef::Texture(second_target),
        GpuResourceRef::TextureView(second_view.clone()),
    ] {
        fragment.declare_resource(resource).unwrap();
    }
    for (name, view, beginning_of_pass, end_of_pass) in [
        ("first queries", first_view, 0, 1),
        ("second queries", second_view, 2, 3),
    ] {
        fragment
            .add_node(
                label(name),
                operation(view, beginning_of_pass, end_of_pass),
                [],
                GpuCapabilityRequirements::new(),
                GpuExecutionPreference::GraphicsRequired,
                provenance(name),
            )
            .unwrap();
    }
    let graph =
        GpuPreparedWorkGraph::prepare(label("disjoint query graph"), [fragment.finish().unwrap()])
            .unwrap();
    assert!(graph.dependencies().is_empty());
}

#[test]
fn partially_overlapping_query_ranges_retain_the_exact_intersection() {
    let mut allocator = allocator();
    let queries = allocator
        .allocate_query_set_handle(
            GpuQuerySetDescriptor::new(common("partial queries"), GpuQueryKind::Timestamp, 8)
                .unwrap(),
        )
        .unwrap();
    let mut fragment = builder("partial queries");
    fragment
        .declare_resource(GpuResourceRef::QuerySet(queries.clone()))
        .unwrap();
    fragment
        .add_input(
            GpuWorkResourceInput::new(
                GpuResourceRef::QuerySet(queries.clone()),
                GpuInitialCoverage::query_ranges(
                    &queries,
                    [GpuQueryRange::new(&queries, 5, 3).unwrap()],
                )
                .unwrap(),
                provenance("partial query input"),
            )
            .unwrap(),
        )
        .unwrap();
    let GpuWorkOperation::Compute(compute) = compute_operation() else {
        panic!("test fixture must create compute work");
    };
    let compute =
        compute.with_timestamp_writes(GpuTimestampWrites::new(&queries, Some(4), None).unwrap());
    fragment
        .add_node(
            label("write queries"),
            GpuWorkOperation::Compute(compute),
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::ComputePreferred,
            provenance("write queries"),
        )
        .unwrap();
    add_compute(
        &mut fragment,
        "read queries",
        [GpuResourceAccess::Query(
            GpuQueryAccess::new(
                &queries,
                GpuQueryRange::new(&queries, 4, 4).unwrap(),
                GpuQueryAccessKind::ResolveSource,
            )
            .unwrap(),
        )],
    );
    let graph =
        GpuPreparedWorkGraph::prepare(label("partial query graph"), [fragment.finish().unwrap()])
            .unwrap();
    assert_eq!(
        dependency_reasons(&graph, 1, 2),
        [GpuDependencyReason::ReadAfterWrite {
            resource: queries.diagnostic_identity(),
            region: GpuDependencyRegion::Query(GpuQueryRange::new(&queries, 4, 1).unwrap()),
        }]
    );
}

#[test]
fn disjoint_texture_subresources_do_not_create_false_dependencies() {
    let mut allocator = allocator();
    let texture = texture(
        &mut allocator,
        "subresources",
        GpuTextureInitialization::Uninitialized,
        2,
        2,
        [GpuTextureUsage::StorageWrite],
    );
    let first_range = GpuTextureSubresourceRange::new(
        texture.descriptor().common().label(),
        0,
        1,
        0,
        1,
        GpuTextureAspect::Color,
    )
    .unwrap();
    let second_range = GpuTextureSubresourceRange::new(
        texture.descriptor().common().label(),
        1,
        1,
        1,
        1,
        GpuTextureAspect::Color,
    )
    .unwrap();
    let access = |range| {
        GpuResourceAccess::Texture(
            GpuTextureAccess::new(
                GpuTextureAccessResource::Texture(texture.clone()),
                range,
                GpuTextureAccessKind::StorageWrite,
            )
            .unwrap(),
        )
    };
    let mut fragment = builder("texture disjoint");
    fragment
        .declare_resource(GpuResourceRef::Texture(texture.clone()))
        .unwrap();
    add_compute(&mut fragment, "first mip", [access(first_range)]);
    add_compute(&mut fragment, "second mip", [access(second_range)]);
    let graph = GpuPreparedWorkGraph::prepare(label("texture graph"), [fragment.finish().unwrap()])
        .unwrap();
    assert!(graph.dependencies().is_empty());
    assert_eq!(graph.topological_order().len(), 2);
}

#[test]
fn partially_overlapping_texture_subresources_retain_normalized_parent_intersection() {
    let mut allocator = allocator();
    let texture = texture(
        &mut allocator,
        "partial texture",
        GpuTextureInitialization::Zeroed,
        4,
        4,
        [
            GpuTextureUsage::CopyDestination,
            GpuTextureUsage::StorageRead,
            GpuTextureUsage::StorageWrite,
        ],
    );
    let writer = GpuTextureSubresourceRange::new(
        texture.descriptor().common().label(),
        0,
        3,
        0,
        3,
        GpuTextureAspect::All,
    )
    .unwrap();
    let reader = GpuTextureSubresourceRange::new(
        texture.descriptor().common().label(),
        1,
        3,
        1,
        3,
        GpuTextureAspect::Color,
    )
    .unwrap();
    let mut fragment = builder("partial texture");
    fragment
        .declare_resource(GpuResourceRef::Texture(texture.clone()))
        .unwrap();
    add_compute(
        &mut fragment,
        "texture writer",
        [texture_access(
            &texture,
            writer,
            GpuTextureAccessKind::StorageWrite,
        )],
    );
    add_compute(
        &mut fragment,
        "texture reader",
        [texture_access(
            &texture,
            reader,
            GpuTextureAccessKind::StorageRead,
        )],
    );
    let graph =
        GpuPreparedWorkGraph::prepare(label("partial texture graph"), [fragment.finish().unwrap()])
            .unwrap();
    let intersection = GpuTextureSubresourceRange::new(
        texture.descriptor().common().label(),
        1,
        2,
        1,
        2,
        GpuTextureAspect::Color,
    )
    .unwrap();
    assert_eq!(
        dependency_reasons(&graph, 1, 2),
        [GpuDependencyReason::ReadAfterWrite {
            resource: texture.diagnostic_identity(),
            region: GpuDependencyRegion::Texture(intersection),
        }]
    );
}

#[test]
fn one_dependency_edge_retains_multiple_distinct_overlap_regions() {
    let mut allocator = allocator();
    let buffer = buffer(
        &mut allocator,
        "multiple overlap regions",
        GpuBufferInitialization::Zeroed,
        [GpuBufferUsage::Storage],
    );
    let mut fragment = builder("multiple overlap regions");
    fragment
        .declare_resource(GpuResourceRef::Buffer(buffer.clone()))
        .unwrap();
    add_compute(
        &mut fragment,
        "two writes",
        [
            buffer_access(
                &buffer,
                GpuBufferRange::new(&buffer, 0, 16).unwrap(),
                GpuBufferAccessKind::StorageWrite,
            ),
            buffer_access(
                &buffer,
                GpuBufferRange::new(&buffer, 32, 16).unwrap(),
                GpuBufferAccessKind::StorageWrite,
            ),
        ],
    );
    add_compute(
        &mut fragment,
        "two reads",
        [
            buffer_access(
                &buffer,
                GpuBufferRange::new(&buffer, 8, 16).unwrap(),
                GpuBufferAccessKind::StorageRead,
            ),
            buffer_access(
                &buffer,
                GpuBufferRange::new(&buffer, 40, 16).unwrap(),
                GpuBufferAccessKind::StorageRead,
            ),
        ],
    );
    let graph = GpuPreparedWorkGraph::prepare(
        label("multiple overlap graph"),
        [fragment.finish().unwrap()],
    )
    .unwrap();
    let reasons = dependency_reasons(&graph, 1, 2);
    assert_eq!(reasons.len(), 2);
    for (offset, size) in [(8, 8), (40, 8)] {
        assert!(reasons.contains(&GpuDependencyReason::ReadAfterWrite {
            resource: buffer.diagnostic_identity(),
            region: GpuDependencyRegion::Buffer(
                GpuBufferRange::new(&buffer, offset, size).unwrap()
            ),
        }));
    }
}

#[test]
fn texture_view_candidates_use_normalized_parent_texture_identity() {
    let mut allocator = allocator();
    let texture = texture(
        &mut allocator,
        "candidate parent texture",
        GpuTextureInitialization::Zeroed,
        2,
        1,
        [GpuTextureUsage::StorageRead, GpuTextureUsage::StorageWrite],
    );
    let range = GpuTextureSubresourceRange::new(
        texture.descriptor().common().label(),
        0,
        1,
        0,
        1,
        GpuTextureAspect::Color,
    )
    .unwrap();
    let view = texture_view(&mut allocator, &texture, "candidate texture view", range);

    let mut fragment = builder("normalized texture candidates");
    fragment
        .declare_resource(GpuResourceRef::Texture(texture.clone()))
        .unwrap();
    fragment
        .declare_resource(GpuResourceRef::TextureView(view.clone()))
        .unwrap();
    add_compute(
        &mut fragment,
        "write parent texture",
        [texture_access(
            &texture,
            range,
            GpuTextureAccessKind::StorageWrite,
        )],
    );
    add_compute(
        &mut fragment,
        "read through view",
        [GpuResourceAccess::Texture(
            GpuTextureAccess::new(
                GpuTextureAccessResource::TextureView(view),
                range,
                GpuTextureAccessKind::StorageRead,
            )
            .unwrap(),
        )],
    );

    let graph = GpuPreparedWorkGraph::prepare(
        label("normalized texture candidate graph"),
        [fragment.finish().unwrap()],
    )
    .unwrap();

    assert_eq!(
        dependency_reasons(&graph, 1, 2),
        [GpuDependencyReason::ReadAfterWrite {
            resource: texture.diagnostic_identity(),
            region: GpuDependencyRegion::Texture(range),
        }]
    );
}

#[test]
fn same_node_access_candidates_never_self_edge_and_duplicate_reasons_collapse() {
    let mut allocator = allocator();
    let buffer = buffer(
        &mut allocator,
        "candidate buffer",
        GpuBufferInitialization::Zeroed,
        [GpuBufferUsage::Storage],
    );
    let first_read = GpuBufferRange::new(&buffer, 0, 32).unwrap();
    let second_read = GpuBufferRange::new(&buffer, 0, 48).unwrap();
    let write = GpuBufferRange::new(&buffer, 8, 8).unwrap();

    let mut fragment = builder("same-node candidates");
    fragment
        .declare_resource(GpuResourceRef::Buffer(buffer.clone()))
        .unwrap();
    add_compute(
        &mut fragment,
        "overlapping reads",
        [
            buffer_access(&buffer, first_read, GpuBufferAccessKind::StorageRead),
            buffer_access(&buffer, second_read, GpuBufferAccessKind::StorageRead),
        ],
    );
    add_compute(
        &mut fragment,
        "write shared overlap",
        [buffer_access(
            &buffer,
            write,
            GpuBufferAccessKind::StorageWrite,
        )],
    );

    let graph = GpuPreparedWorkGraph::prepare(
        label("same-node candidate graph"),
        [fragment.finish().unwrap()],
    )
    .unwrap();

    assert!(
        graph
            .dependencies()
            .iter()
            .all(|dependency| dependency.before() != dependency.after())
    );
    assert_eq!(graph.dependencies().len(), 1);
    assert_eq!(
        dependency_reasons(&graph, 1, 2),
        [GpuDependencyReason::WriteAfterRead {
            resource: buffer.diagnostic_identity(),
            region: GpuDependencyRegion::Buffer(write),
        }]
    );
}
