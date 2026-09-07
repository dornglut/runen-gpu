use super::support::*;

#[test]
fn descriptor_initialization_is_region_aware_and_generic_writes_do_not_initialize() {
    let mut allocator = allocator();
    let zeroed = buffer(
        &mut allocator,
        "zeroed",
        GpuBufferInitialization::Zeroed,
        [GpuBufferUsage::Storage],
    );
    let zeroed_range = GpuBufferRange::whole(&zeroed).unwrap();
    let mut readable = builder("descriptor initialized");
    readable
        .declare_resource(GpuResourceRef::Buffer(zeroed.clone()))
        .unwrap();
    add_compute(
        &mut readable,
        "read",
        [buffer_access(
            &zeroed,
            zeroed_range,
            GpuBufferAccessKind::StorageRead,
        )],
    );
    assert!(
        GpuPreparedWorkGraph::prepare(label("descriptor graph"), [readable.finish().unwrap()])
            .is_ok()
    );

    let prepared_data = PreparedGpuData::<TransferData>::from_pod_transfer(
        "prepared buffer bytes",
        &[0_u8; 64],
        provenance("prepared buffer bytes"),
    )
    .unwrap();
    let prepared = buffer(
        &mut allocator,
        "prepared",
        GpuBufferInitialization::Prepared(prepared_data),
        [GpuBufferUsage::Storage, GpuBufferUsage::CopyDestination],
    );
    let mut readable = builder("prepared descriptor is not entry coverage");
    readable
        .declare_resource(GpuResourceRef::Buffer(prepared.clone()))
        .unwrap();
    add_compute(
        &mut readable,
        "caller-only read prepared",
        [buffer_access(
            &prepared,
            GpuBufferRange::whole(&prepared).unwrap(),
            GpuBufferAccessKind::StorageRead,
        )],
    );
    let error = GpuPreparedWorkGraph::prepare(
        label("prepared descriptor graph"),
        [readable.finish().unwrap()],
    )
    .unwrap_err();
    assert_eq!(error.cause(), GpuWorkGraphCause::ReadBeforeInitialization);
    assert_eq!(error.resource(), Some(prepared.diagnostic_identity()));

    let uninitialized = buffer(
        &mut allocator,
        "uninitialized",
        GpuBufferInitialization::Uninitialized,
        [GpuBufferUsage::Storage],
    );
    let written = GpuBufferRange::new(&uninitialized, 0, 16).unwrap();
    let mut partial = builder("partial");
    partial
        .declare_resource(GpuResourceRef::Buffer(uninitialized.clone()))
        .unwrap();
    add_compute(
        &mut partial,
        "write",
        [buffer_access(
            &uninitialized,
            written,
            GpuBufferAccessKind::StorageWrite,
        )],
    );
    add_compute(
        &mut partial,
        "read written",
        [buffer_access(
            &uninitialized,
            written,
            GpuBufferAccessKind::StorageRead,
        )],
    );
    let error = GpuPreparedWorkGraph::prepare(label("partial graph"), [partial.finish().unwrap()])
        .unwrap_err();
    assert_eq!(error.cause(), GpuWorkGraphCause::ReadBeforeInitialization);
    assert_eq!(error.resource(), Some(uninitialized.diagnostic_identity()));
}

#[test]
fn texture_descriptor_initialization_publishes_only_established_zeroed_coverage() {
    let mut allocator = allocator();
    let zeroed = texture(
        &mut allocator,
        "zeroed texture",
        GpuTextureInitialization::Zeroed,
        2,
        1,
        [GpuTextureUsage::Sampled, GpuTextureUsage::CopyDestination],
    );
    let prepared = texture(
        &mut allocator,
        "prepared texture",
        prepared_texture_initialization("prepared texture"),
        2,
        1,
        [GpuTextureUsage::Sampled, GpuTextureUsage::CopyDestination],
    );
    let mut fragment = builder("descriptor texture coverage");
    for resource in [&zeroed, &prepared] {
        fragment
            .declare_resource(GpuResourceRef::Texture(resource.clone()))
            .unwrap();
    }
    add_compute(
        &mut fragment,
        "read zeroed",
        [texture_access(
            &zeroed,
            GpuTextureSubresourceRange::whole(&zeroed).unwrap(),
            GpuTextureAccessKind::SampledRead,
        )],
    );
    let graph = GpuPreparedWorkGraph::prepare(
        label("descriptor texture graph"),
        [fragment.finish().unwrap()],
    )
    .unwrap();
    let zeroed_summary = graph
        .initialization()
        .iter()
        .find(|summary| summary.resource().diagnostic_identity() == zeroed.diagnostic_identity())
        .unwrap();
    assert_eq!(
        zeroed_summary
            .initial()
            .unwrap()
            .texture_subresource_values()
            .unwrap()
            .len(),
        2
    );
    let prepared_summary = graph
        .initialization()
        .iter()
        .find(|summary| summary.resource().diagnostic_identity() == prepared.diagnostic_identity())
        .unwrap();
    assert!(prepared_summary.initial().is_none());
    assert!(prepared_summary.final_coverage().is_none());
}

#[test]
fn caller_only_texture_reads_reject_prepared_metadata_and_uninitialized_storage() {
    let mut allocator = allocator();
    let prepared = texture(
        &mut allocator,
        "prepared texture",
        prepared_texture_initialization("prepared texture"),
        2,
        1,
        [GpuTextureUsage::Sampled, GpuTextureUsage::CopyDestination],
    );
    let uninitialized = texture(
        &mut allocator,
        "uninitialized texture",
        GpuTextureInitialization::Uninitialized,
        1,
        1,
        [GpuTextureUsage::Sampled],
    );
    let prepared_base = GpuTextureSubresourceRange::new(
        prepared.descriptor().common().label(),
        0,
        1,
        0,
        1,
        GpuTextureAspect::Color,
    )
    .unwrap();
    let prepared_mip_one = GpuTextureSubresourceRange::new(
        prepared.descriptor().common().label(),
        1,
        1,
        0,
        1,
        GpuTextureAspect::Color,
    )
    .unwrap();
    for (name, texture, range) in [
        ("prepared base metadata", prepared.clone(), prepared_base),
        ("prepared higher mip metadata", prepared, prepared_mip_one),
        (
            "uninitialized texture",
            uninitialized.clone(),
            GpuTextureSubresourceRange::whole(&uninitialized).unwrap(),
        ),
    ] {
        let mut fragment = builder(name);
        fragment
            .declare_resource(GpuResourceRef::Texture(texture.clone()))
            .unwrap();
        add_compute(
            &mut fragment,
            "invalid caller-only read",
            [texture_access(
                &texture,
                range,
                GpuTextureAccessKind::SampledRead,
            )],
        );
        assert_eq!(
            GpuPreparedWorkGraph::prepare(label(name), [fragment.finish().unwrap()])
                .unwrap_err()
                .cause(),
            GpuWorkGraphCause::ReadBeforeInitialization
        );
    }
}

#[test]
fn attachment_store_preserves_and_discard_invalidates_exact_coverage() {
    let mut allocator = allocator();
    let texture = texture(
        &mut allocator,
        "attachment",
        GpuTextureInitialization::Uninitialized,
        1,
        1,
        [
            GpuTextureUsage::ColorAttachment,
            GpuTextureUsage::CopyDestination,
            GpuTextureUsage::Sampled,
        ],
    );
    let range = GpuTextureSubresourceRange::whole(&texture).unwrap();
    let view = texture_view(&mut allocator, &texture, "attachment view", range);
    let render = |store| {
        GpuWorkOperation::Render(
            GpuRenderOperation::new(
                [GpuRenderColorAttachment::new(
                    view.clone(),
                    GpuColorAttachmentLoad::Clear(
                        GpuColorClearValue::new(0.0, 0.0, 0.0, 1.0).unwrap(),
                    ),
                    store,
                    None,
                )
                .unwrap()],
                None,
                [],
                None,
            )
            .unwrap(),
        )
    };
    let sampled = || {
        GpuResourceAccess::Texture(
            GpuTextureAccess::new(
                GpuTextureAccessResource::Texture(texture.clone()),
                range,
                GpuTextureAccessKind::SampledRead,
            )
            .unwrap(),
        )
    };

    let mut stored = builder("stored");
    stored
        .declare_resource(GpuResourceRef::Texture(texture.clone()))
        .unwrap();
    stored
        .declare_resource(GpuResourceRef::TextureView(view.clone()))
        .unwrap();
    stored
        .add_node(
            label("clear store"),
            render(GpuAttachmentStore::Store),
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::GraphicsRequired,
            provenance("clear store"),
        )
        .unwrap();
    add_compute(&mut stored, "sample", [sampled()]);
    assert!(
        GpuPreparedWorkGraph::prepare(label("stored graph"), [stored.finish().unwrap()]).is_ok()
    );

    let mut discarded = builder("discarded");
    discarded
        .declare_resource(GpuResourceRef::Texture(texture.clone()))
        .unwrap();
    discarded
        .declare_resource(GpuResourceRef::TextureView(view.clone()))
        .unwrap();
    discarded
        .add_node(
            label("clear discard"),
            render(GpuAttachmentStore::Discard),
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::GraphicsRequired,
            provenance("clear discard"),
        )
        .unwrap();
    add_compute(&mut discarded, "sample", [sampled()]);
    let error =
        GpuPreparedWorkGraph::prepare(label("discard graph"), [discarded.finish().unwrap()])
            .unwrap_err();
    assert_eq!(error.cause(), GpuWorkGraphCause::ReadBeforeInitialization);
}

#[test]
fn depth_attachment_load_clear_store_and_discard_drive_initialization() {
    let mut allocator = allocator();
    let depth = depth_texture(&mut allocator, "depth attachment");
    let depth_range = GpuTextureSubresourceRange::whole(&depth).unwrap();
    let depth_view = texture_view(&mut allocator, &depth, "depth attachment view", depth_range);
    let color = texture(
        &mut allocator,
        "depth test color action",
        GpuTextureInitialization::Uninitialized,
        1,
        1,
        [GpuTextureUsage::ColorAttachment],
    );
    let color_range = GpuTextureSubresourceRange::whole(&color).unwrap();
    let color_view = texture_view(&mut allocator, &color, "depth test color view", color_range);
    let render = |load, store| {
        GpuWorkOperation::Render(
            GpuRenderOperation::new(
                [GpuRenderColorAttachment::new(
                    color_view.clone(),
                    GpuColorAttachmentLoad::Clear(
                        GpuColorClearValue::new(0.0, 0.0, 0.0, 1.0).unwrap(),
                    ),
                    GpuAttachmentStore::Store,
                    None,
                )
                .unwrap()],
                Some(
                    GpuRenderDepthStencilAttachment::new(
                        depth_view.clone(),
                        GpuDepthStencilAccess::ReadWrite,
                        load,
                        store,
                    )
                    .unwrap(),
                ),
                [],
                None,
            )
            .unwrap(),
        )
    };
    let sampled = || texture_access(&depth, depth_range, GpuTextureAccessKind::SampledRead);
    let clear = GpuDepthAttachmentLoad::Clear(GpuDepthClearValue::new(0.5).unwrap());
    for (name, store, succeeds) in [
        ("stored depth", GpuAttachmentStore::Store, true),
        ("discarded depth", GpuAttachmentStore::Discard, false),
    ] {
        let mut fragment = builder(name);
        for resource in [&depth, &color] {
            fragment
                .declare_resource(GpuResourceRef::Texture(resource.clone()))
                .unwrap();
        }
        for view in [&depth_view, &color_view] {
            fragment
                .declare_resource(GpuResourceRef::TextureView(view.clone()))
                .unwrap();
        }
        fragment
            .add_node(
                label("clear depth"),
                render(clear, store),
                [],
                GpuCapabilityRequirements::new(),
                GpuExecutionPreference::GraphicsRequired,
                provenance("clear depth"),
            )
            .unwrap();
        add_compute(&mut fragment, "sample depth", [sampled()]);
        let result = GpuPreparedWorkGraph::prepare(label(name), [fragment.finish().unwrap()]);
        assert_eq!(result.is_ok(), succeeds);
        if !succeeds {
            assert_eq!(
                result.unwrap_err().cause(),
                GpuWorkGraphCause::ReadBeforeInitialization
            );
        }
    }

    let mut load = builder("load depth");
    for resource in [&depth, &color] {
        load.declare_resource(GpuResourceRef::Texture(resource.clone()))
            .unwrap();
    }
    for view in [&depth_view, &color_view] {
        load.declare_resource(GpuResourceRef::TextureView(view.clone()))
            .unwrap();
    }
    load.add_node(
        label("load depth"),
        render(GpuDepthAttachmentLoad::Load, GpuAttachmentStore::Store),
        [],
        GpuCapabilityRequirements::new(),
        GpuExecutionPreference::GraphicsRequired,
        provenance("load depth"),
    )
    .unwrap();
    assert_eq!(
        GpuPreparedWorkGraph::prepare(label("load depth graph"), [load.finish().unwrap()])
            .unwrap_err()
            .cause(),
        GpuWorkGraphCause::ReadBeforeInitialization
    );
}

#[test]
fn timestamp_resolve_and_copy_form_one_initialized_dependency_chain() {
    let mut allocator = allocator();
    let queries = allocator
        .allocate_query_set_handle(
            GpuQuerySetDescriptor::new(common("timestamps"), GpuQueryKind::Timestamp, 2).unwrap(),
        )
        .unwrap();
    let resolve = buffer(
        &mut allocator,
        "resolve",
        GpuBufferInitialization::Uninitialized,
        [GpuBufferUsage::QueryResolve, GpuBufferUsage::CopySource],
    );
    let readback = buffer(
        &mut allocator,
        "readback",
        GpuBufferInitialization::Uninitialized,
        [GpuBufferUsage::CopyDestination],
    );
    let target = texture(
        &mut allocator,
        "timestamp render target",
        GpuTextureInitialization::Uninitialized,
        1,
        1,
        [GpuTextureUsage::ColorAttachment],
    );
    let target_range = GpuTextureSubresourceRange::whole(&target).unwrap();
    let target_view = texture_view(
        &mut allocator,
        &target,
        "timestamp render target view",
        target_range,
    );
    let query_range = GpuQueryRange::whole(&queries).unwrap();
    let timestamp_writes = GpuTimestampWrites::new(&queries, Some(0), Some(1)).unwrap();
    let render = GpuWorkOperation::Render(
        GpuRenderOperation::new(
            [GpuRenderColorAttachment::new(
                target_view.clone(),
                GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 1.0).unwrap()),
                GpuAttachmentStore::Store,
                None,
            )
            .unwrap()],
            None,
            [],
            Some(timestamp_writes),
        )
        .unwrap(),
    );
    let query_resolve = GpuQueryResolveOperation::new(&queries, query_range, &resolve, 0).unwrap();
    let resolve_range = query_resolve.destination_range();
    let mut unresolved = builder("unresolved timing");
    for resource in [
        GpuResourceRef::QuerySet(queries.clone()),
        GpuResourceRef::Buffer(resolve.clone()),
    ] {
        unresolved.declare_resource(resource).unwrap();
    }
    unresolved
        .add_node(
            label("resolve unwritten timestamps"),
            GpuWorkOperation::Resolve(query_resolve.clone()),
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::TransferPreferred,
            provenance("resolve unwritten timestamps"),
        )
        .unwrap();
    assert_eq!(
        GpuPreparedWorkGraph::prepare(
            label("unresolved timing graph"),
            [unresolved.finish().unwrap()],
        )
        .unwrap_err()
        .cause(),
        GpuWorkGraphCause::ReadBeforeInitialization
    );
    let copy = GpuCopyOperation::buffer_to_buffer(
        GpuBufferRegion::new(&resolve, resolve_range).unwrap(),
        GpuBufferRegion::new(
            &readback,
            GpuBufferRange::new(&readback, 0, resolve_range.size()).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();

    let mut fragment = builder("timing");
    for resource in [
        GpuResourceRef::QuerySet(queries),
        GpuResourceRef::Buffer(resolve),
        GpuResourceRef::Buffer(readback),
        GpuResourceRef::Texture(target),
        GpuResourceRef::TextureView(target_view),
    ] {
        fragment.declare_resource(resource).unwrap();
    }
    fragment
        .add_node(
            label("write timestamps"),
            render,
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::GraphicsRequired,
            provenance("write timestamps"),
        )
        .unwrap();
    fragment
        .add_node(
            label("resolve timestamps"),
            GpuWorkOperation::Resolve(query_resolve),
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::TransferPreferred,
            provenance("resolve timestamps"),
        )
        .unwrap();
    fragment
        .add_node(
            label("copy readback"),
            GpuWorkOperation::Copy(copy),
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::TransferPreferred,
            provenance("copy readback"),
        )
        .unwrap();
    let fragment = fragment.finish().unwrap();
    let graph = GpuPreparedWorkGraph::prepare(label("timing graph"), [fragment.clone()]).unwrap();
    let repeated = GpuPreparedWorkGraph::prepare(label("timing graph"), [fragment]).unwrap();
    assert_eq!(graph.nodes(), repeated.nodes());
    assert_eq!(graph.topological_order(), repeated.topological_order());
    assert_eq!(graph.dependencies(), repeated.dependencies());
    assert_eq!(graph.initialization(), repeated.initialization());
    assert_eq!(graph.requirements(), repeated.requirements());
    assert_eq!(graph.outputs(), repeated.outputs());
    assert_eq!(graph.diagnostics(), repeated.diagnostics());
    assert_eq!(
        graph
            .nodes()
            .iter()
            .map(GpuPreparedWorkNode::id)
            .collect::<Vec<_>>(),
        graph.topological_order()
    );
    assert_eq!(
        graph
            .nodes()
            .iter()
            .map(|node| node.node().label().as_str())
            .collect::<Vec<_>>(),
        vec!["write timestamps", "resolve timestamps", "copy readback"]
    );
    assert!(graph.nodes().iter().all(|node| {
        node.node()
            .accesses()
            .windows(2)
            .all(|pair| pair[0] < pair[1])
    }));
    assert!(graph.dependencies().iter().any(|dependency| {
        dependency.before().local_node() == 1 && dependency.after().local_node() == 2
    }));
    assert!(graph.dependencies().iter().any(|dependency| {
        dependency.before().local_node() == 2 && dependency.after().local_node() == 3
    }));
    assert_eq!(
        graph
            .requirements()
            .get(GpuCapabilityFeature::TimestampQuery),
        Some(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::TimestampQuery
        ))
    );
    assert_eq!(
        graph
            .requirements()
            .iter()
            .map(GpuCapabilityRequirement::feature)
            .collect::<Vec<_>>(),
        vec![
            GpuCapabilityFeature::RenderPipeline,
            GpuCapabilityFeature::Copy,
            GpuCapabilityFeature::TimestampQuery,
        ]
    );
}

#[test]
fn generic_timestamp_access_does_not_initialize_query_state() {
    let mut allocator = allocator();
    let queries = allocator
        .allocate_query_set_handle(
            GpuQuerySetDescriptor::new(common("generic timestamps"), GpuQueryKind::Timestamp, 1)
                .unwrap(),
        )
        .unwrap();
    let destination = buffer(
        &mut allocator,
        "generic timestamp resolve",
        GpuBufferInitialization::Uninitialized,
        [GpuBufferUsage::QueryResolve],
    );
    let range = GpuQueryRange::whole(&queries).unwrap();
    let resolve = GpuQueryResolveOperation::new(&queries, range, &destination, 0).unwrap();
    let mut fragment = builder("generic timestamp");
    for resource in [
        GpuResourceRef::QuerySet(queries.clone()),
        GpuResourceRef::Buffer(destination),
    ] {
        fragment.declare_resource(resource).unwrap();
    }
    add_compute(
        &mut fragment,
        "generic timestamp write",
        [GpuResourceAccess::Query(
            GpuQueryAccess::new(&queries, range, GpuQueryAccessKind::WriteTimestamp).unwrap(),
        )],
    );
    fragment
        .add_node(
            label("resolve generic timestamp"),
            GpuWorkOperation::Resolve(resolve),
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::TransferPreferred,
            provenance("resolve generic timestamp"),
        )
        .unwrap();
    assert_eq!(
        GpuPreparedWorkGraph::prepare(
            label("generic timestamp graph"),
            [fragment.finish().unwrap()]
        )
        .unwrap_err()
        .cause(),
        GpuWorkGraphCause::ReadBeforeInitialization
    );
}

#[test]
fn buffer_zero_initializes_only_its_checked_range() {
    let mut allocator = allocator();
    let buffer = buffer(
        &mut allocator,
        "zero region",
        GpuBufferInitialization::Uninitialized,
        [GpuBufferUsage::CopyDestination, GpuBufferUsage::Storage],
    );
    let zeroed = GpuBufferRange::new(&buffer, 8, 16).unwrap();
    let clear =
        GpuClearOperation::buffer_zero(GpuBufferRegion::new(&buffer, zeroed).unwrap()).unwrap();
    let mut fragment = builder("buffer zero");
    fragment
        .declare_resource(GpuResourceRef::Buffer(buffer.clone()))
        .unwrap();
    fragment
        .add_node(
            label("zero"),
            GpuWorkOperation::Clear(clear),
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::TransferPreferred,
            provenance("zero"),
        )
        .unwrap();
    add_compute(
        &mut fragment,
        "read zeroed",
        [buffer_access(
            &buffer,
            zeroed,
            GpuBufferAccessKind::StorageRead,
        )],
    );
    let graph =
        GpuPreparedWorkGraph::prepare(label("buffer zero graph"), [fragment.finish().unwrap()])
            .unwrap();
    let summary = graph
        .initialization()
        .iter()
        .find(|summary| summary.resource().diagnostic_identity() == buffer.diagnostic_identity())
        .unwrap();
    let final_coverage = summary.final_coverage().unwrap().buffer_values().unwrap();
    assert_eq!(final_coverage, [GpuBufferCoverage::dense(zeroed)]);
}

#[test]
fn buffer_to_buffer_copy_requires_and_initializes_exact_interior_ranges() {
    let mut allocator = allocator();
    let source = buffer(
        &mut allocator,
        "interior copy source",
        GpuBufferInitialization::Uninitialized,
        [GpuBufferUsage::CopySource, GpuBufferUsage::Storage],
    );
    let destination = buffer(
        &mut allocator,
        "interior copy destination",
        GpuBufferInitialization::Uninitialized,
        [GpuBufferUsage::CopyDestination, GpuBufferUsage::Storage],
    );
    let source_range = GpuBufferRange::new(&source, 8, 16).unwrap();
    let destination_range = GpuBufferRange::new(&destination, 32, 16).unwrap();
    let source_input = GpuWorkResourceInput::new(
        GpuResourceRef::Buffer(source.clone()),
        GpuInitialCoverage::buffer(&source, [GpuBufferCoverage::dense(source_range)]).unwrap(),
        provenance("interior copy source input"),
    )
    .unwrap();
    let copy = GpuCopyOperation::buffer_to_buffer(
        GpuBufferRegion::new(&source, source_range).unwrap(),
        GpuBufferRegion::new(&destination, destination_range).unwrap(),
    )
    .unwrap();
    let make_fragment = |source_read: Option<GpuBufferRange>, destination_read| {
        let mut fragment = builder("interior buffer copy");
        for resource in [
            GpuResourceRef::Buffer(source.clone()),
            GpuResourceRef::Buffer(destination.clone()),
        ] {
            fragment.declare_resource(resource).unwrap();
        }
        fragment.add_input(source_input.clone()).unwrap();
        if let Some(range) = source_read {
            add_compute(
                &mut fragment,
                "read source",
                [buffer_access(
                    &source,
                    range,
                    GpuBufferAccessKind::StorageRead,
                )],
            );
        }
        fragment
            .add_node(
                label("copy interior range"),
                GpuWorkOperation::Copy(copy.clone()),
                [],
                GpuCapabilityRequirements::new(),
                GpuExecutionPreference::TransferPreferred,
                provenance("copy interior range"),
            )
            .unwrap();
        add_compute(
            &mut fragment,
            "read destination",
            [buffer_access(
                &destination,
                destination_read,
                GpuBufferAccessKind::StorageRead,
            )],
        );
        fragment.finish().unwrap()
    };

    let prepared = GpuPreparedWorkGraph::prepare(
        label("exact interior buffer copy"),
        [make_fragment(None, destination_range)],
    )
    .unwrap();
    let destination_coverage = prepared
        .initialization()
        .iter()
        .find(|summary| {
            summary.resource().diagnostic_identity() == destination.diagnostic_identity()
        })
        .unwrap()
        .final_coverage()
        .unwrap()
        .buffer_values()
        .unwrap();
    assert_eq!(
        destination_coverage,
        [GpuBufferCoverage::dense(destination_range)]
    );

    for fragment in [
        make_fragment(
            Some(GpuBufferRange::new(&source, 7, 18).unwrap()),
            destination_range,
        ),
        make_fragment(None, GpuBufferRange::new(&destination, 31, 18).unwrap()),
    ] {
        assert_eq!(
            GpuPreparedWorkGraph::prepare(label("neighbor interior buffer copy"), [fragment])
                .unwrap_err()
                .cause(),
            GpuWorkGraphCause::ReadBeforeInitialization
        );
    }
}

#[test]
fn query_resolve_uses_exact_input_indices_and_interior_destination_bytes() {
    let mut allocator = allocator();
    let queries = allocator
        .allocate_query_set_handle(
            GpuQuerySetDescriptor::new(
                common("isolated query resolve"),
                GpuQueryKind::Timestamp,
                8,
            )
            .unwrap(),
        )
        .unwrap();
    let destination = buffer(
        &mut allocator,
        "isolated query destination",
        GpuBufferInitialization::Uninitialized,
        [GpuBufferUsage::QueryResolve, GpuBufferUsage::Storage],
    );
    let query_range = GpuQueryRange::new(&queries, 2, 3).unwrap();
    let destination_range = GpuBufferRange::new(&destination, 16, 24).unwrap();
    let query_input = GpuWorkResourceInput::new(
        GpuResourceRef::QuerySet(queries.clone()),
        GpuInitialCoverage::query_ranges(&queries, [query_range]).unwrap(),
        provenance("isolated query input"),
    )
    .unwrap();
    let resolve = GpuQueryResolveOperation::new(&queries, query_range, &destination, 16).unwrap();
    assert_eq!(resolve.destination_range(), destination_range);
    let make_fragment = |query_read: Option<GpuQueryRange>, destination_read| {
        let mut fragment = builder("isolated query resolve");
        for resource in [
            GpuResourceRef::QuerySet(queries.clone()),
            GpuResourceRef::Buffer(destination.clone()),
        ] {
            fragment.declare_resource(resource).unwrap();
        }
        fragment.add_input(query_input.clone()).unwrap();
        if let Some(range) = query_read {
            add_compute(
                &mut fragment,
                "read neighboring queries",
                [GpuResourceAccess::Query(
                    GpuQueryAccess::new(&queries, range, GpuQueryAccessKind::ResolveSource)
                        .unwrap(),
                )],
            );
        }
        fragment
            .add_node(
                label("resolve explicit query input"),
                GpuWorkOperation::Resolve(resolve.clone()),
                [],
                GpuCapabilityRequirements::new(),
                GpuExecutionPreference::TransferPreferred,
                provenance("resolve explicit query input"),
            )
            .unwrap();
        add_compute(
            &mut fragment,
            "read resolved bytes",
            [buffer_access(
                &destination,
                destination_read,
                GpuBufferAccessKind::StorageRead,
            )],
        );
        fragment.finish().unwrap()
    };

    let prepared = GpuPreparedWorkGraph::prepare(
        label("exact isolated query resolve"),
        [make_fragment(None, destination_range)],
    )
    .unwrap();
    let query_coverage = prepared
        .initialization()
        .iter()
        .find(|summary| summary.resource().diagnostic_identity() == queries.diagnostic_identity())
        .unwrap()
        .final_coverage()
        .unwrap()
        .query_range_values()
        .unwrap();
    assert_eq!(query_coverage, [query_range]);
    let destination_coverage = prepared
        .initialization()
        .iter()
        .find(|summary| {
            summary.resource().diagnostic_identity() == destination.diagnostic_identity()
        })
        .unwrap()
        .final_coverage()
        .unwrap()
        .buffer_values()
        .unwrap();
    assert_eq!(
        destination_coverage,
        [GpuBufferCoverage::dense(destination_range)]
    );

    for fragment in [
        make_fragment(
            Some(GpuQueryRange::new(&queries, 1, 5).unwrap()),
            destination_range,
        ),
        make_fragment(None, GpuBufferRange::new(&destination, 15, 26).unwrap()),
    ] {
        assert_eq!(
            GpuPreparedWorkGraph::prepare(label("neighbor isolated query resolve"), [fragment])
                .unwrap_err()
                .cause(),
            GpuWorkGraphCause::ReadBeforeInitialization
        );
    }
}

#[test]
fn operation_effects_require_their_checked_write_role() {
    let mut allocator = allocator();
    let buffer = buffer(
        &mut allocator,
        "effect role",
        GpuBufferInitialization::Uninitialized,
        [GpuBufferUsage::CopyDestination, GpuBufferUsage::Storage],
    );
    let range = GpuBufferRange::new(&buffer, 0, 16).unwrap();
    let clear =
        GpuClearOperation::buffer_zero(GpuBufferRegion::new(&buffer, range).unwrap()).unwrap();
    let mut fragment = builder("effect role");
    fragment
        .declare_resource(GpuResourceRef::Buffer(buffer.clone()))
        .unwrap();
    fragment
        .add_node(
            label("clear"),
            GpuWorkOperation::Clear(clear),
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::TransferPreferred,
            provenance("clear"),
        )
        .unwrap();
    let mut fragment = fragment.finish().unwrap();
    fragment.nodes[0].accesses = vec![buffer_access(
        &buffer,
        range,
        GpuBufferAccessKind::StorageWrite,
    )];
    assert_eq!(
        GpuPreparedWorkGraph::prepare(label("effect role graph"), [fragment])
            .unwrap_err()
            .cause(),
        GpuWorkGraphCause::OperationAccessContradiction
    );
}

#[test]
fn generic_attachment_shaped_access_cannot_discard_initialization() {
    let mut allocator = allocator();
    let texture = texture(
        &mut allocator,
        "generic discard",
        GpuTextureInitialization::Zeroed,
        1,
        1,
        [
            GpuTextureUsage::ColorAttachment,
            GpuTextureUsage::CopyDestination,
            GpuTextureUsage::Sampled,
        ],
    );
    let range = GpuTextureSubresourceRange::whole(&texture).unwrap();
    let mut fragment = builder("generic discard");
    fragment
        .declare_resource(GpuResourceRef::Texture(texture.clone()))
        .unwrap();
    add_compute(
        &mut fragment,
        "generic discard shaped access",
        [texture_access(
            &texture,
            range,
            GpuTextureAccessKind::ColorAttachment {
                load_kind: GpuAttachmentLoadKind::Clear,
                store: GpuAttachmentStore::Discard,
            },
        )],
    );
    add_compute(
        &mut fragment,
        "sample retained texture",
        [texture_access(
            &texture,
            range,
            GpuTextureAccessKind::SampledRead,
        )],
    );
    assert!(
        GpuPreparedWorkGraph::prepare(label("generic discard graph"), [fragment.finish().unwrap()])
            .is_ok()
    );
}

#[test]
fn complete_d2_array_copy_initializes_only_the_selected_nonzero_layers() {
    let mut allocator = allocator();
    let source = texture(
        &mut allocator,
        "array copy source",
        GpuTextureInitialization::Zeroed,
        1,
        4,
        [
            GpuTextureUsage::CopySource,
            GpuTextureUsage::CopyDestination,
        ],
    );
    let destination = texture(
        &mut allocator,
        "array copy destination",
        GpuTextureInitialization::Uninitialized,
        1,
        4,
        [GpuTextureUsage::CopyDestination, GpuTextureUsage::Sampled],
    );
    let source_region = GpuTextureCopyRegion::new(
        &source,
        0,
        GpuTextureOrigin::new(0, 0, 1),
        GpuTextureAspect::Color,
        GpuCopyExtent::new(8, 8, 2).unwrap(),
    )
    .unwrap();
    let destination_region = GpuTextureCopyRegion::new(
        &destination,
        0,
        GpuTextureOrigin::new(0, 0, 1),
        GpuTextureAspect::Color,
        GpuCopyExtent::new(8, 8, 2).unwrap(),
    )
    .unwrap();
    let copy = GpuCopyOperation::texture_to_texture(source_region, destination_region).unwrap();
    let selected = GpuTextureSubresourceRange::new(
        destination.descriptor().common().label(),
        0,
        1,
        1,
        2,
        GpuTextureAspect::Color,
    )
    .unwrap();
    let make_fragment = |read| {
        let mut fragment = builder("array copy");
        for resource in [
            GpuResourceRef::Texture(source.clone()),
            GpuResourceRef::Texture(destination.clone()),
        ] {
            fragment.declare_resource(resource).unwrap();
        }
        fragment
            .add_node(
                label("copy selected layers"),
                GpuWorkOperation::Copy(copy.clone()),
                [],
                GpuCapabilityRequirements::new(),
                GpuExecutionPreference::TransferPreferred,
                provenance("copy selected layers"),
            )
            .unwrap();
        add_compute(
            &mut fragment,
            "sample destination",
            [texture_access(
                &destination,
                read,
                GpuTextureAccessKind::SampledRead,
            )],
        );
        fragment.finish().unwrap()
    };
    assert!(
        GpuPreparedWorkGraph::prepare(
            label("selected array copy graph"),
            [make_fragment(selected)]
        )
        .is_ok()
    );
    assert_eq!(
        GpuPreparedWorkGraph::prepare(
            label("whole array copy graph"),
            [make_fragment(
                GpuTextureSubresourceRange::whole(&destination).unwrap()
            )]
        )
        .unwrap_err()
        .cause(),
        GpuWorkGraphCause::ReadBeforeInitialization
    );
}

#[test]
fn d3_texture_copy_initializes_a_mip_only_for_the_complete_volume() {
    let mut allocator = allocator();
    let source_label = label("d3 copy source");
    let source = allocator
        .allocate_texture_handle(
            GpuTextureDescriptor::new(
                common("d3 copy source"),
                GpuTextureDimension::D3,
                GpuTextureExtent::new(&source_label, GpuTextureDimension::D3, 8, 8, 4).unwrap(),
                1,
                1,
                GpuTextureFormat::Rgba8Unorm,
                GpuTextureUsages::new(
                    &source_label,
                    [
                        GpuTextureUsage::CopySource,
                        GpuTextureUsage::CopyDestination,
                    ],
                )
                .unwrap(),
                GpuTextureInitialization::Zeroed,
            )
            .unwrap(),
        )
        .unwrap();
    let destination_label = label("d3 copy destination");
    let destination = allocator
        .allocate_texture_handle(
            GpuTextureDescriptor::new(
                common("d3 copy destination"),
                GpuTextureDimension::D3,
                GpuTextureExtent::new(&destination_label, GpuTextureDimension::D3, 8, 8, 4)
                    .unwrap(),
                1,
                1,
                GpuTextureFormat::Rgba8Unorm,
                GpuTextureUsages::new(
                    &destination_label,
                    [GpuTextureUsage::CopyDestination, GpuTextureUsage::Sampled],
                )
                .unwrap(),
                GpuTextureInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();
    let make_fragment = |depth| {
        let extent = GpuCopyExtent::new(8, 8, depth).unwrap();
        let copy = GpuCopyOperation::texture_to_texture(
            GpuTextureCopyRegion::new(
                &source,
                0,
                GpuTextureOrigin::new(0, 0, 0),
                GpuTextureAspect::Color,
                extent,
            )
            .unwrap(),
            GpuTextureCopyRegion::new(
                &destination,
                0,
                GpuTextureOrigin::new(0, 0, 0),
                GpuTextureAspect::Color,
                extent,
            )
            .unwrap(),
        )
        .unwrap();
        let mut fragment = builder("d3 texture copy");
        for resource in [
            GpuResourceRef::Texture(source.clone()),
            GpuResourceRef::Texture(destination.clone()),
        ] {
            fragment.declare_resource(resource).unwrap();
        }
        fragment
            .add_node(
                label("copy d3 volume"),
                GpuWorkOperation::Copy(copy),
                [],
                GpuCapabilityRequirements::new(),
                GpuExecutionPreference::TransferPreferred,
                provenance("copy d3 volume"),
            )
            .unwrap();
        add_compute(
            &mut fragment,
            "sample d3 destination",
            [texture_access(
                &destination,
                GpuTextureSubresourceRange::whole(&destination).unwrap(),
                GpuTextureAccessKind::SampledRead,
            )],
        );
        fragment.finish().unwrap()
    };

    assert_eq!(
        GpuPreparedWorkGraph::prepare(label("partial d3 copy"), [make_fragment(2)])
            .unwrap_err()
            .cause(),
        GpuWorkGraphCause::ReadBeforeInitialization
    );
    let prepared =
        GpuPreparedWorkGraph::prepare(label("complete d3 copy"), [make_fragment(4)]).unwrap();
    let destination_coverage = prepared
        .initialization()
        .iter()
        .find(|summary| {
            summary.resource().diagnostic_identity() == destination.diagnostic_identity()
        })
        .unwrap()
        .final_coverage()
        .unwrap()
        .texture_subresource_values()
        .unwrap();
    assert_eq!(
        destination_coverage,
        [GpuTextureSubresourceRange::whole(&destination).unwrap()]
    );
}

#[test]
fn buffer_effect_union_and_prepared_publication_are_canonical_and_deterministic() {
    let mut allocator = allocator();
    let buffer = buffer(
        &mut allocator,
        "canonical effects",
        GpuBufferInitialization::Uninitialized,
        [GpuBufferUsage::CopyDestination],
    );
    let make_fragment = || {
        let mut fragment = builder("canonical effects");
        fragment
            .declare_resource(GpuResourceRef::Buffer(buffer.clone()))
            .unwrap();
        for (name, offset) in [("clear second", 16), ("clear first", 0)] {
            let range = GpuBufferRange::new(&buffer, offset, 16).unwrap();
            fragment
                .add_node(
                    label(name),
                    GpuWorkOperation::Clear(
                        GpuClearOperation::buffer_zero(
                            GpuBufferRegion::new(&buffer, range).unwrap(),
                        )
                        .unwrap(),
                    ),
                    [],
                    GpuCapabilityRequirements::new(),
                    GpuExecutionPreference::TransferPreferred,
                    provenance(name),
                )
                .unwrap();
        }
        fragment.finish().unwrap()
    };
    let first =
        GpuPreparedWorkGraph::prepare(label("canonical effects graph"), [make_fragment()]).unwrap();
    let second =
        GpuPreparedWorkGraph::prepare(label("canonical effects graph"), [make_fragment()]).unwrap();
    assert_eq!(first.initialization(), second.initialization());
    let coverage = first
        .initialization()
        .iter()
        .find(|summary| summary.resource().diagnostic_identity() == buffer.diagnostic_identity())
        .unwrap()
        .final_coverage()
        .unwrap()
        .buffer_values()
        .unwrap();
    assert_eq!(
        coverage,
        [GpuBufferCoverage::dense(
            GpuBufferRange::new(&buffer, 0, 32).unwrap()
        )]
    );
}

#[test]
fn strided_effect_union_and_prepared_publication_retain_the_canonical_superset() {
    let mut allocator = allocator();
    let destination_label = label("strided union destination");
    let destination = allocator
        .allocate_buffer_handle(
            GpuBufferDescriptor::new(
                common("strided union destination"),
                512,
                GpuBufferUsages::new(&destination_label, [GpuBufferUsage::CopyDestination])
                    .unwrap(),
                GpuBufferInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();
    let source = texture(
        &mut allocator,
        "strided union source",
        GpuTextureInitialization::Zeroed,
        1,
        1,
        [
            GpuTextureUsage::CopySource,
            GpuTextureUsage::CopyDestination,
        ],
    );
    let copy = |height| {
        GpuCopyOperation::texture_to_buffer(
            GpuTextureCopyRegion::new(
                &source,
                0,
                GpuTextureOrigin::new(0, 0, 0),
                GpuTextureAspect::Color,
                GpuCopyExtent::new(8, height, 1).unwrap(),
            )
            .unwrap(),
            GpuBufferTextureLayout::new(&destination, 0, 64, 0).unwrap(),
        )
        .unwrap()
    };
    let mut fragment = builder("strided union");
    for resource in [
        GpuResourceRef::Texture(source.clone()),
        GpuResourceRef::Buffer(destination.clone()),
    ] {
        fragment.declare_resource(resource).unwrap();
    }
    for (name, height) in [("copy complete rows", 8), ("copy contained rows", 4)] {
        fragment
            .add_node(
                label(name),
                GpuWorkOperation::Copy(copy(height)),
                [],
                GpuCapabilityRequirements::new(),
                GpuExecutionPreference::TransferPreferred,
                provenance(name),
            )
            .unwrap();
    }
    let fragment = fragment.finish().unwrap();
    let first =
        GpuPreparedWorkGraph::prepare(label("strided union graph"), [fragment.clone()]).unwrap();
    let second = GpuPreparedWorkGraph::prepare(label("strided union graph"), [fragment]).unwrap();
    assert_eq!(first.initialization(), second.initialization());
    let coverage = first
        .initialization()
        .iter()
        .find(|summary| {
            summary.resource().diagnostic_identity() == destination.diagnostic_identity()
        })
        .unwrap()
        .final_coverage()
        .unwrap()
        .buffer_values()
        .unwrap();
    assert_eq!(
        coverage,
        [GpuBufferCoverage::strided(
            GpuBufferStridedCoverage::new(&destination, 0, 32, 64, 8, 0, 1).unwrap()
        )]
    );
}

#[test]
fn very_large_strided_dense_union_stays_compact_through_prepared_publication() {
    let mut allocator = allocator();
    let destination_label = label("large strided dense union destination");
    let destination = allocator
        .allocate_buffer_handle(
            GpuBufferDescriptor::new(
                common("large strided dense union destination"),
                64_000_000_000_000,
                GpuBufferUsages::new(&destination_label, [GpuBufferUsage::CopyDestination])
                    .unwrap(),
                GpuBufferInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();
    let segment_count = 1_000_000_u32;
    let group_count = 1_000_000_u32;
    let strided = GpuBufferStridedCoverage::new(
        &destination,
        0,
        32,
        64,
        segment_count,
        64_000_000,
        group_count,
    )
    .unwrap();
    let last_segment = u64::from(group_count - 1) * 64_000_000 + u64::from(segment_count - 1) * 64;
    let dense_segment = GpuBufferRange::new(&destination, last_segment, 32).unwrap();
    let input = GpuWorkResourceInput::new(
        GpuResourceRef::Buffer(destination.clone()),
        GpuInitialCoverage::buffer(
            &destination,
            [
                GpuBufferCoverage::strided(strided.clone()),
                GpuBufferCoverage::dense(dense_segment),
            ],
        )
        .unwrap(),
        provenance("large strided dense input"),
    )
    .unwrap();
    let mut fragment = builder("large strided dense union");
    fragment
        .declare_resource(GpuResourceRef::Buffer(destination.clone()))
        .unwrap();
    fragment.add_input(input).unwrap();
    fragment
        .add_node(
            label("clear last strided segment"),
            GpuWorkOperation::Clear(
                GpuClearOperation::buffer_zero(
                    GpuBufferRegion::new(&destination, dense_segment).unwrap(),
                )
                .unwrap(),
            ),
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::TransferPreferred,
            provenance("clear last strided segment"),
        )
        .unwrap();
    let prepared = GpuPreparedWorkGraph::prepare(
        label("large strided dense union graph"),
        [fragment.finish().unwrap()],
    )
    .unwrap();
    let coverage = prepared
        .initialization()
        .iter()
        .find(|summary| {
            summary.resource().diagnostic_identity() == destination.diagnostic_identity()
        })
        .unwrap()
        .final_coverage()
        .unwrap()
        .buffer_values()
        .unwrap();
    assert_eq!(coverage, [GpuBufferCoverage::strided(strided)]);
}

#[test]
fn padded_buffer_texture_copy_requires_and_initializes_only_logical_bytes() {
    let mut allocator = allocator();
    let source_label = label("padded source");
    let source = allocator
        .allocate_buffer_handle(
            GpuBufferDescriptor::new(
                common("padded source"),
                512,
                GpuBufferUsages::new(&source_label, [GpuBufferUsage::CopySource]).unwrap(),
                GpuBufferInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();
    let destination = texture(
        &mut allocator,
        "padded destination",
        GpuTextureInitialization::Uninitialized,
        1,
        1,
        [GpuTextureUsage::CopyDestination, GpuTextureUsage::Sampled],
    );
    let logical_rows = GpuBufferStridedCoverage::new(&source, 0, 32, 64, 8, 0, 1).unwrap();
    let input = GpuWorkResourceInput::new(
        GpuResourceRef::Buffer(source.clone()),
        GpuInitialCoverage::buffer(&source, [GpuBufferCoverage::strided(logical_rows.clone())])
            .unwrap(),
        provenance("padded source input"),
    )
    .unwrap();
    let destination_region = GpuTextureCopyRegion::new(
        &destination,
        0,
        GpuTextureOrigin::new(0, 0, 0),
        GpuTextureAspect::Color,
        GpuCopyExtent::new(8, 8, 1).unwrap(),
    )
    .unwrap();
    let copy = GpuCopyOperation::buffer_to_texture(
        GpuBufferTextureLayout::new(&source, 0, 64, 0).unwrap(),
        destination_region,
    )
    .unwrap();
    let mut fragment = builder("padded copy");
    for resource in [
        GpuResourceRef::Buffer(source.clone()),
        GpuResourceRef::Texture(destination.clone()),
    ] {
        fragment.declare_resource(resource).unwrap();
    }
    fragment.add_input(input).unwrap();
    fragment
        .add_node(
            label("copy"),
            GpuWorkOperation::Copy(copy),
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::TransferPreferred,
            provenance("copy"),
        )
        .unwrap();
    add_compute(
        &mut fragment,
        "sample copied texture",
        [texture_access(
            &destination,
            GpuTextureSubresourceRange::whole(&destination).unwrap(),
            GpuTextureAccessKind::SampledRead,
        )],
    );
    assert!(
        GpuPreparedWorkGraph::prepare(label("padded copy graph"), [fragment.finish().unwrap()])
            .is_ok()
    );

    let partial_destination = texture(
        &mut allocator,
        "partial padded destination",
        GpuTextureInitialization::Uninitialized,
        1,
        1,
        [GpuTextureUsage::CopyDestination, GpuTextureUsage::Sampled],
    );
    let partial_copy = GpuCopyOperation::buffer_to_texture(
        GpuBufferTextureLayout::new(&source, 0, 64, 0).unwrap(),
        GpuTextureCopyRegion::new(
            &partial_destination,
            0,
            GpuTextureOrigin::new(0, 0, 0),
            GpuTextureAspect::Color,
            GpuCopyExtent::new(4, 8, 1).unwrap(),
        )
        .unwrap(),
    )
    .unwrap();
    let mut partial = builder("partial padded copy");
    for resource in [
        GpuResourceRef::Buffer(source.clone()),
        GpuResourceRef::Texture(partial_destination.clone()),
    ] {
        partial.declare_resource(resource).unwrap();
    }
    partial
        .add_input(
            GpuWorkResourceInput::new(
                GpuResourceRef::Buffer(source.clone()),
                GpuInitialCoverage::buffer(&source, [GpuBufferCoverage::strided(logical_rows)])
                    .unwrap(),
                provenance("partial padded source input"),
            )
            .unwrap(),
        )
        .unwrap();
    partial
        .add_node(
            label("partial copy"),
            GpuWorkOperation::Copy(partial_copy),
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::TransferPreferred,
            provenance("partial copy"),
        )
        .unwrap();
    add_compute(
        &mut partial,
        "sample partial texture",
        [texture_access(
            &partial_destination,
            GpuTextureSubresourceRange::whole(&partial_destination).unwrap(),
            GpuTextureAccessKind::SampledRead,
        )],
    );
    assert_eq!(
        GpuPreparedWorkGraph::prepare(
            label("partial padded copy graph"),
            [partial.finish().unwrap()]
        )
        .unwrap_err()
        .cause(),
        GpuWorkGraphCause::ReadBeforeInitialization
    );
}

#[test]
fn multi_image_strided_coverage_round_trips_through_output_import_and_copy() {
    let mut allocator = allocator();
    let source = texture(
        &mut allocator,
        "multi image source",
        GpuTextureInitialization::Zeroed,
        1,
        2,
        [
            GpuTextureUsage::CopySource,
            GpuTextureUsage::CopyDestination,
        ],
    );
    let destination = texture(
        &mut allocator,
        "multi image destination",
        GpuTextureInitialization::Uninitialized,
        1,
        2,
        [GpuTextureUsage::CopyDestination],
    );
    let staging_label = label("multi image staging");
    let staging = allocator
        .allocate_buffer_handle(
            GpuBufferDescriptor::new(
                common("multi image staging"),
                1024,
                GpuBufferUsages::new(
                    &staging_label,
                    [
                        GpuBufferUsage::CopySource,
                        GpuBufferUsage::CopyDestination,
                        GpuBufferUsage::Storage,
                    ],
                )
                .unwrap(),
                GpuBufferInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();
    let extent = GpuCopyExtent::new(8, 4, 2).unwrap();
    let layout = GpuBufferTextureLayout::new(&staging, 0, 64, 6).unwrap();
    let exact = GpuBufferStridedCoverage::new(&staging, 0, 32, 64, 4, 384, 2).unwrap();
    let key = GpuExportKey::new("multi-image.staging").unwrap();

    let mut producer = builder("multi image producer");
    for resource in [
        GpuResourceRef::Texture(source.clone()),
        GpuResourceRef::Buffer(staging.clone()),
    ] {
        producer.declare_resource(resource).unwrap();
    }
    producer
        .add_node(
            label("copy texture images to staging"),
            GpuWorkOperation::Copy(
                GpuCopyOperation::texture_to_buffer(
                    GpuTextureCopyRegion::new(
                        &source,
                        0,
                        GpuTextureOrigin::new(0, 0, 0),
                        GpuTextureAspect::Color,
                        extent,
                    )
                    .unwrap(),
                    layout.clone(),
                )
                .unwrap(),
            ),
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::TransferPreferred,
            provenance("copy texture images to staging"),
        )
        .unwrap();
    producer
        .add_output(
            GpuWorkOutput::new(
                GpuExportRelationship::new(
                    GpuResourceRef::Buffer(staging.clone()),
                    key.clone(),
                    GpuResourceAccessIntent::Write,
                    provenance("multi image staging output"),
                ),
                GpuInitialCoverage::buffer(&staging, [GpuBufferCoverage::strided(exact.clone())])
                    .unwrap(),
            )
            .unwrap(),
        )
        .unwrap();
    let producer = producer.finish().unwrap();

    let make_consumer = |padding_read: Option<GpuBufferRange>| {
        let mut consumer = builder("multi image consumer");
        for resource in [
            GpuResourceRef::Buffer(staging.clone()),
            GpuResourceRef::Texture(destination.clone()),
        ] {
            consumer.declare_resource(resource).unwrap();
        }
        consumer
            .add_import(GpuWorkImport::new(
                GpuResourceRef::Buffer(staging.clone()),
                key.clone(),
                GpuResourceAccessIntent::Read,
                provenance("multi image staging import"),
            ))
            .unwrap();
        consumer
            .add_node(
                label("copy staging to texture images"),
                GpuWorkOperation::Copy(
                    GpuCopyOperation::buffer_to_texture(
                        layout.clone(),
                        GpuTextureCopyRegion::new(
                            &destination,
                            0,
                            GpuTextureOrigin::new(0, 0, 0),
                            GpuTextureAspect::Color,
                            extent,
                        )
                        .unwrap(),
                    )
                    .unwrap(),
                ),
                [],
                GpuCapabilityRequirements::new(),
                GpuExecutionPreference::TransferPreferred,
                provenance("copy staging to texture images"),
            )
            .unwrap();
        if let Some(range) = padding_read {
            add_compute(
                &mut consumer,
                "read multi image padding",
                [buffer_access(
                    &staging,
                    range,
                    GpuBufferAccessKind::StorageRead,
                )],
            );
        }
        consumer.finish().unwrap()
    };

    let prepared = GpuPreparedWorkGraph::prepare(
        label("multi image composition"),
        [make_consumer(None), producer.clone()],
    )
    .unwrap();
    let staging_coverage = prepared
        .initialization()
        .iter()
        .find(|summary| summary.resource().diagnostic_identity() == staging.diagnostic_identity())
        .unwrap()
        .final_coverage()
        .unwrap()
        .buffer_values()
        .unwrap();
    assert_eq!(
        staging_coverage,
        [GpuBufferCoverage::strided(exact.clone())]
    );
    assert!(prepared.dependencies().iter().any(|dependency| {
        dependency.reasons().iter().any(|reason| {
            matches!(reason, GpuDependencyReason::ReadAfterWrite { resource, .. }
                if *resource == staging.diagnostic_identity())
        })
    }));

    for padding in [
        GpuBufferRange::new(&staging, 32, 32).unwrap(),
        GpuBufferRange::new(&staging, 224, 160).unwrap(),
    ] {
        assert_eq!(
            GpuPreparedWorkGraph::prepare(
                label("multi image padding rejection"),
                [producer.clone(), make_consumer(Some(padding))],
            )
            .unwrap_err()
            .cause(),
            GpuWorkGraphCause::ReadBeforeInitialization
        );
    }
}

#[test]
fn explicit_graph_entry_input_initializes_only_declared_coverage() {
    let mut allocator = allocator();
    let imported_label = label("imported retained");
    let retained_label = label("owned retained");
    let commons = [
        GpuResourceCommon::imported(
            imported_label.clone(),
            GpuResourceLifetime::Retained,
            provenance("imported retained"),
        ),
        GpuResourceCommon::owned(
            retained_label.clone(),
            GpuResourceLifetime::Retained,
            GpuMemoryIntent::Device,
            GpuReconstruction::SourceBacked,
            provenance("owned retained"),
        )
        .unwrap(),
    ];
    for common in commons {
        let resource_label = common.label().clone();
        let resource = allocator
            .allocate_buffer_handle(
                GpuBufferDescriptor::new(
                    common,
                    64,
                    GpuBufferUsages::new(&resource_label, [GpuBufferUsage::Storage]).unwrap(),
                    GpuBufferInitialization::Uninitialized,
                )
                .unwrap(),
            )
            .unwrap();
        let initialized = GpuBufferRange::new(&resource, 16, 16).unwrap();
        let input = GpuWorkResourceInput::new(
            GpuResourceRef::Buffer(resource.clone()),
            GpuInitialCoverage::buffer(&resource, [GpuBufferCoverage::dense(initialized)]).unwrap(),
            provenance("external input"),
        )
        .unwrap();
        let prepare_read = |name, read_range| {
            let mut fragment = builder(name);
            fragment
                .declare_resource(GpuResourceRef::Buffer(resource.clone()))
                .unwrap();
            fragment.add_input(input.clone()).unwrap();
            add_compute(
                &mut fragment,
                "read input",
                [buffer_access(
                    &resource,
                    read_range,
                    GpuBufferAccessKind::StorageRead,
                )],
            );
            GpuPreparedWorkGraph::prepare(label(name), [fragment.finish().unwrap()])
        };
        assert!(prepare_read("initialized input", initialized).is_ok());
        assert_eq!(
            prepare_read(
                "uninitialized input",
                GpuBufferRange::new(&resource, 0, 16).unwrap(),
            )
            .unwrap_err()
            .cause(),
            GpuWorkGraphCause::ReadBeforeInitialization
        );
    }
}

#[test]
fn output_access_and_coverage_mismatches_are_structured() {
    let mut allocator = allocator();
    let resource = buffer(
        &mut allocator,
        "mismatch",
        GpuBufferInitialization::Uninitialized,
        [GpuBufferUsage::Storage, GpuBufferUsage::CopyDestination],
    );
    let written = GpuBufferRange::new(&resource, 0, 16).unwrap();
    let overstated = GpuBufferRange::new(&resource, 0, 32).unwrap();
    let make_fragment = |intent, coverage_range| {
        let mut fragment = builder("mismatch producer");
        fragment
            .declare_resource(GpuResourceRef::Buffer(resource.clone()))
            .unwrap();
        fragment
            .add_node(
                label("write"),
                GpuWorkOperation::Clear(
                    GpuClearOperation::buffer_zero(
                        GpuBufferRegion::new(&resource, written).unwrap(),
                    )
                    .unwrap(),
                ),
                [],
                GpuCapabilityRequirements::new(),
                GpuExecutionPreference::TransferPreferred,
                provenance("write"),
            )
            .unwrap();
        fragment
            .add_output(
                GpuWorkOutput::new(
                    GpuExportRelationship::new(
                        GpuResourceRef::Buffer(resource.clone()),
                        GpuExportKey::new("mismatch.output").unwrap(),
                        intent,
                        provenance("mismatch output"),
                    ),
                    GpuInitialCoverage::buffer(
                        &resource,
                        [GpuBufferCoverage::dense(coverage_range)],
                    )
                    .unwrap(),
                )
                .unwrap(),
            )
            .unwrap();
        fragment.finish().unwrap()
    };
    assert_eq!(
        GpuPreparedWorkGraph::prepare(
            label("intent mismatch"),
            [make_fragment(GpuResourceAccessIntent::Read, written)]
        )
        .unwrap_err()
        .cause(),
        GpuWorkGraphCause::ImportExportMismatch
    );
    assert_eq!(
        GpuPreparedWorkGraph::prepare(
            label("coverage mismatch"),
            [make_fragment(GpuResourceAccessIntent::Write, overstated)]
        )
        .unwrap_err()
        .cause(),
        GpuWorkGraphCause::ImportExportMismatch
    );
}

#[test]
fn multisample_resolve_initializes_destination_despite_source_discard() {
    let mut allocator = allocator();
    let resource_label = label("msaa");
    let source = allocator
        .allocate_texture_handle(
            GpuTextureDescriptor::new(
                common("msaa"),
                GpuTextureDimension::D2,
                GpuTextureExtent::new(&resource_label, GpuTextureDimension::D2, 8, 8, 1).unwrap(),
                1,
                4,
                GpuTextureFormat::Rgba8Unorm,
                GpuTextureUsages::new(&resource_label, [GpuTextureUsage::ColorAttachment]).unwrap(),
                GpuTextureInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap();
    let destination = texture(
        &mut allocator,
        "resolved",
        GpuTextureInitialization::Uninitialized,
        1,
        1,
        [GpuTextureUsage::ColorAttachment, GpuTextureUsage::Sampled],
    );
    let source_range = GpuTextureSubresourceRange::whole(&source).unwrap();
    let destination_range = GpuTextureSubresourceRange::whole(&destination).unwrap();
    let source_view = texture_view(&mut allocator, &source, "msaa view", source_range);
    let destination_view = texture_view(
        &mut allocator,
        &destination,
        "resolved view",
        destination_range,
    );
    let resolve_target = GpuMultisampleResolveTarget::new(destination_view.clone()).unwrap();
    let attachment = GpuRenderColorAttachment::new(
        source_view.clone(),
        GpuColorAttachmentLoad::Clear(GpuColorClearValue::new(0.0, 0.0, 0.0, 1.0).unwrap()),
        GpuAttachmentStore::Discard,
        Some(resolve_target),
    )
    .unwrap();
    let mut fragment = builder("resolve");
    for resource in [
        GpuResourceRef::Texture(source),
        GpuResourceRef::Texture(destination.clone()),
        GpuResourceRef::TextureView(source_view),
        GpuResourceRef::TextureView(destination_view),
    ] {
        fragment.declare_resource(resource).unwrap();
    }
    fragment
        .add_node(
            label("render resolve"),
            GpuWorkOperation::Render(
                GpuRenderOperation::new([attachment], None, [], None).unwrap(),
            ),
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::GraphicsRequired,
            provenance("render resolve"),
        )
        .unwrap();
    add_compute(
        &mut fragment,
        "sample resolve",
        [GpuResourceAccess::Texture(
            GpuTextureAccess::new(
                GpuTextureAccessResource::Texture(destination),
                destination_range,
                GpuTextureAccessKind::SampledRead,
            )
            .unwrap(),
        )],
    );
    assert!(
        GpuPreparedWorkGraph::prepare(label("resolve graph"), [fragment.finish().unwrap()]).is_ok()
    );
}
