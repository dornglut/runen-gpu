use super::support::*;

fn producer_fragment(
    buffer: &GpuBufferHandle,
    key: GpuExportKey,
    range: GpuBufferRange,
) -> GpuWorkFragment {
    let mut producer = builder("producer");
    producer
        .declare_resource(GpuResourceRef::Buffer(buffer.clone()))
        .unwrap();
    producer
        .add_node(
            label("produce"),
            GpuWorkOperation::Clear(
                GpuClearOperation::buffer_zero(GpuBufferRegion::new(buffer, range).unwrap())
                    .unwrap(),
            ),
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::TransferPreferred,
            provenance("produce"),
        )
        .unwrap();
    let coverage = GpuInitialCoverage::buffer(buffer, [GpuBufferCoverage::dense(range)]).unwrap();
    producer
        .add_output(
            GpuWorkOutput::new(
                GpuExportRelationship::new(
                    GpuResourceRef::Buffer(buffer.clone()),
                    key,
                    GpuResourceAccessIntent::Write,
                    provenance("producer output"),
                ),
                coverage,
            )
            .unwrap(),
        )
        .unwrap();
    producer.finish().unwrap()
}

fn consumer_fragment(
    buffer: &GpuBufferHandle,
    key: GpuExportKey,
    range: GpuBufferRange,
) -> GpuWorkFragment {
    let mut consumer = builder("consumer");
    consumer
        .declare_resource(GpuResourceRef::Buffer(buffer.clone()))
        .unwrap();
    consumer
        .add_import(GpuWorkImport::new(
            GpuResourceRef::Buffer(buffer.clone()),
            key,
            GpuResourceAccessIntent::Read,
            provenance("consumer import"),
        ))
        .unwrap();
    add_compute(
        &mut consumer,
        "consume",
        [buffer_access(
            buffer,
            range,
            GpuBufferAccessKind::StorageRead,
        )],
    );
    consumer.finish().unwrap()
}

fn semantic_dependencies(
    graph: &GpuPreparedWorkGraph,
) -> Vec<(String, String, Vec<GpuDependencyReason>)> {
    let node_label = |id| {
        graph
            .nodes()
            .iter()
            .find(|node| node.id() == id)
            .unwrap()
            .node()
            .label()
            .as_str()
            .to_string()
    };
    graph
        .dependencies()
        .iter()
        .map(|dependency| {
            (
                node_label(dependency.before()),
                node_label(dependency.after()),
                dependency.reasons().to_vec(),
            )
        })
        .collect()
}

#[test]
fn typed_import_export_causality_overrides_fragment_input_order() {
    let mut allocator = allocator();
    let shared = buffer(
        &mut allocator,
        "shared",
        GpuBufferInitialization::Uninitialized,
        [GpuBufferUsage::Storage, GpuBufferUsage::CopyDestination],
    );
    let range = GpuBufferRange::new(&shared, 0, 32).unwrap();
    let key = GpuExportKey::new("shared.ready").unwrap();
    let producer = producer_fragment(&shared, key.clone(), range);
    let consumer = consumer_fragment(&shared, key, range);
    let producer_first =
        GpuPreparedWorkGraph::prepare(label("composition"), [producer.clone(), consumer.clone()])
            .unwrap();
    let consumer_first =
        GpuPreparedWorkGraph::prepare(label("composition"), [consumer, producer]).unwrap();
    assert_eq!(
        semantic_dependencies(&producer_first),
        semantic_dependencies(&consumer_first)
    );
    assert_eq!(
        producer_first.initialization(),
        consumer_first.initialization()
    );
    assert_eq!(producer_first.requirements(), consumer_first.requirements());
    assert_eq!(producer_first.outputs(), consumer_first.outputs());
    assert_eq!(consumer_first.topological_order()[0].fragment_ordinal(), 1);
    assert_eq!(consumer_first.topological_order()[1].fragment_ordinal(), 0);
    assert!(
        consumer_first.dependencies()[0]
            .reasons()
            .iter()
            .any(|reason| matches!(reason, GpuDependencyReason::ReadAfterWrite { .. }))
    );
}

#[test]
fn cross_fragment_conflict_without_typed_causality_fails_before_initialization() {
    let mut allocator = allocator();
    let shared = buffer(
        &mut allocator,
        "unbound shared",
        GpuBufferInitialization::Uninitialized,
        [GpuBufferUsage::Storage, GpuBufferUsage::CopyDestination],
    );
    let range = GpuBufferRange::whole(&shared).unwrap();
    let mut writer = builder("writer");
    writer
        .declare_resource(GpuResourceRef::Buffer(shared.clone()))
        .unwrap();
    add_compute(
        &mut writer,
        "write",
        [buffer_access(
            &shared,
            range,
            GpuBufferAccessKind::StorageWrite,
        )],
    );
    let mut reader = builder("reader");
    reader
        .declare_resource(GpuResourceRef::Buffer(shared.clone()))
        .unwrap();
    add_compute(
        &mut reader,
        "read",
        [buffer_access(
            &shared,
            range,
            GpuBufferAccessKind::StorageRead,
        )],
    );
    let writer = writer.finish().unwrap();
    let reader = reader.finish().unwrap();
    let writer_first =
        GpuPreparedWorkGraph::prepare(label("missing causality"), [writer.clone(), reader.clone()])
            .unwrap_err();
    let reader_first =
        GpuPreparedWorkGraph::prepare(label("missing causality"), [reader, writer]).unwrap_err();
    assert_eq!(
        writer_first.cause(),
        GpuWorkGraphCause::MissingCrossFragmentCausality
    );
    assert_eq!(writer_first.cause(), reader_first.cause());
    assert_eq!(writer_first.resource(), reader_first.resource());
}

#[test]
fn imports_reject_mismatched_resources_and_insufficient_export_coverage() {
    let mut allocator = allocator();
    let produced = buffer(
        &mut allocator,
        "produced",
        GpuBufferInitialization::Uninitialized,
        [GpuBufferUsage::Storage, GpuBufferUsage::CopyDestination],
    );
    let other = buffer(
        &mut allocator,
        "other",
        GpuBufferInitialization::Uninitialized,
        [GpuBufferUsage::Storage],
    );
    let produced_range = GpuBufferRange::new(&produced, 0, 16).unwrap();
    let key = GpuExportKey::new("produced.ready").unwrap();
    let producer = producer_fragment(&produced, key.clone(), produced_range);
    let mismatched_consumer = consumer_fragment(
        &other,
        key.clone(),
        GpuBufferRange::new(&other, 0, 16).unwrap(),
    );
    assert_eq!(
        GpuPreparedWorkGraph::prepare(
            label("mismatched resource"),
            [producer.clone(), mismatched_consumer],
        )
        .unwrap_err()
        .cause(),
        GpuWorkGraphCause::ImportExportMismatch
    );

    let oversized_consumer =
        consumer_fragment(&produced, key, GpuBufferRange::whole(&produced).unwrap());
    assert_eq!(
        GpuPreparedWorkGraph::prepare(
            label("insufficient coverage"),
            [producer, oversized_consumer],
        )
        .unwrap_err()
        .cause(),
        GpuWorkGraphCause::ReadBeforeInitialization
    );
}

#[test]
fn duplicate_export_keys_and_ambiguous_writers_are_rejected() {
    let mut allocator = allocator();
    let shared = buffer(
        &mut allocator,
        "multi producer",
        GpuBufferInitialization::Uninitialized,
        [GpuBufferUsage::Storage, GpuBufferUsage::CopyDestination],
    );
    let range = GpuBufferRange::whole(&shared).unwrap();
    let duplicate = GpuExportKey::new("duplicate").unwrap();
    let first = producer_fragment(&shared, duplicate.clone(), range);
    let second = producer_fragment(&shared, duplicate, range);
    let error = GpuPreparedWorkGraph::prepare(label("duplicates"), [first, second]).unwrap_err();
    assert_eq!(error.cause(), GpuWorkGraphCause::DuplicateExportKey);

    let first = producer_fragment(&shared, GpuExportKey::new("first").unwrap(), range);
    let second = producer_fragment(&shared, GpuExportKey::new("second").unwrap(), range);
    let error = GpuPreparedWorkGraph::prepare(label("ambiguous"), [first, second]).unwrap_err();
    assert_eq!(error.cause(), GpuWorkGraphCause::AmbiguousWriter);
}

#[test]
fn explicit_non_data_order_succeeds_and_rejects_duplicate_or_unknown_endpoints() {
    let mut ordered = builder("ordered");
    let first = add_compute(&mut ordered, "first", []);
    let second = add_compute(&mut ordered, "second", []);
    ordered
        .add_explicit_order(GpuExplicitOrder::new(&first, &second, "phase order").unwrap())
        .unwrap();
    let graph =
        GpuPreparedWorkGraph::prepare(label("ordered graph"), [ordered.finish().unwrap()]).unwrap();
    assert_eq!(graph.dependencies().len(), 1);
    assert_eq!(
        graph.dependencies()[0].reasons(),
        [GpuDependencyReason::ExplicitNonData {
            reason: "phase order".to_string(),
        }]
    );

    let mut duplicate = builder("duplicate");
    let first = add_compute(&mut duplicate, "first", []);
    let second = add_compute(&mut duplicate, "second", []);
    let order = GpuExplicitOrder::new(&first, &second, "one edge").unwrap();
    duplicate.add_explicit_order(order.clone()).unwrap();
    assert_eq!(
        duplicate.add_explicit_order(order).unwrap_err().cause(),
        GpuWorkAuthoringCause::DuplicateExplicitOrder
    );

    let mut missing = builder("missing endpoint");
    let first = add_compute(&mut missing, "first", []);
    let second = add_compute(&mut missing, "second", []);
    missing
        .add_explicit_order(GpuExplicitOrder::new(&first, &second, "missing").unwrap())
        .unwrap();
    let mut missing = missing.finish().unwrap();
    missing.nodes.pop();
    assert_eq!(
        GpuPreparedWorkGraph::prepare(label("missing endpoint graph"), [missing])
            .unwrap_err()
            .cause(),
        GpuWorkGraphCause::UnknownIdentity
    );
}

#[test]
fn explicit_order_rejects_redundancy_conflict_and_cycles() {
    let mut allocator = allocator();
    let resource = buffer(
        &mut allocator,
        "explicit",
        GpuBufferInitialization::Zeroed,
        [GpuBufferUsage::Storage],
    );
    let range = GpuBufferRange::whole(&resource).unwrap();

    let data_fragment = |reverse: bool| {
        let mut fragment = builder(if reverse { "conflict" } else { "redundant" });
        fragment
            .declare_resource(GpuResourceRef::Buffer(resource.clone()))
            .unwrap();
        let write = add_compute(
            &mut fragment,
            "write",
            [buffer_access(
                &resource,
                range,
                GpuBufferAccessKind::StorageWrite,
            )],
        );
        let read = add_compute(
            &mut fragment,
            "read",
            [buffer_access(
                &resource,
                range,
                GpuBufferAccessKind::StorageRead,
            )],
        );
        let order = if reverse {
            GpuExplicitOrder::new(&read, &write, "reverse data").unwrap()
        } else {
            GpuExplicitOrder::new(&write, &read, "duplicate data").unwrap()
        };
        fragment.add_explicit_order(order).unwrap();
        fragment.finish().unwrap()
    };
    assert_eq!(
        GpuPreparedWorkGraph::prepare(label("redundant graph"), [data_fragment(false)])
            .unwrap_err()
            .cause(),
        GpuWorkGraphCause::RedundantExplicitDataOrder
    );
    assert_eq!(
        GpuPreparedWorkGraph::prepare(label("conflict graph"), [data_fragment(true)])
            .unwrap_err()
            .cause(),
        GpuWorkGraphCause::ExplicitOrderConflict
    );

    let mut cycle = builder("cycle");
    let first = add_compute(&mut cycle, "first", []);
    let second = add_compute(&mut cycle, "second", []);
    let third = add_compute(&mut cycle, "third", []);
    for order in [
        GpuExplicitOrder::new(&first, &second, "one").unwrap(),
        GpuExplicitOrder::new(&second, &third, "two").unwrap(),
        GpuExplicitOrder::new(&third, &first, "three").unwrap(),
    ] {
        cycle.add_explicit_order(order).unwrap();
    }
    assert_eq!(
        GpuPreparedWorkGraph::prepare(label("cycle graph"), [cycle.finish().unwrap()])
            .unwrap_err()
            .cause(),
        GpuWorkGraphCause::Cycle
    );
}

#[test]
fn foreign_node_identity_and_capability_contradiction_fail_structurally() {
    let mut first = builder("first");
    let first_node = add_compute(&mut first, "first node", []);
    let mut second = builder("second");
    let second_node = add_compute(&mut second, "second node", []);
    assert_eq!(
        GpuExplicitOrder::new(&first_node, &second_node, "foreign")
            .unwrap_err()
            .cause(),
        GpuWorkAuthoringCause::ForeignIdentity
    );

    let mut requirements = GpuCapabilityRequirements::new();
    requirements
        .insert(GpuCapabilityRequirement::Disabled(
            GpuCapabilityFeature::Compute,
        ))
        .unwrap();
    let error = first
        .add_node(
            label("disabled compute"),
            compute_operation(),
            [],
            requirements,
            GpuExecutionPreference::Automatic,
            provenance("disabled compute"),
        )
        .unwrap_err();
    assert_eq!(
        error.cause(),
        GpuWorkAuthoringCause::MechanicalCapabilityContradiction
    );
}

#[test]
fn prepared_graph_rejects_a_foreign_resource_identity() {
    let mut allocator = allocator();
    let declared = buffer(
        &mut allocator,
        "declared",
        GpuBufferInitialization::Zeroed,
        [GpuBufferUsage::Storage],
    );
    let foreign = buffer(
        &mut allocator,
        "foreign",
        GpuBufferInitialization::Zeroed,
        [GpuBufferUsage::Storage],
    );
    let mut fragment = builder("foreign resource");
    fragment
        .declare_resource(GpuResourceRef::Buffer(declared.clone()))
        .unwrap();
    add_compute(
        &mut fragment,
        "read",
        [buffer_access(
            &declared,
            GpuBufferRange::whole(&declared).unwrap(),
            GpuBufferAccessKind::StorageRead,
        )],
    );
    let mut fragment = fragment.finish().unwrap();
    fragment.nodes[0].accesses = vec![buffer_access(
        &foreign,
        GpuBufferRange::whole(&foreign).unwrap(),
        GpuBufferAccessKind::StorageRead,
    )];
    let error =
        GpuPreparedWorkGraph::prepare(label("foreign resource graph"), [fragment]).unwrap_err();
    assert_eq!(error.cause(), GpuWorkGraphCause::UnknownIdentity);
    assert_eq!(error.resource(), Some(foreign.diagnostic_identity()));
}

#[test]
fn independent_ready_nodes_have_deterministic_inspection_order() {
    let mut first = builder("first fragment");
    add_compute(&mut first, "first one", []);
    add_compute(&mut first, "first two", []);
    let mut second = builder("second fragment");
    add_compute(&mut second, "second one", []);
    let graph = GpuPreparedWorkGraph::prepare(
        label("deterministic"),
        [first.finish().unwrap(), second.finish().unwrap()],
    )
    .unwrap();
    assert_eq!(
        graph
            .topological_order()
            .iter()
            .map(|id| (id.fragment_ordinal(), id.local_node()))
            .collect::<Vec<_>>(),
        vec![(0, 1), (0, 2), (1, 1)]
    );
}
