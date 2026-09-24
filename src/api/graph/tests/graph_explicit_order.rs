use super::support::*;

fn independent_fragment(name: &str, node_name: &str) -> (GpuWorkFragment, GpuWorkNodeId) {
    let mut fragment = builder(name);
    let node = add_compute(&mut fragment, node_name, []);
    (fragment.finish().unwrap(), node)
}

fn prepared_position(graph: &GpuPreparedWorkGraph, label: &str) -> usize {
    let node = graph
        .nodes()
        .iter()
        .find(|prepared| prepared.node().label().as_str() == label)
        .unwrap()
        .id();
    graph
        .topological_order()
        .iter()
        .position(|prepared| *prepared == node)
        .unwrap()
}

fn data_fragments() -> (
    GpuWorkFragment,
    GpuWorkNodeId,
    GpuWorkFragment,
    GpuWorkNodeId,
) {
    let mut allocator = allocator();
    let shared = buffer(
        &mut allocator,
        "graph-order shared",
        GpuBufferInitialization::Uninitialized,
        [GpuBufferUsage::Storage, GpuBufferUsage::CopyDestination],
    );
    let range = GpuBufferRange::whole(&shared).unwrap();
    let export_key = GpuExportKey::new("graph-order.shared.ready").unwrap();

    let mut producer = builder("graph-order producer");
    producer
        .declare_resource(GpuResourceRef::Buffer(shared.clone()))
        .unwrap();
    let producer_node = producer
        .add_node(
            label("graph-order write"),
            GpuWorkOperation::Clear(
                GpuClearOperation::buffer_zero(GpuBufferRegion::new(&shared, range).unwrap())
                    .unwrap(),
            ),
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::TransferPreferred,
            provenance("graph-order write"),
        )
        .unwrap();
    producer
        .add_output(
            GpuWorkOutput::new(
                GpuExportRelationship::new(
                    GpuResourceRef::Buffer(shared.clone()),
                    export_key.clone(),
                    GpuResourceAccessIntent::Write,
                    provenance("graph-order output"),
                ),
                GpuInitialCoverage::buffer(&shared, [GpuBufferCoverage::dense(range)]).unwrap(),
            )
            .unwrap(),
        )
        .unwrap();

    let mut consumer = builder("graph-order consumer");
    consumer
        .declare_resource(GpuResourceRef::Buffer(shared.clone()))
        .unwrap();
    consumer
        .add_import(GpuWorkImport::new(
            GpuResourceRef::Buffer(shared.clone()),
            export_key,
            GpuResourceAccessIntent::Read,
            provenance("graph-order import"),
        ))
        .unwrap();
    let consumer_node = add_compute(
        &mut consumer,
        "graph-order read",
        [buffer_access(
            &shared,
            range,
            GpuBufferAccessKind::StorageRead,
        )],
    );

    (
        producer.finish().unwrap(),
        producer_node,
        consumer.finish().unwrap(),
        consumer_node,
    )
}

#[test]
fn graph_scope_order_is_distinct_from_fragment_local_order() {
    let mut local = builder("local order");
    let first = add_compute(&mut local, "local first", []);
    let second = add_compute(&mut local, "local second", []);
    assert_eq!(
        GpuGraphExplicitOrder::new(&first, &second, "wrong scope")
            .unwrap_err()
            .cause(),
        GpuWorkAuthoringCause::InvalidExplicitOrder
    );
    local
        .add_explicit_order(GpuExplicitOrder::new(&first, &second, "local scope").unwrap())
        .unwrap();

    let mut foreign = builder("foreign order");
    let foreign_node = add_compute(&mut foreign, "foreign node", []);
    assert_eq!(
        GpuExplicitOrder::new(&first, &foreign_node, "wrong local scope")
            .unwrap_err()
            .cause(),
        GpuWorkAuthoringCause::ForeignIdentity
    );
    GpuGraphExplicitOrder::new(&first, &foreign_node, "graph scope").unwrap();
}

#[test]
fn graph_scope_order_resolves_immutable_fragments_independent_of_input_order() {
    let (first_fragment, first) = independent_fragment("first fragment", "first node");
    let (second_fragment, second) = independent_fragment("second fragment", "second node");
    let order = GpuGraphExplicitOrder::new(&first, &second, "first before second").unwrap();

    for fragments in [
        vec![first_fragment.clone(), second_fragment.clone()],
        vec![second_fragment.clone(), first_fragment.clone()],
    ] {
        let graph = GpuPreparedWorkGraph::prepare_with_orders(
            label("graph order"),
            fragments,
            [order.clone()],
        )
        .unwrap();
        assert!(prepared_position(&graph, "first node") < prepared_position(&graph, "second node"));
        let dependency = graph
            .dependencies()
            .iter()
            .find(|dependency| {
                graph
                    .nodes()
                    .iter()
                    .find(|node| node.id() == dependency.before())
                    .is_some_and(|node| node.node().label().as_str() == "first node")
                    && graph
                        .nodes()
                        .iter()
                        .find(|node| node.id() == dependency.after())
                        .is_some_and(|node| node.node().label().as_str() == "second node")
            })
            .unwrap();
        assert_eq!(
            dependency.reasons(),
            [GpuDependencyReason::ExplicitNonData {
                reason: "first before second".to_string(),
            }]
        );
    }
}

#[test]
fn graph_scope_order_rejects_endpoint_whose_fragment_is_absent() {
    let (first_fragment, first) = independent_fragment("present fragment", "present node");
    let (_absent_fragment, absent) = independent_fragment("absent fragment", "absent node");
    let order = GpuGraphExplicitOrder::new(&first, &absent, "missing fragment").unwrap();

    assert_eq!(
        GpuPreparedWorkGraph::prepare_with_orders(
            label("missing graph-order endpoint"),
            [first_fragment],
            [order],
        )
        .unwrap_err()
        .cause(),
        GpuWorkGraphCause::ForeignIdentity
    );
}

#[test]
fn graph_scope_order_rejects_ambiguous_cloned_fragment_identity() {
    let (duplicated_fragment, duplicated_node) =
        independent_fragment("duplicated fragment", "duplicated node");
    let (other_fragment, other_node) = independent_fragment("other fragment", "other node");
    let order =
        GpuGraphExplicitOrder::new(&duplicated_node, &other_node, "ambiguous duplicate").unwrap();

    assert_eq!(
        GpuPreparedWorkGraph::prepare_with_orders(
            label("ambiguous graph-order endpoint"),
            [
                duplicated_fragment.clone(),
                duplicated_fragment,
                other_fragment,
            ],
            [order],
        )
        .unwrap_err()
        .cause(),
        GpuWorkGraphCause::UnknownIdentity
    );
}

#[test]
fn graph_scope_order_preserves_typed_data_authority() {
    let (producer, producer_node, consumer, consumer_node) = data_fragments();
    let redundant =
        GpuGraphExplicitOrder::new(&producer_node, &consumer_node, "duplicate data").unwrap();
    assert_eq!(
        GpuPreparedWorkGraph::prepare_with_orders(
            label("redundant graph order"),
            [consumer.clone(), producer.clone()],
            [redundant],
        )
        .unwrap_err()
        .cause(),
        GpuWorkGraphCause::RedundantExplicitDataOrder
    );

    let conflict =
        GpuGraphExplicitOrder::new(&consumer_node, &producer_node, "reverse data").unwrap();
    assert_eq!(
        GpuPreparedWorkGraph::prepare_with_orders(
            label("conflicting graph order"),
            [producer, consumer],
            [conflict],
        )
        .unwrap_err()
        .cause(),
        GpuWorkGraphCause::ExplicitOrderConflict
    );
}

#[test]
fn graph_scope_orders_participate_in_cross_fragment_cycle_detection() {
    let (first_fragment, first) = independent_fragment("cycle first", "cycle first node");
    let (second_fragment, second) = independent_fragment("cycle second", "cycle second node");
    let (third_fragment, third) = independent_fragment("cycle third", "cycle third node");
    let orders = [
        GpuGraphExplicitOrder::new(&first, &second, "first to second").unwrap(),
        GpuGraphExplicitOrder::new(&second, &third, "second to third").unwrap(),
        GpuGraphExplicitOrder::new(&third, &first, "third to first").unwrap(),
    ];

    assert_eq!(
        GpuPreparedWorkGraph::prepare_with_orders(
            label("cross-fragment cycle"),
            [third_fragment, first_fragment, second_fragment],
            orders,
        )
        .unwrap_err()
        .cause(),
        GpuWorkGraphCause::Cycle
    );
}

#[test]
fn graph_scope_orders_bracket_immutable_work_with_timestamp_markers() {
    let mut allocator = allocator();
    let queries = allocator
        .allocate_query_set_handle(
            GpuQuerySetDescriptor::new(
                common("graph-order timestamps"),
                GpuQueryKind::Timestamp,
                2,
            )
            .unwrap(),
        )
        .unwrap();
    let work = buffer(
        &mut allocator,
        "graph-order bounded work",
        GpuBufferInitialization::Uninitialized,
        [GpuBufferUsage::CopyDestination],
    );

    let mut start_fragment = builder("timestamp start fragment");
    start_fragment
        .declare_resource(GpuResourceRef::QuerySet(queries.clone()))
        .unwrap();
    let start = start_fragment
        .add_node(
            label("timestamp start"),
            GpuWorkOperation::TimestampMarker(
                GpuTimestampMarkerOperation::new(&queries, 0).unwrap(),
            ),
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::Automatic,
            provenance("timestamp start"),
        )
        .unwrap();

    let mut work_fragment = builder("bounded work fragment");
    work_fragment
        .declare_resource(GpuResourceRef::Buffer(work.clone()))
        .unwrap();
    let bounded = work_fragment
        .add_node(
            label("bounded clear"),
            GpuWorkOperation::Clear(
                GpuClearOperation::buffer_zero(GpuBufferRegion::whole(&work).unwrap()).unwrap(),
            ),
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::TransferPreferred,
            provenance("bounded clear"),
        )
        .unwrap();

    let mut end_fragment = builder("timestamp end fragment");
    end_fragment
        .declare_resource(GpuResourceRef::QuerySet(queries.clone()))
        .unwrap();
    let end = end_fragment
        .add_node(
            label("timestamp end"),
            GpuWorkOperation::TimestampMarker(
                GpuTimestampMarkerOperation::new(&queries, 1).unwrap(),
            ),
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::Automatic,
            provenance("timestamp end"),
        )
        .unwrap();

    let graph = GpuPreparedWorkGraph::prepare_with_orders(
        label("immutable timestamp bracket"),
        [
            end_fragment.finish().unwrap(),
            work_fragment.finish().unwrap(),
            start_fragment.finish().unwrap(),
        ],
        [
            GpuGraphExplicitOrder::new(&start, &bounded, "start before bounded work").unwrap(),
            GpuGraphExplicitOrder::new(&bounded, &end, "bounded work before end").unwrap(),
        ],
    )
    .unwrap();

    let start_position = prepared_position(&graph, "timestamp start");
    let work_position = prepared_position(&graph, "bounded clear");
    let end_position = prepared_position(&graph, "timestamp end");
    assert!(start_position < work_position);
    assert!(work_position < end_position);
    assert_eq!(
        graph
            .dependencies()
            .iter()
            .filter(|dependency| {
                dependency
                    .reasons()
                    .iter()
                    .any(|reason| matches!(reason, GpuDependencyReason::ExplicitNonData { .. }))
            })
            .count(),
        2
    );
}
