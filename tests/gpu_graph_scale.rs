//! G6-S01 CPU-side characterization for canonical RunenGPU graph preparation.
//!
//! This is evidence, not a performance budget. Structural assertions are normative; hosted-CI
//! elapsed times are retained only as characterization data.

use runen_gpu::*;
use serde_json::{Value, json};
use std::path::PathBuf;
use std::time::Instant;

const TIERS: [usize; 3] = [8, 80, 800];
const WARMUP_SAMPLES: usize = 2;
const MEASURED_SAMPLES: usize = 5;
const BUFFER_BYTES: u64 = 16;
const REPORT_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Topology {
    SerialCopyChain,
    DisjointClears,
}

impl Topology {
    const fn key(self) -> &'static str {
        match self {
            Self::SerialCopyChain => "serial-copy-chain",
            Self::DisjointClears => "disjoint-clears",
        }
    }

    const fn expected_access_count(self, nodes: usize) -> usize {
        match self {
            Self::SerialCopyChain => 2 * nodes - 1,
            Self::DisjointClears => nodes,
        }
    }

    const fn expected_dependency_count(self, nodes: usize) -> usize {
        match self {
            Self::SerialCopyChain => nodes - 1,
            Self::DisjointClears => 0,
        }
    }
}

#[derive(Debug)]
struct CaseEvidence {
    topology: Topology,
    node_count: usize,
    median_prepare_ns: u64,
    json: Value,
}

fn label(value: impl AsRef<str>) -> GpuResourceLabel {
    GpuResourceLabel::new(value.as_ref()).unwrap()
}

fn scale_buffer(
    resources: &mut GpuResourceScope,
    topology: Topology,
    index: usize,
) -> GpuBufferHandle {
    let usages = match topology {
        Topology::SerialCopyChain => {
            vec![GpuBufferUsage::CopySource, GpuBufferUsage::CopyDestination]
        }
        Topology::DisjointClears => vec![GpuBufferUsage::CopyDestination],
    };
    resources
        .buffer(
            GpuBufferDescriptor::ordinary_owned(
                format!("graph scale {} buffer {index:04}", topology.key()),
                GpuResourceLifetime::Transient,
                GpuReconstruction::SourceBacked,
                BUFFER_BYTES,
                usages,
                GpuBufferInitialization::Uninitialized,
            )
            .unwrap(),
        )
        .unwrap()
}

fn author_case(topology: Topology, node_count: usize) -> GpuWorkFragment {
    assert!(node_count > 0);
    let mut resources = GpuResourceScope::new();
    let buffers = (0..node_count)
        .map(|index| scale_buffer(&mut resources, topology, index))
        .collect::<Vec<_>>();

    GpuWorkFragment::build(
        format!("graph scale {} {node_count}", topology.key()),
        |work| {
            match topology {
                Topology::SerialCopyChain => {
                    work.operation(
                        "clear chain source",
                        GpuClearOperation::buffer_zero(
                            GpuBufferRegion::whole(&buffers[0]).unwrap(),
                        )
                        .unwrap(),
                    )?;
                    for index in 0..(node_count - 1) {
                        work.operation(
                            format!("copy chain {index:04} to {:04}", index + 1),
                            GpuCopyOperation::buffer_to_buffer(
                                GpuBufferRegion::whole(&buffers[index]).unwrap(),
                                GpuBufferRegion::whole(&buffers[index + 1]).unwrap(),
                            )
                            .unwrap(),
                        )?;
                    }
                }
                Topology::DisjointClears => {
                    for (index, buffer) in buffers.iter().enumerate() {
                        work.operation(
                            format!("clear independent {index:04}"),
                            GpuClearOperation::buffer_zero(GpuBufferRegion::whole(buffer).unwrap())
                                .unwrap(),
                        )?;
                    }
                }
            }
            Ok(())
        },
    )
    .unwrap()
}

fn prepare_case(label: GpuResourceLabel, fragment: GpuWorkFragment) -> GpuPreparedWorkGraph {
    GpuPreparedWorkGraph::prepare(label, [fragment]).unwrap()
}

fn assert_structure(
    topology: Topology,
    node_count: usize,
    authored_resource_count: usize,
    authored_access_count: usize,
    prepared: &GpuPreparedWorkGraph,
) {
    assert_eq!(authored_resource_count, node_count);
    assert_eq!(
        authored_access_count,
        topology.expected_access_count(node_count)
    );
    assert_eq!(prepared.nodes().len(), node_count);
    assert_eq!(prepared.topological_order().len(), node_count);
    assert_eq!(
        prepared.dependencies().len(),
        topology.expected_dependency_count(node_count)
    );
}

fn nanos(started: Instant) -> u64 {
    u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX)
}

fn summarize(samples: &[u64]) -> (u64, u64, u64) {
    assert!(!samples.is_empty());
    let mut ordered = samples.to_vec();
    ordered.sort_unstable();
    (
        ordered[0],
        ordered[ordered.len() / 2],
        *ordered.last().unwrap(),
    )
}

fn measure_case(topology: Topology, node_count: usize) -> CaseEvidence {
    let author_started = Instant::now();
    let fragment = author_case(topology, node_count);
    let author_ns = nanos(author_started);
    let prepared_label = label(format!(
        "graph scale prepared {} {node_count}",
        topology.key()
    ));
    let resource_count = fragment.resources().len();
    let access_count = fragment
        .nodes()
        .iter()
        .map(|node| node.accesses().len())
        .sum::<usize>();

    for _ in 0..WARMUP_SAMPLES {
        let candidate = fragment.clone();
        let graph_label = prepared_label.clone();
        let prepared = prepare_case(graph_label, candidate);
        assert_structure(
            topology,
            node_count,
            resource_count,
            access_count,
            &prepared,
        );
    }

    let mut samples = Vec::with_capacity(MEASURED_SAMPLES);
    let mut dependency_count = None;
    let mut topological_order_count = None;
    for _ in 0..MEASURED_SAMPLES {
        // Immutable input duplication and graph-label duplication deliberately occur outside the
        // measured interval. The retained metric is only the canonical graph
        // preparation/validation authority.
        let candidate = fragment.clone();
        let graph_label = prepared_label.clone();
        let started = Instant::now();
        let prepared = prepare_case(graph_label, candidate);
        let elapsed_ns = nanos(started);
        assert_structure(
            topology,
            node_count,
            resource_count,
            access_count,
            &prepared,
        );
        dependency_count = Some(prepared.dependencies().len());
        topological_order_count = Some(prepared.topological_order().len());
        samples.push(elapsed_ns);
    }

    let (min_ns, median_ns, max_ns) = summarize(&samples);
    let per_node_ns = median_ns as f64 / node_count as f64;
    CaseEvidence {
        topology,
        node_count,
        median_prepare_ns: median_ns,
        json: json!({
            "topology": topology.key(),
            "node_count": node_count,
            "resource_count": resource_count,
            "access_count": access_count,
            "dependency_count": dependency_count.unwrap(),
            "topological_order_count": topological_order_count.unwrap(),
            "authoring_ns": author_ns,
            "warmup_samples": WARMUP_SAMPLES,
            "measured_samples": MEASURED_SAMPLES,
            "prepare_including_validation_ns": {
                "samples": samples,
                "min": min_ns,
                "median": median_ns,
                "max": max_ns,
                "median_per_node": per_node_ns,
            },
        }),
    }
}

const FRAGMENTATION_CASES: [(usize, usize); 6] =
    [(80, 1), (80, 8), (80, 80), (800, 1), (800, 8), (800, 80)];

fn author_disjoint_fragments(node_count: usize, fragment_count: usize) -> Vec<GpuWorkFragment> {
    assert!(fragment_count > 0);
    assert_eq!(node_count % fragment_count, 0);
    let nodes_per_fragment = node_count / fragment_count;
    (0..fragment_count)
        .map(|fragment_index| {
            let mut resources = GpuResourceScope::new();
            let first_node = fragment_index * nodes_per_fragment;
            let buffers = (first_node..(first_node + nodes_per_fragment))
                .map(|index| scale_buffer(&mut resources, Topology::DisjointClears, index))
                .collect::<Vec<_>>();
            GpuWorkFragment::build(
                format!("graph scale disjoint fragment {fragment_index:04}"),
                |work| {
                    for (index, buffer) in buffers.iter().enumerate() {
                        work.operation(
                            format!("clear independent {:04}", first_node + index),
                            GpuClearOperation::buffer_zero(GpuBufferRegion::whole(buffer).unwrap())
                                .unwrap(),
                        )?;
                    }
                    Ok(())
                },
            )
            .unwrap()
        })
        .collect()
}

fn measure_disjoint_fragmentation(node_count: usize, fragment_count: usize) -> Value {
    let author_started = Instant::now();
    let fragments = author_disjoint_fragments(node_count, fragment_count);
    let author_ns = nanos(author_started);
    let resource_count = fragments
        .iter()
        .map(|fragment| fragment.resources().len())
        .sum::<usize>();
    let access_count = fragments
        .iter()
        .flat_map(|fragment| fragment.nodes())
        .map(|node| node.accesses().len())
        .sum::<usize>();
    assert_eq!(resource_count, node_count);
    assert_eq!(access_count, node_count);

    let graph_label = label(format!(
        "graph scale disjoint {node_count} nodes in {fragment_count} fragments"
    ));
    let mut observed_order: Option<Vec<_>> = None;
    let mut samples = Vec::with_capacity(MEASURED_SAMPLES);

    for index in 0..(WARMUP_SAMPLES + MEASURED_SAMPLES) {
        // Clone and label preparation are excluded just as in the retained
        // single-fragment comparison. The complete graph preparation is timed.
        let candidate = fragments.clone();
        let current_label = graph_label.clone();
        let started = Instant::now();
        let prepared = GpuPreparedWorkGraph::prepare(current_label, candidate).unwrap();
        let elapsed = nanos(started);

        assert_eq!(prepared.nodes().len(), node_count);
        assert_eq!(prepared.topological_order().len(), node_count);
        assert_eq!(prepared.dependencies().len(), 0);
        assert_eq!(
            prepared
                .topological_order()
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            node_count
        );
        match &observed_order {
            Some(previous) => assert_eq!(prepared.topological_order(), previous.as_slice()),
            None => observed_order = Some(prepared.topological_order().to_vec()),
        }
        if index >= WARMUP_SAMPLES {
            samples.push(elapsed);
        }
    }

    let (min_ns, median_ns, max_ns) = summarize(&samples);
    json!({
        "topology": "disjoint-clears",
        "node_count": node_count,
        "fragment_count": fragment_count,
        "nodes_per_fragment": node_count / fragment_count,
        "resource_count": resource_count,
        "access_count": access_count,
        "dependency_count": 0,
        "topological_order_count": node_count,
        "order_stable_across_samples": true,
        "authoring_ns": author_ns,
        "warmup_samples": WARMUP_SAMPLES,
        "measured_samples": MEASURED_SAMPLES,
        "prepare_including_validation_ns": {
            "samples": samples,
            "min": min_ns,
            "median": median_ns,
            "max": max_ns,
            "median_per_node": median_ns as f64 / node_count as f64,
        }
    })
}

const SHARED_RESOURCE_CASES: [(usize, usize); 6] =
    [(80, 1), (80, 8), (80, 80), (800, 1), (800, 8), (800, 80)];

fn author_shared_resource_fanout(
    node_count: usize,
    fragment_count: usize,
) -> (Vec<GpuWorkFragment>, GpuWorkResourceId) {
    assert!(node_count >= fragment_count);
    assert!(fragment_count > 0);
    let mut resources = GpuResourceScope::new();
    let shared = scale_buffer(&mut resources, Topology::SerialCopyChain, 0);
    let destinations = (1..node_count)
        .map(|index| scale_buffer(&mut resources, Topology::DisjointClears, index))
        .collect::<Vec<_>>();
    let export_key =
        GpuExportKey::new(format!("graph-scale.fanout.{node_count}.{fragment_count}")).unwrap();
    let shared_range = GpuBufferRange::whole(&shared).unwrap();

    let producer = GpuWorkFragment::build(
        format!("graph scale fanout producer {node_count}"),
        |work| {
            work.operation(
                "initialize fanout source",
                GpuClearOperation::buffer_zero(GpuBufferRegion::whole(&shared).unwrap()).unwrap(),
            )?;
            if fragment_count == 1 {
                for (index, destination) in destinations.iter().enumerate() {
                    work.operation(
                        format!("fanout read {index:04}"),
                        GpuCopyOperation::buffer_to_buffer(
                            GpuBufferRegion::whole(&shared).unwrap(),
                            GpuBufferRegion::whole(destination).unwrap(),
                        )
                        .unwrap(),
                    )?;
                }
            } else {
                let output_provenance =
                    GpuResourceProvenance::new(label("fanout producer output"), None, None);
                work.add_output(
                    GpuWorkOutput::new(
                        GpuExportRelationship::new(
                            shared.clone().into(),
                            export_key.clone(),
                            GpuResourceAccessIntent::Write,
                            output_provenance,
                        ),
                        GpuInitialCoverage::buffer(
                            &shared,
                            [GpuBufferCoverage::dense(shared_range)],
                        )
                        .unwrap(),
                    )
                    .unwrap(),
                )?;
            }
            Ok(())
        },
    )
    .unwrap();

    let mut fragments = vec![producer];
    if fragment_count > 1 {
        let consumer_count = fragment_count - 1;
        let reader_count = node_count - 1;
        for consumer in 0..consumer_count {
            let start = consumer * reader_count / consumer_count;
            let end = (consumer + 1) * reader_count / consumer_count;
            assert!(start < end);
            fragments.push(
                GpuWorkFragment::build(
                    format!("graph scale fanout consumer {consumer:04}"),
                    |work| {
                        work.declare_resource(shared.clone().into())?;
                        work.add_import(GpuWorkImport::new(
                            shared.clone().into(),
                            export_key.clone(),
                            GpuResourceAccessIntent::Read,
                            GpuResourceProvenance::new(
                                label(format!("fanout consumer {consumer:04} import")),
                                None,
                                None,
                            ),
                        ))?;
                        for (offset, destination) in destinations[start..end].iter().enumerate() {
                            work.operation(
                                format!("fanout read {:04}", start + offset),
                                GpuCopyOperation::buffer_to_buffer(
                                    GpuBufferRegion::whole(&shared).unwrap(),
                                    GpuBufferRegion::whole(destination).unwrap(),
                                )
                                .unwrap(),
                            )?;
                        }
                        Ok(())
                    },
                )
                .unwrap(),
            );
        }
    }
    (fragments, shared.diagnostic_identity())
}

fn measure_shared_resource_fanout(node_count: usize, fragment_count: usize) -> Value {
    let author_started = Instant::now();
    let (fragments, shared_resource) = author_shared_resource_fanout(node_count, fragment_count);
    let author_ns = nanos(author_started);
    assert_eq!(fragments.len(), fragment_count);
    let declared_resource_count = fragments
        .iter()
        .flat_map(GpuWorkFragment::resources)
        .map(GpuResourceRef::diagnostic_identity)
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    let access_count = fragments
        .iter()
        .flat_map(GpuWorkFragment::nodes)
        .map(|node| node.accesses().len())
        .sum::<usize>();
    let import_count = fragments
        .iter()
        .map(|fragment| fragment.imports().len())
        .sum::<usize>();
    let export_count = fragments
        .iter()
        .map(|fragment| fragment.outputs().len())
        .sum::<usize>();
    assert_eq!(declared_resource_count, node_count);
    assert_eq!(access_count, 2 * node_count - 1);
    assert_eq!(import_count, fragment_count - 1);
    assert_eq!(export_count, usize::from(fragment_count > 1));

    let graph_label = label(format!(
        "graph scale typed fanout {node_count} nodes in {fragment_count} fragments"
    ));
    let mut observed_order: Option<Vec<_>> = None;
    let mut samples = Vec::with_capacity(MEASURED_SAMPLES);
    for sample in 0..(WARMUP_SAMPLES + MEASURED_SAMPLES) {
        let candidate = fragments.clone();
        let current_label = graph_label.clone();
        let started = Instant::now();
        let prepared = GpuPreparedWorkGraph::prepare(current_label, candidate).unwrap();
        let elapsed = nanos(started);

        assert_eq!(prepared.nodes().len(), node_count);
        assert_eq!(prepared.topological_order().len(), node_count);
        assert_eq!(prepared.dependencies().len(), node_count - 1);
        let writer = prepared.topological_order().first().unwrap();
        for dependency in prepared.dependencies() {
            assert_eq!(dependency.before(), *writer);
            assert_eq!(dependency.reasons().len(), 1);
            assert!(matches!(
                dependency.reasons().first(),
                Some(GpuDependencyReason::ReadAfterWrite { resource, .. }) if *resource == shared_resource
            ));
        }
        assert_eq!(
            prepared
                .topological_order()
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            node_count
        );
        match &observed_order {
            Some(previous) => assert_eq!(prepared.topological_order(), previous.as_slice()),
            None => observed_order = Some(prepared.topological_order().to_vec()),
        }
        if sample >= WARMUP_SAMPLES {
            samples.push(elapsed);
        }
    }

    let (min_ns, median_ns, max_ns) = summarize(&samples);
    json!({
        "topology": "typed-shared-fanout",
        "node_count": node_count,
        "fragment_count": fragment_count,
        "resource_count": declared_resource_count,
        "access_count": access_count,
        "dependency_count": node_count - 1,
        "import_count": import_count,
        "export_count": export_count,
        "topological_order_count": node_count,
        "order_stable_across_samples": true,
        "all_edges_are_shared_source_read_after_write": true,
        "authoring_ns": author_ns,
        "warmup_samples": WARMUP_SAMPLES,
        "measured_samples": MEASURED_SAMPLES,
        "prepare_including_validation_ns": {
            "samples": samples,
            "min": min_ns,
            "median": median_ns,
            "max": max_ns,
            "median_per_node": median_ns as f64 / node_count as f64,
        }
    })
}

const NODE_LOOKUP_CASES: [(usize, usize); 6] =
    [(80, 1), (80, 8), (80, 80), (800, 1), (800, 8), (800, 80)];
const NODE_LOOKUP_REPEATS: usize = 32;

#[derive(Clone, Copy)]
enum NodeLookupStrategy {
    Linear,
    BTreeIndex,
    HashIndex,
}

impl NodeLookupStrategy {
    const fn key(self) -> &'static str {
        match self {
            Self::Linear => "linear-find",
            Self::BTreeIndex => "btree-index-build-and-find",
            Self::HashIndex => "hash-index-build-and-find",
        }
    }
}

fn run_node_lookup_pass(graph: &GpuPreparedWorkGraph, strategy: NodeLookupStrategy) -> usize {
    let nodes = graph.nodes();
    let order = graph.topological_order();
    let mut found_count = 0;
    match strategy {
        NodeLookupStrategy::Linear => {
            for id in order {
                let found = nodes
                    .iter()
                    .find(|node| node.id() == *id)
                    .expect("prepared topological identity must exist");
                assert_eq!(found.id(), *id);
                std::hint::black_box(found.node().accesses().len());
                found_count += 1;
            }
        }
        NodeLookupStrategy::BTreeIndex => {
            let index = nodes
                .iter()
                .map(|node| (node.id(), node))
                .collect::<std::collections::BTreeMap<_, _>>();
            assert_eq!(index.len(), nodes.len());
            for id in order {
                let found = index
                    .get(id)
                    .expect("indexed prepared topological identity must exist");
                assert_eq!(found.id(), *id);
                std::hint::black_box(found.node().accesses().len());
                found_count += 1;
            }
        }
        NodeLookupStrategy::HashIndex => {
            let index = nodes
                .iter()
                .map(|node| (node.id(), node))
                .collect::<std::collections::HashMap<_, _>>();
            assert_eq!(index.len(), nodes.len());
            for id in order {
                let found = index
                    .get(id)
                    .expect("indexed prepared topological identity must exist");
                assert_eq!(found.id(), *id);
                std::hint::black_box(found.node().accesses().len());
                found_count += 1;
            }
        }
    }
    found_count
}

fn measure_node_lookup(node_count: usize, fragment_count: usize) -> Value {
    // Graph authoring, hazard inference, and topological sorting are intentionally
    // outside this lookup-only timing window.
    let fragments = author_disjoint_fragments(node_count, fragment_count);
    let prepared = GpuPreparedWorkGraph::prepare(
        label(format!(
            "lookup {node_count} nodes, {fragment_count} fragments"
        )),
        fragments,
    )
    .unwrap();
    assert_eq!(prepared.nodes().len(), node_count);
    assert_eq!(prepared.topological_order().len(), node_count);
    assert_eq!(prepared.dependencies().len(), 0);
    assert_eq!(
        prepared
            .nodes()
            .iter()
            .map(GpuPreparedWorkNode::id)
            .collect::<std::collections::BTreeSet<_>>()
            .len(),
        node_count
    );

    let mut measurements = Vec::new();
    for strategy in [
        NodeLookupStrategy::Linear,
        NodeLookupStrategy::BTreeIndex,
        NodeLookupStrategy::HashIndex,
    ] {
        let mut samples = Vec::with_capacity(MEASURED_SAMPLES);
        for sample in 0..(WARMUP_SAMPLES + MEASURED_SAMPLES) {
            let started = Instant::now();
            for _ in 0..NODE_LOOKUP_REPEATS {
                assert_eq!(run_node_lookup_pass(&prepared, strategy), node_count);
            }
            let elapsed = nanos(started);
            if sample >= WARMUP_SAMPLES {
                samples.push(elapsed);
            }
        }
        let (min_ns, median_ns, max_ns) = summarize(&samples);
        measurements.push(json!({
            "strategy": strategy.key(),
            "samples_ns": samples,
            "min_ns": min_ns,
            "median_ns": median_ns,
            "max_ns": max_ns,
            "median_per_graph_pass_ns": median_ns as f64 / NODE_LOOKUP_REPEATS as f64,
            "successful_node_lookups_per_sample": node_count * NODE_LOOKUP_REPEATS,
        }));
    }
    json!({
        "topology": "disjoint-clears",
        "node_count": node_count,
        "fragment_count": fragment_count,
        "dependency_count": 0,
        "topological_order_count": node_count,
        "inner_repetitions": NODE_LOOKUP_REPEATS,
        "warmup_samples": WARMUP_SAMPLES,
        "measured_samples": MEASURED_SAMPLES,
        "measurements": measurements,
    })
}

fn scaling_ratio(cases: &[CaseEvidence], topology: Topology) -> f64 {
    let smallest = cases
        .iter()
        .find(|case| case.topology == topology && case.node_count == TIERS[0])
        .unwrap();
    let largest = cases
        .iter()
        .find(|case| case.topology == topology && case.node_count == TIERS[2])
        .unwrap();
    if smallest.median_prepare_ns == 0 {
        0.0
    } else {
        largest.median_prepare_ns as f64 / smallest.median_prepare_ns as f64
    }
}

fn artifact_path() -> PathBuf {
    std::env::var_os("RUNEN_GPU_PROOF_ARTIFACT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("target/runengpu-proof-artifacts"))
        .join("graph-scale")
        .join("report.json")
}

#[test]
#[ignore = "retained CPU timing characterization; executed by RunenGPU Conformance CI"]
fn graph_preparation_scale_characterization_records_structural_and_timing_evidence() {
    assert_eq!(TIERS[2] / TIERS[0], 100);

    let mut cases = Vec::new();
    for topology in [Topology::SerialCopyChain, Topology::DisjointClears] {
        for node_count in TIERS {
            cases.push(measure_case(topology, node_count));
        }
    }

    let serial_ratio = scaling_ratio(&cases, Topology::SerialCopyChain);
    let independent_ratio = scaling_ratio(&cases, Topology::DisjointClears);
    assert!(serial_ratio.is_finite());
    assert!(independent_ratio.is_finite());

    let fragmentation_cases = FRAGMENTATION_CASES
        .into_iter()
        .map(|(nodes, fragments)| measure_disjoint_fragmentation(nodes, fragments))
        .collect::<Vec<_>>();

    let shared_resource_cases = SHARED_RESOURCE_CASES
        .into_iter()
        .map(|(nodes, fragments)| measure_shared_resource_fanout(nodes, fragments))
        .collect::<Vec<_>>();

    let node_lookup_cases = NODE_LOOKUP_CASES
        .into_iter()
        .map(|(nodes, fragments)| measure_node_lookup(nodes, fragments))
        .collect::<Vec<_>>();

    let report = json!({
        "schema_version": REPORT_SCHEMA_VERSION,
        "requirement": "G6-S01",
        "subject": "canonical RunenGPU work-graph preparation scaling",
        "git_revision": std::env::var("RUNEN_GPU_PROOF_REVISION").ok(),
        "runner": {
            "os": std::env::var("RUNNER_OS").ok(),
            "arch": std::env::var("RUNNER_ARCH").ok(),
        },
        "tiers": TIERS,
        "largest_to_smallest_node_ratio": TIERS[2] / TIERS[0],
        "topologies": [
            {
                "key": Topology::SerialCopyChain.key(),
                "description": "one zero-clear followed by buffer-to-buffer copies; typed RAW hazards form a deep serial chain"
            },
            {
                "key": Topology::DisjointClears.key(),
                "description": "one zero-clear per disjoint buffer; no shared-resource dependency"
            }
        ],
        "measurement_boundary": {
            "timed": "GpuPreparedWorkGraph::prepare including its canonical validation work",
            "fragment_clone": "excluded from timed interval",
            "graph_label_clone": "excluded from timed interval",
            "separate_validation_timing_available": false,
            "separate_validation_timing_reason": "the accepted public path performs graph validation inside GpuPreparedWorkGraph::prepare and exposes no standalone validation phase",
            "allocation_or_memory_high_water_available": false,
            "allocation_or_memory_high_water_reason": "the accepted public graph-preparation path exposes no portable allocation/high-water metric; this proof does not add production or benchmark-only allocator authority"
        },
        "timing_policy": "characterization only; hosted-CI elapsed times and scaling ratios are not pass/fail budgets",
        "scaling": {
            Topology::SerialCopyChain.key(): {
                "largest_to_smallest_median_prepare_ratio": serial_ratio,
            },
            Topology::DisjointClears.key(): {
                "largest_to_smallest_median_prepare_ratio": independent_ratio,
            }
        },
        "cases": cases.into_iter().map(|case| case.json).collect::<Vec<_>>(),
        "fragmentation_cases": fragmentation_cases,
        "shared_resource_cases": shared_resource_cases,
        "node_lookup_cases": node_lookup_cases,
        "node_lookup_policy": "test-profile CPU-only lookup comparison; linear search versus transient BTreeMap/HashMap including index construction on each pass, no physical WGPU plan, shader or resource realization; non-gating",
        "shared_resource_policy": "fixed total nodes: one writer and N-1 read/copy consumers; typed export/import when fragmented, one RAW dependency per consumer; GPU physical preparation not measured; non-gating",
        "fragmentation_policy": "fixed total nodes and unique resources; fragment count alone varies; no typed shared resources, no physical execution preparation; measured time is non-gating",
    });

    let path = artifact_path();
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    assert!(path.metadata().unwrap().len() > 0);
}
