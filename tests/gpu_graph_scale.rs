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
    });

    let path = artifact_path();
    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    std::fs::write(&path, serde_json::to_vec_pretty(&report).unwrap()).unwrap();
    assert!(path.metadata().unwrap().len() > 0);
}
