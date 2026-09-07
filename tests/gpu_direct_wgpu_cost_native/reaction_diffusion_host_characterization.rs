use super::{envelope_json, label, retained, runengpu_context, runengpu_envelope_sample};
use crate::common::{MEASURED_SAMPLES, Measurements, WARMUP_SAMPLES};
use runen_gpu::{GpuContext, GpuPreparedWorkGraph};
use serde_json::{Value, json};

const PHASES: [&str; 6] = [
    "authoring",
    "graph_prepare",
    "backend_prepare",
    "submit_encode_and_queue",
    "completion_readback",
    "total",
];
const STRUCTURE_COUNTS: [&str; 8] = [
    "resource_count",
    "node_count",
    "access_count",
    "dependency_count",
    "topological_order_count",
    "initialization_summary_count",
    "diagnostic_count",
    "readback_count",
];

fn graph_structure(sources: &retained::ProgramSources, envelope: retained::Envelope) -> Value {
    let (graph_label, fragment, readbacks) = retained::offscreen_work(sources, envelope);
    let resource_count = fragment.resources().len();
    let node_count = fragment.nodes().len();
    let access_count = fragment
        .nodes()
        .iter()
        .map(|node| node.accesses().len())
        .sum::<usize>();
    let graph = GpuPreparedWorkGraph::prepare(label(&graph_label), [fragment]).unwrap();

    assert_eq!(graph.nodes().len(), node_count);
    assert_eq!(graph.topological_order().len(), node_count);

    json!({
        "fragment_count": 1,
        "resource_count": resource_count,
        "node_count": node_count,
        "access_count": access_count,
        "dependency_count": graph.dependencies().len(),
        "topological_order_count": graph.topological_order().len(),
        "initialization_summary_count": graph.initialization().len(),
        "diagnostic_count": graph.diagnostics().len(),
        "output_count": graph.outputs().len(),
        "readback_count": readbacks.len(),
    })
}

fn measure_envelope(
    context: &GpuContext,
    sources: &retained::ProgramSources,
    envelope: retained::Envelope,
) -> Value {
    for _ in 0..WARMUP_SAMPLES {
        let _ = runengpu_envelope_sample(context, sources, envelope);
    }

    let mut measurements = Measurements::default();
    for _ in 0..MEASURED_SAMPLES {
        measurements.push(runengpu_envelope_sample(context, sources, envelope));
    }

    json!({
        "envelope": envelope_json(envelope),
        "structure": graph_structure(sources, envelope),
        "runengpu": measurements.to_json(),
    })
}

fn median_phase(envelope: &Value, phase: &str) -> f64 {
    envelope["runengpu"]["summary_us"][phase]["median"]
        .as_f64()
        .unwrap()
}

fn structure_count(envelope: &Value, name: &str) -> f64 {
    envelope["structure"][name].as_u64().unwrap() as f64
}

fn growth_summary(smaller: &Value, larger: &Value) -> Value {
    let phase_growth = PHASES
        .into_iter()
        .map(|phase| {
            let smaller_median = median_phase(smaller, phase);
            let larger_median = median_phase(larger, phase);
            assert!(smaller_median > 0.0);
            (
                phase.to_owned(),
                json!({
                    "smaller_median_us": smaller_median,
                    "larger_median_us": larger_median,
                    "larger_over_smaller_ratio": larger_median / smaller_median,
                }),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    let structure_growth = STRUCTURE_COUNTS
        .into_iter()
        .map(|name| {
            let smaller_count = structure_count(smaller, name);
            let larger_count = structure_count(larger, name);
            assert!(smaller_count > 0.0);
            (
                name.to_owned(),
                json!({
                    "smaller": smaller_count,
                    "larger": larger_count,
                    "larger_over_smaller_ratio": larger_count / smaller_count,
                }),
            )
        })
        .collect::<serde_json::Map<_, _>>();

    json!({
        "structure_growth": structure_growth,
        "phase_median_growth": phase_growth,
    })
}

pub(crate) fn evidence() -> Value {
    let context = runengpu_context();
    let sources = retained::admitted_sources();
    let smaller = measure_envelope(&context, &sources, retained::ENVELOPES[0]);
    let larger = measure_envelope(&context, &sources, retained::ENVELOPES[1]);
    let growth = growth_summary(&smaller, &larger);

    json!({
        "purpose": "G6 retained host-cost characterization at the measured repository revision",
        "measurement_boundary": {
            "path": "existing canonical RunenGPU reaction-diffusion authoring -> GpuPreparedWorkGraph::prepare -> prepare_submission -> submit_prepared -> completion/readback",
            "production_instrumentation_added": false,
            "second_preparation_authority_added": false,
            "direct_wgpu_remeasured_here": false,
            "direct_wgpu_reference": "use the sibling retained G6-P01 aggregate comparison from the same portfolio report",
            "context_reused_across_envelopes": true,
            "program_sources_reused_across_envelopes": true,
            "warmup_samples_per_envelope": WARMUP_SAMPLES,
            "measured_samples_per_envelope": MEASURED_SAMPLES,
        },
        "source_audit": {
            "authoring_whole_builder_snapshot_per_lexical_operation": false,
            "authoring_transaction_mechanism": "bounded newly inserted resource journal with rollback on failed lexical insertion",
            "prepare_node_access_membership_scans_fragment_resources": false,
            "prepare_resource_membership_mechanism": "fragment-local diagnostic-identity index built during canonical fragment resource registration",
            "canonical_prepare_initialization_passes": [
                "validate_fragment_initialization",
                "simulate_prepared_initialization",
            ],
            "retained_workload_fragment_count": 1,
            "cross_fragment_hazard_path_exercised": false,
            "status": "observations at the measured repository revision; correction sequencing remains issue-owned",
        },
        "envelopes": [smaller, larger],
        "larger_over_smaller": growth,
    })
}
