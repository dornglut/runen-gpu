//! Opt-in G3 phase observer qualification. A thread-local collector proves
//! the existing tracing subscriber can observe exclusive stage scopes without
//! adding a new production profiler API or persistent per-node telemetry.
use super::support::*;
use std::collections::BTreeMap;
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicU64, Ordering},
};
use tracing::{
    Event, Metadata, Subscriber,
    span::{Attributes, Id, Record},
};

#[derive(Default)]
struct RecordedStages {
    names: BTreeMap<u64, &'static str>,
    entered: Vec<&'static str>,
    stack: Vec<&'static str>,
}

struct StageCollector {
    state: Arc<Mutex<RecordedStages>>,
    next: AtomicU64,
}

impl Subscriber for StageCollector {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        metadata.target() == "runen_gpu::graph" && *metadata.level() == tracing::Level::TRACE
    }

    fn new_span(&self, attributes: &Attributes<'_>) -> Id {
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        assert_ne!(id, 0, "span identities must be nonzero");
        let mut state = self.state.lock().unwrap();
        state.names.insert(id, attributes.metadata().name());
        Id::from_u64(id)
    }

    fn record(&self, _span: &Id, _values: &Record<'_>) {}

    fn record_follows_from(&self, _span: &Id, _follows: &Id) {}

    fn event(&self, _event: &Event<'_>) {}

    fn enter(&self, span: &Id) {
        let mut state = self.state.lock().unwrap();
        let name = *state
            .names
            .get(&span.into_u64())
            .expect("registered G3 span");
        if name != "g3.prepare_core" {
            assert_eq!(
                state.stack.as_slice(),
                ["g3.prepare_core"],
                "each G3 stage must be exclusively nested under its parent"
            );
        }
        state.entered.push(name);
        state.stack.push(name);
    }

    fn exit(&self, span: &Id) {
        let mut state = self.state.lock().unwrap();
        let name = *state
            .names
            .get(&span.into_u64())
            .expect("registered G3 span");
        assert_eq!(
            state.stack.pop(),
            Some(name),
            "G3 stages must exit in LIFO order"
        );
    }
}

#[test]
fn g3_stages_are_observable_nonoverlapping_and_drop_cleanly_on_error() {
    // Author before the collector is installed so the captured scope is G3
    // preparation alone, not unrelated checked shader source authoring.
    let mut ordinary = builder("traced successful source");
    add_compute(&mut ordinary, "first independent step", []);
    add_compute(&mut ordinary, "second independent step", []);
    let complete = ordinary.finish().unwrap();

    let mut present = builder("traced required-order source");
    let existing = add_compute(&mut present, "existing first step", []);
    let complete_with_missing = present.finish().unwrap();
    let mut missing = builder("excluded source");
    let excluded = add_compute(&mut missing, "missing endpoint", []);
    let requirement = GpuGraphRequiredOrder::new(&existing, &excluded, "excluded end").unwrap();

    let captured = Arc::new(Mutex::new(RecordedStages::default()));
    let collector = StageCollector {
        state: Arc::clone(&captured),
        next: AtomicU64::new(1),
    };
    tracing::subscriber::with_default(collector, || {
        let prepared = GpuPreparedWorkGraph::prepare(label("traced success"), [complete]).unwrap();
        assert_eq!(prepared.nodes().len(), 2);
        assert_eq!(prepared.topological_order().len(), 2);
        assert_eq!(
            GpuPreparedWorkGraph::prepare_with_required_orders(
                label("traced early error"),
                [complete_with_missing],
                [requirement],
            )
            .unwrap_err()
            .cause(),
            GpuWorkGraphCause::ForeignIdentity
        );
    });

    let state = captured.lock().unwrap();
    assert!(
        state.stack.is_empty(),
        "errors must exit all active G3 spans"
    );
    let stages = [
        "g3.prepare_core",
        "g3.collect_arguments",
        "g3.resource_and_node_validation",
        "g3.retained_and_reconstruction",
        "g3.boundary_binding",
        "g3.typed_hazard_inference",
        "g3.strict_and_required_order_admission",
        "g3.topological_order",
        "g3.initial_content_and_boundary_validation",
        "g3.capability_requirements",
        "g3.dynamic_initialization_simulation",
        "g3.diagnostics_and_result",
    ];
    assert_eq!(
        &state.entered[..stages.len()],
        stages,
        "the successful graph must expose all exclusive phases"
    );
    assert_eq!(
        state
            .entered
            .iter()
            .filter(|&&name| name == "g3.prepare_core")
            .count(),
        2,
        "successful and rejected G3 calls must each have a parent span"
    );
    assert!(
        state.entered[stages.len()..].contains(&"g3.strict_and_required_order_admission"),
        "a rejected required order must still identify its failed phase"
    );
}
