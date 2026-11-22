use super::super::{
    GpuCapabilityFeature, GpuCapabilityRequirement, GpuCapabilityRequirements, GpuResourceLabel,
    GpuResourceRef, GpuRetainedInitializationSeed, GpuWorkGraphCause, GpuWorkGraphError,
    GpuWorkGraphErrorContext, GpuWorkGraphErrorSource, GpuWorkResourceId,
};
use super::{
    authoring::{
        GpuGraphExplicitOrder, GpuGraphRequiredOrder, GpuWorkFragment, GpuWorkNode, GpuWorkOutput,
    },
    composition::{
        bind_imports, collect_output_bindings, topological_fragment_order,
        validate_boundary_access_intents,
    },
    coverage::{GpuInitialCoverage, canonical_storage_resource, storage_identity},
    dependency::{GpuDependencyReason, GpuWorkDependency},
    diagnostics::{GpuPreparedWorkDiagnostic, GraphErrorOrigin, graph_error},
    hazards::{
        DataDependencyPaths, add_explicit_orders, add_graph_explicit_orders,
        add_graph_required_orders, infer_cross_fragment_hazards, infer_fragment_hazards,
        topological_node_order,
    },
    identity::GpuPreparedWorkNodeId,
    initial_content::{GpuPreparedInitialContent, derive_prepared_initial_content},
    initialization::{
        GpuPreparedResourceInitialization, simulate_prepared_initialization,
        validate_fragment_initialization,
    },
    same_resource_descriptor,
};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::OnceLock,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuPreparedWorkNode {
    id: GpuPreparedWorkNodeId,
    fragment_label: GpuResourceLabel,
    node: GpuWorkNode,
}

impl GpuPreparedWorkNode {
    pub const fn id(&self) -> GpuPreparedWorkNodeId {
        self.id
    }

    pub fn fragment_label(&self) -> &GpuResourceLabel {
        &self.fragment_label
    }

    pub fn node(&self) -> &GpuWorkNode {
        &self.node
    }
}

/// Immutable deterministic result of graph preparation.
///
/// ```compile_fail
/// use runen_gpu::GpuPreparedWorkGraph;
///
/// fn mutate(graph: &mut GpuPreparedWorkGraph) {
///     graph.topological_order.clear();
/// }
/// ```
#[derive(Clone)]
pub struct GpuPreparedWorkGraph {
    label: GpuResourceLabel,
    nodes: Vec<GpuPreparedWorkNode>,
    topological_order: Vec<GpuPreparedWorkNodeId>,
    dependencies: Vec<GpuWorkDependency>,
    initialization: Vec<GpuPreparedResourceInitialization>,
    requirements: GpuCapabilityRequirements,
    outputs: Vec<GpuWorkOutput>,
    diagnostics: OnceLock<Vec<GpuPreparedWorkDiagnostic>>,
    initialization_diagnostics: Vec<GpuPreparedWorkDiagnostic>,
    initial_content: Vec<GpuPreparedInitialContent>,
    retained_seed: Vec<GpuRetainedInitializationSeed>,
    reconstruction_targets: BTreeSet<GpuWorkResourceId>,
    failure_preserved_coverage: BTreeMap<GpuWorkResourceId, GpuInitialCoverage>,
}

// Preserve the observable Debug payload of prepared graphs. Formatting is
// an explicit diagnostic observation and may initialize the lazy view, but
// ordinary G3/G5 work does not pay for a duplicated dependency vector.
impl core::fmt::Debug for GpuPreparedWorkGraph {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("GpuPreparedWorkGraph")
            .field("label", &self.label)
            .field("nodes", &self.nodes)
            .field("topological_order", &self.topological_order)
            .field("dependencies", &self.dependencies)
            .field("initialization", &self.initialization)
            .field("requirements", &self.requirements)
            .field("outputs", &self.outputs)
            .field("diagnostics", &self.diagnostics())
            .field("initial_content", &self.initial_content)
            .field("retained_seed", &self.retained_seed)
            .field("reconstruction_targets", &self.reconstruction_targets)
            .field("failure_preserved_coverage", &self.failure_preserved_coverage)
            .finish()
    }
}

impl GpuPreparedWorkGraph {
    pub fn prepare(
        label: GpuResourceLabel,
        fragments: impl IntoIterator<Item = GpuWorkFragment>,
    ) -> Result<Self, GpuWorkGraphError> {
        Self::prepare_with_orders(
            label,
            fragments,
            std::iter::empty::<GpuGraphExplicitOrder>(),
        )
    }

    pub fn prepare_with_orders(
        label: GpuResourceLabel,
        fragments: impl IntoIterator<Item = GpuWorkFragment>,
        graph_orders: impl IntoIterator<Item = GpuGraphExplicitOrder>,
    ) -> Result<Self, GpuWorkGraphError> {
        Self::prepare_with_retained_coverage_and_orders(label, fragments, graph_orders, &[])
    }

    /// Prepares the full canonical graph once while admitting render/compute
    /// control requirements that are already guaranteed by typed dependencies.
    ///
    /// Unlike strict graph orders, redundant required orders are dropped
    /// without changing the canonical graph. Reverse typed causality and
    /// cycles still fail with RunenGPU's typed errors.
    pub fn prepare_with_required_orders(
        label: GpuResourceLabel,
        fragments: impl IntoIterator<Item = GpuWorkFragment>,
        required_orders: impl IntoIterator<Item = GpuGraphRequiredOrder>,
    ) -> Result<Self, GpuWorkGraphError> {
        Self::prepare_with_retained_coverage_and_required_orders(
            label,
            fragments,
            required_orders,
            &[],
        )
    }

    pub(crate) fn prepare_with_retained_coverage(
        label: GpuResourceLabel,
        fragments: impl IntoIterator<Item = GpuWorkFragment>,
        retained_coverage: &[GpuRetainedInitializationSeed],
    ) -> Result<Self, GpuWorkGraphError> {
        Self::prepare_with_retained_coverage_and_orders(
            label,
            fragments,
            std::iter::empty::<GpuGraphExplicitOrder>(),
            retained_coverage,
        )
    }

    pub(crate) fn prepare_with_retained_coverage_and_orders(
        label: GpuResourceLabel,
        fragments: impl IntoIterator<Item = GpuWorkFragment>,
        graph_orders: impl IntoIterator<Item = GpuGraphExplicitOrder>,
        retained_coverage: &[GpuRetainedInitializationSeed],
    ) -> Result<Self, GpuWorkGraphError> {
        Self::prepare_core(
            label,
            fragments,
            graph_orders,
            std::iter::empty::<GpuGraphRequiredOrder>(),
            retained_coverage,
            &[],
        )
    }

    pub(crate) fn prepare_with_retained_coverage_and_required_orders(
        label: GpuResourceLabel,
        fragments: impl IntoIterator<Item = GpuWorkFragment>,
        required_orders: impl IntoIterator<Item = GpuGraphRequiredOrder>,
        retained_coverage: &[GpuRetainedInitializationSeed],
    ) -> Result<Self, GpuWorkGraphError> {
        Self::prepare_core(
            label,
            fragments,
            std::iter::empty::<GpuGraphExplicitOrder>(),
            required_orders,
            retained_coverage,
            &[],
        )
    }

    pub(crate) fn prepare_with_retained_coverage_and_reconstruction(
        label: GpuResourceLabel,
        fragments: impl IntoIterator<Item = GpuWorkFragment>,
        retained_coverage: &[GpuRetainedInitializationSeed],
        reconstruction_targets: &[GpuResourceRef],
    ) -> Result<Self, GpuWorkGraphError> {
        Self::prepare_core(
            label,
            fragments,
            std::iter::empty::<GpuGraphExplicitOrder>(),
            std::iter::empty::<GpuGraphRequiredOrder>(),
            retained_coverage,
            reconstruction_targets,
        )
    }

    fn prepare_core(
        label: GpuResourceLabel,
        fragments: impl IntoIterator<Item = GpuWorkFragment>,
        graph_orders: impl IntoIterator<Item = GpuGraphExplicitOrder>,
        required_orders: impl IntoIterator<Item = GpuGraphRequiredOrder>,
        retained_coverage: &[GpuRetainedInitializationSeed],
        reconstruction_targets: &[GpuResourceRef],
    ) -> Result<Self, GpuWorkGraphError> {
        // Only the existing, explicitly opted-in tracing subscriber records
        // these spans. No graph telemetry store, allocation or per-node event.
        let _g3_profile = tracing::trace_span!(
            target: "runen_gpu::graph",
            "g3.prepare_core"
        )
        .entered();
        let stage = tracing::trace_span!(
            target: "runen_gpu::graph",
            "g3.collect_arguments"
        )
        .entered();
        let graph_orders = graph_orders.into_iter().collect::<Vec<_>>();
        let required_orders = required_orders.into_iter().collect::<Vec<_>>();
        let fragments = fragments.into_iter().collect::<Vec<_>>();
        let graph_label = label.as_str();
        let mut declared_resources = BTreeMap::<GpuWorkResourceId, GpuResourceRef>::new();
        let mut storage_resources = BTreeMap::<GpuWorkResourceId, GpuResourceRef>::new();
        let mut prepared_nodes = Vec::new();
        let mut node_locations = BTreeMap::<GpuPreparedWorkNodeId, (usize, usize)>::new();
        drop(stage);
        let stage = tracing::trace_span!(
            target: "runen_gpu::graph",
            "g3.resource_and_node_validation"
        )
        .entered();

        for (fragment_index, fragment) in fragments.iter().enumerate() {
            let fragment_ordinal = u32::try_from(fragment_index).map_err(|_| {
                graph_error(
                    "assign prepared fragment identity",
                    graph_label,
                    GraphErrorOrigin::new(Some(fragment), None),
                    None,
                    None,
                    GpuWorkGraphCause::UnknownIdentity,
                    "prepare fewer fragments in one bounded graph",
                )
            })?;
            let fragment_resource_identities = register_fragment_resources(
                graph_label,
                fragment,
                &mut declared_resources,
                &mut storage_resources,
            )?;
            for (node_index, node) in fragment.nodes().iter().enumerate() {
                let expected_local = u64::try_from(node_index)
                    .ok()
                    .and_then(|index| index.checked_add(1));
                if !node.id().belongs_to(&fragment.identity)
                    || expected_local != Some(node.id().diagnostic_local())
                {
                    return Err(graph_error(
                        "validate fragment-local GPU work-node identity",
                        graph_label,
                        GraphErrorOrigin::new(Some(fragment), Some(node)),
                        None,
                        None,
                        GpuWorkGraphCause::ForeignIdentity,
                        "retain nodes allocated monotonically by their originating fragment builder",
                    ));
                }
                validate_node_resources(
                    graph_label,
                    fragment,
                    node,
                    &fragment_resource_identities,
                )?;
                let id = GpuPreparedWorkNodeId::new(fragment_ordinal, node.id().local);
                if node_locations
                    .insert(id, (fragment_index, node_index))
                    .is_some()
                {
                    return Err(graph_error(
                        "validate prepared GPU work-node identity",
                        graph_label,
                        GraphErrorOrigin::new(Some(fragment), Some(node)),
                        Some(id),
                        None,
                        GpuWorkGraphCause::UnknownIdentity,
                        "use distinct monotonically allocated local node identities",
                    ));
                }
                prepared_nodes.push(GpuPreparedWorkNode {
                    id,
                    fragment_label: fragment.label().clone(),
                    node: node.clone(),
                });
            }
        }

        drop(stage);
        let stage = tracing::trace_span!(
            target: "runen_gpu::graph",
            "g3.retained_and_reconstruction"
        )
        .entered();
        let reconstruction_targets = validate_reconstruction_targets(
            graph_label,
            reconstruction_targets,
            &storage_resources,
        )?;
        let retained_seed = retained_coverage
            .iter()
            .filter(|seed| {
                let seed_storage = canonical_storage_resource(seed.resource());
                storage_resources
                    .get(&seed.resource_identity())
                    .is_some_and(|current_storage| {
                        current_storage.common().lifetime().is_retained()
                            && seed_storage.common().lifetime().is_retained()
                            && same_resource_descriptor(current_storage, &seed_storage)
                    })
            })
            .cloned()
            .collect::<Vec<_>>();

        drop(stage);
        let stage = tracing::trace_span!(
            target: "runen_gpu::graph",
            "g3.boundary_binding"
        )
        .entered();
        let output_bindings = collect_output_bindings(graph_label, &fragments)?;
        let (import_bindings, relations) = bind_imports(graph_label, &fragments, &output_bindings)?;
        validate_boundary_access_intents(graph_label, &fragments)?;
        drop(stage);
        let stage = tracing::trace_span!(
            target: "runen_gpu::graph",
            "g3.typed_hazard_inference"
        )
        .entered();
        let mut inferred_edges = BTreeMap::<
            (GpuPreparedWorkNodeId, GpuPreparedWorkNodeId),
            BTreeSet<GpuDependencyReason>,
        >::new();
        infer_fragment_hazards(graph_label, &fragments, &mut inferred_edges)?;
        infer_cross_fragment_hazards(graph_label, &fragments, &relations, &mut inferred_edges)?;
        drop(stage);
        let stage = tracing::trace_span!(
            target: "runen_gpu::graph",
            "g3.strict_and_required_order_admission"
        )
        .entered();
        // No index or additional traversal is needed when no explicit orders
        // exist. Most preparation paths must not pay to optimize absent work.
        if !graph_orders.is_empty()
            || !required_orders.is_empty()
            || fragments
                .iter()
                .any(|fragment| !fragment.explicit_orders().is_empty())
        {
            // Canonical typed edges are inferred once. Explicit-order checks
            // query only outgoing neighbors and see each admitted control edge.
            let mut data_paths = DataDependencyPaths::from_edges(&inferred_edges);
            add_explicit_orders(
                graph_label,
                &fragments,
                &mut inferred_edges,
                &mut data_paths,
            )?;
            add_graph_explicit_orders(
                graph_label,
                &fragments,
                &graph_orders,
                &mut inferred_edges,
                &mut data_paths,
            )?;
            add_graph_required_orders(
                graph_label,
                &fragments,
                &required_orders,
                &node_locations,
                &mut inferred_edges,
                &mut data_paths,
            )?;
        }
        drop(stage);
        let stage = tracing::trace_span!(
            target: "runen_gpu::graph",
            "g3.topological_order"
        )
        .entered();
        let topological_order =
            topological_node_order(graph_label, &fragments, &node_locations, &inferred_edges)?;
        let fragment_order = topological_fragment_order(graph_label, &fragments, &import_bindings)?;
        drop(stage);
        let stage = tracing::trace_span!(
            target: "runen_gpu::graph",
            "g3.initial_content_and_boundary_validation"
        )
        .entered();
        let initial_content = derive_prepared_initial_content(graph_label, &fragments)?
            .into_iter()
            .filter(|candidate| {
                !retained_seed
                    .iter()
                    .any(|seed| seed.resource_identity() == candidate.resource_identity())
            })
            .collect::<Vec<_>>();
        validate_reconstruction_participation(
            graph_label,
            &reconstruction_targets,
            &prepared_nodes,
            &initial_content,
        )?;
        validate_fragment_initialization(
            graph_label,
            &fragments,
            &fragment_order,
            &import_bindings,
            &initial_content,
            &retained_seed,
        )?;

        drop(stage);
        let stage = tracing::trace_span!(
            target: "runen_gpu::graph",
            "g3.capability_requirements"
        )
        .entered();
        let mut requirements = GpuCapabilityRequirements::new();
        for prepared in &prepared_nodes {
            requirements = requirements
                .merge(prepared.node().requirements())
                .map_err(|source| {
                    GpuWorkGraphError::with_source(
                        "merge GPU work-graph capability requirements",
                        GpuWorkGraphErrorContext::new(
                            graph_label,
                            Some(prepared.fragment_label().as_str().to_string()),
                            Some(prepared.node().label().as_str().to_string()),
                            Some(prepared.id()),
                            None,
                            None,
                            Some(prepared.node().provenance().clone()),
                        ),
                        GpuWorkGraphCause::MechanicalCapabilityContradiction,
                        "remove graph-wide caller constraints that disable mechanically required capabilities",
                        GpuWorkGraphErrorSource::Capability(source),
                    )
                })?;
        }
        if !initial_content.is_empty() {
            requirements
                .insert(GpuCapabilityRequirement::Required(GpuCapabilityFeature::Copy))
                .map_err(|source| {
                    GpuWorkGraphError::with_source(
                        "merge prepared initial-content capability requirement",
                        GpuWorkGraphErrorContext::new(
                            graph_label,
                            None,
                            None,
                            None,
                            initial_content
                                .first()
                                .map(GpuPreparedInitialContent::resource_identity),
                            None,
                            None,
                        ),
                        GpuWorkGraphCause::MechanicalCapabilityContradiction,
                        "permit the Copy capability required by canonical prepared initial-content transfer",
                        GpuWorkGraphErrorSource::Capability(source),
                    )
                })?;
        }

        drop(stage);
        let stage = tracing::trace_span!(
            target: "runen_gpu::graph",
            "g3.dynamic_initialization_simulation"
        )
        .entered();
        let (initialization, initialization_diagnostics, failure_preserved_coverage) =
            simulate_prepared_initialization(
                graph_label,
                &fragments,
                &storage_resources,
                &node_locations,
                &topological_order,
                &initial_content,
                &retained_seed,
            )?;
        drop(stage);
        let _materialization = tracing::trace_span!(
            target: "runen_gpu::graph",
            "g3.diagnostics_and_result"
        )
        .entered();
        let dependencies = inferred_edges
            .into_iter()
            .map(|((before, after), reasons)| GpuWorkDependency {
                before,
                after,
                reasons: reasons.into_iter().collect(),
            })
            .collect::<Vec<_>>();
        let outputs = fragments
            .iter()
            .flat_map(|fragment| fragment.outputs().iter().cloned())
            .collect::<Vec<_>>();
        // Many real-frame consumers only use the validated execution order
        // and operation/retained-state facts. Do not clone thousands of
        // dependency reasons into a second diagnostic vector until a caller
        // explicitly requests the public diagnostic view.
        let diagnostics = OnceLock::new();

        Ok(Self {
            label,
            nodes: prepared_nodes,
            topological_order,
            dependencies,
            initialization,
            requirements,
            outputs,
            diagnostics,
            initialization_diagnostics,
            initial_content,
            retained_seed,
            reconstruction_targets,
            failure_preserved_coverage,
        })
    }

    pub fn label(&self) -> &GpuResourceLabel {
        &self.label
    }

    pub fn nodes(&self) -> &[GpuPreparedWorkNode] {
        &self.nodes
    }

    pub fn topological_order(&self) -> &[GpuPreparedWorkNodeId] {
        &self.topological_order
    }

    pub fn dependencies(&self) -> &[GpuWorkDependency] {
        &self.dependencies
    }

    pub fn initialization(&self) -> &[GpuPreparedResourceInitialization] {
        &self.initialization
    }

    pub fn requirements(&self) -> &GpuCapabilityRequirements {
        &self.requirements
    }

    pub fn outputs(&self) -> &[GpuWorkOutput] {
        &self.outputs
    }

    pub fn diagnostics(&self) -> &[GpuPreparedWorkDiagnostic] {
        self.diagnostics.get_or_init(|| {
            let mut diagnostics = self
                .dependencies
                .iter()
                .cloned()
                .map(GpuPreparedWorkDiagnostic::Dependency)
                .collect::<Vec<_>>();
            diagnostics.extend(self.initialization_diagnostics.iter().cloned());
            diagnostics.extend(
                self.outputs
                    .iter()
                    .map(|output| GpuPreparedWorkDiagnostic::Output {
                        export_key: output.relationship().export_key().clone(),
                        resource: storage_identity(output.relationship().resource()),
                    }),
            );
            diagnostics
        })
    }

    pub(crate) fn initial_content(&self) -> &[GpuPreparedInitialContent] {
        &self.initial_content
    }

    pub(crate) fn retained_seed(&self) -> &[GpuRetainedInitializationSeed] {
        &self.retained_seed
    }

    pub(crate) fn reconstruction_targets(&self) -> &BTreeSet<GpuWorkResourceId> {
        &self.reconstruction_targets
    }

    pub(crate) fn failure_preserved_coverage(
        &self,
        resource: GpuWorkResourceId,
    ) -> Option<&GpuInitialCoverage> {
        self.failure_preserved_coverage.get(&resource)
    }
}

fn validate_reconstruction_targets(
    graph_label: &str,
    targets: &[GpuResourceRef],
    storage: &BTreeMap<GpuWorkResourceId, GpuResourceRef>,
) -> Result<BTreeSet<GpuWorkResourceId>, GpuWorkGraphError> {
    let mut validated = BTreeSet::new();
    for target in targets {
        let target = canonical_storage_resource(target);
        let identity = storage_identity(&target);
        if !target.common().lifetime().is_retained()
            || !matches!(
                &target,
                GpuResourceRef::Buffer(_)
                    | GpuResourceRef::Texture(_)
                    | GpuResourceRef::QuerySet(_)
            )
        {
            return Err(graph_error(
                "validate retained reconstruction target",
                graph_label,
                GraphErrorOrigin::new(None, None),
                None,
                Some(identity),
                GpuWorkGraphCause::UnknownIdentity,
                "target only retained storage resources for current-state reconstruction",
            ));
        }
        let Some(declared) = storage.get(&identity) else {
            return Err(graph_error(
                "validate retained reconstruction target",
                graph_label,
                GraphErrorOrigin::new(None, None),
                None,
                Some(identity),
                GpuWorkGraphCause::UnknownIdentity,
                "declare the exact reconstruction target in the prepared work graph",
            ));
        };
        if !same_resource_descriptor(declared, &target) {
            return Err(graph_error(
                "validate retained reconstruction target",
                graph_label,
                GraphErrorOrigin::new(None, None),
                None,
                Some(identity),
                GpuWorkGraphCause::UnknownIdentity,
                "use the exact declared descriptor for the reconstruction target identity",
            ));
        }
        validated.insert(identity);
    }
    Ok(validated)
}

fn validate_reconstruction_participation(
    graph_label: &str,
    targets: &BTreeSet<GpuWorkResourceId>,
    nodes: &[GpuPreparedWorkNode],
    initial_content: &[GpuPreparedInitialContent],
) -> Result<(), GpuWorkGraphError> {
    let participating = nodes
        .iter()
        .flat_map(|prepared| prepared.node().accesses().iter())
        .map(|access| access.resource_identity())
        .chain(
            initial_content
                .iter()
                .map(GpuPreparedInitialContent::resource_identity),
        )
        .collect::<BTreeSet<_>>();
    for &target in targets {
        if !participating.contains(&target) {
            return Err(graph_error(
                "validate retained reconstruction participation",
                graph_label,
                GraphErrorOrigin::new(None, None),
                None,
                Some(target),
                GpuWorkGraphCause::OperationAccessContradiction,
                "make each reconstruction target participate in canonical initialization or accepted work",
            ));
        }
    }
    Ok(())
}

fn register_fragment_resources(
    graph_label: &str,
    fragment: &GpuWorkFragment,
    declared: &mut BTreeMap<GpuWorkResourceId, GpuResourceRef>,
    storage: &mut BTreeMap<GpuWorkResourceId, GpuResourceRef>,
) -> Result<BTreeSet<GpuWorkResourceId>, GpuWorkGraphError> {
    let mut fragment_resource_identities = BTreeSet::new();
    for resource in fragment.resources() {
        let identity = resource.diagnostic_identity();
        if declared
            .get(&identity)
            .is_some_and(|existing| existing != resource)
        {
            return Err(graph_error(
                "register GPU work resource",
                graph_label,
                GraphErrorOrigin::new(Some(fragment), None),
                None,
                Some(identity),
                GpuWorkGraphCause::UnknownIdentity,
                "use one kind-preserving handle for each logical resource identity",
            ));
        }
        declared.entry(identity).or_insert_with(|| resource.clone());
        let storage_resource = canonical_storage_resource(resource);
        let storage_identity = storage_identity(&storage_resource);
        if storage
            .get(&storage_identity)
            .is_some_and(|existing| existing != &storage_resource)
        {
            return Err(graph_error(
                "register normalized GPU storage resource",
                graph_label,
                GraphErrorOrigin::new(Some(fragment), None),
                None,
                Some(storage_identity),
                GpuWorkGraphCause::UnknownIdentity,
                "retain one descriptor and kind for each normalized storage identity",
            ));
        }
        storage.entry(storage_identity).or_insert(storage_resource);
        fragment_resource_identities.insert(resource.diagnostic_identity());
    }
    Ok(fragment_resource_identities)
}

fn validate_node_resources(
    graph_label: &str,
    fragment: &GpuWorkFragment,
    node: &GpuWorkNode,
    fragment_resource_identities: &BTreeSet<GpuWorkResourceId>,
) -> Result<(), GpuWorkGraphError> {
    node.operation().validate_shape().map_err(|source| {
        GpuWorkGraphError::with_source(
            "validate GPU work operation",
            GpuWorkGraphErrorContext::new(
                graph_label,
                Some(fragment.label().as_str().to_string()),
                Some(node.label().as_str().to_string()),
                None,
                source.resource(),
                None,
                Some(node.provenance().clone()),
            ),
            GpuWorkGraphCause::OperationAccessContradiction,
            "retain the checked operation shape accepted during fragment authoring",
            GpuWorkGraphErrorSource::Operation(source),
        )
    })?;
    for access in node.accesses() {
        let identity = access.declared_resource_identity();
        if !fragment_resource_identities.contains(&identity) {
            return Err(graph_error(
                "validate GPU work-node resource identity",
                graph_label,
                GraphErrorOrigin::new(Some(fragment), Some(node)),
                None,
                Some(identity),
                GpuWorkGraphCause::UnknownIdentity,
                "declare every accessed typed resource in its immutable fragment",
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod lazy_diagnostics_tests {
    use super::*;
    use crate::api::{
        GpuBufferDescriptor, GpuBufferInitialization, GpuBufferRegion, GpuBufferUsage,
        GpuClearOperation, GpuReconstruction, GpuResourceLifetime, GpuResourceScope,
    };

    fn two_writer_fragment() -> GpuWorkFragment {
        let mut resources = GpuResourceScope::new();
        let buffer = resources
            .buffer(
                GpuBufferDescriptor::ordinary_owned(
                    "G3 diagnostic source",
                    GpuResourceLifetime::Transient,
                    GpuReconstruction::SourceBacked,
                    16,
                    [GpuBufferUsage::CopyDestination],
                    GpuBufferInitialization::Uninitialized,
                )
                .unwrap(),
            )
            .unwrap();
        GpuWorkFragment::build("two live data writers", |builder| {
            builder.operation(
                "first typed write",
                GpuClearOperation::buffer_zero(GpuBufferRegion::whole(&buffer).unwrap()).unwrap(),
            )?;
            builder.operation(
                "second typed write",
                GpuClearOperation::buffer_zero(GpuBufferRegion::whole(&buffer).unwrap()).unwrap(),
            )?;
            Ok(())
        })
        .unwrap()
    }

    #[test]
    fn diagnostic_observation_retains_order_and_clone_before_after_materialization() {
        let prepared = GpuPreparedWorkGraph::prepare(
            GpuResourceLabel::new("unobserved G3 diagnostics").unwrap(),
            [two_writer_fragment()],
        )
        .unwrap();
        assert_eq!(prepared.dependencies().len(), 1);
        assert!(
            prepared.diagnostics.get().is_none(),
            "G3 must not eagerly clone dependency reasons into diagnostic entries"
        );
        let before = prepared.clone();
        assert!(before.diagnostics.get().is_none());
        let observed = prepared.diagnostics();
        assert_eq!(
            observed[0],
            GpuPreparedWorkDiagnostic::Dependency(prepared.dependencies()[0].clone())
        );
        let initial_address = observed.as_ptr();
        assert_eq!(prepared.diagnostics().as_ptr(), initial_address);
        assert!(
            before.diagnostics.get().is_none(),
            "observing one clone must not mutate another clone"
        );
        assert_eq!(before.diagnostics(), prepared.diagnostics());
        let after = prepared.clone();
        assert_eq!(after.diagnostics(), prepared.diagnostics());
        let display = format!("{after:?}");
        assert!(
            display.contains("diagnostics: ["),
            "derived Debug's diagnostic field remains observable"
        );
    }

    #[test]
    fn empty_graph_exposes_same_lazy_empty_diagnostics_slice() {
        let fragment = GpuWorkFragment::build("empty graph", |_| Ok(())).unwrap();
        let prepared =
            GpuPreparedWorkGraph::prepare(GpuResourceLabel::new("empty G3").unwrap(), [fragment])
                .unwrap();
        assert!(prepared.diagnostics.get().is_none());
        assert!(prepared.dependencies().is_empty());
        assert!(prepared.diagnostics().is_empty());
        assert!(prepared.diagnostics.get().is_some());
    }
}
