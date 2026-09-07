use super::super::{
    GpuCapabilityFeature, GpuCapabilityRequirement, GpuCapabilityRequirements, GpuResourceLabel,
    GpuResourceRef, GpuRetainedInitializationSeed, GpuWorkGraphCause, GpuWorkGraphError,
    GpuWorkGraphErrorContext, GpuWorkGraphErrorSource, GpuWorkResourceId,
};
use super::{
    authoring::{GpuWorkFragment, GpuWorkNode, GpuWorkOutput},
    composition::{
        bind_imports, collect_output_bindings, topological_fragment_order,
        validate_boundary_access_intents,
    },
    coverage::{GpuInitialCoverage, canonical_storage_resource, storage_identity},
    dependency::{GpuDependencyReason, GpuWorkDependency},
    diagnostics::{GpuPreparedWorkDiagnostic, GraphErrorOrigin, graph_error},
    hazards::{
        add_explicit_orders, infer_cross_fragment_hazards, infer_fragment_hazards,
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
use std::collections::{BTreeMap, BTreeSet};

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
#[derive(Debug, Clone)]
pub struct GpuPreparedWorkGraph {
    label: GpuResourceLabel,
    nodes: Vec<GpuPreparedWorkNode>,
    topological_order: Vec<GpuPreparedWorkNodeId>,
    dependencies: Vec<GpuWorkDependency>,
    initialization: Vec<GpuPreparedResourceInitialization>,
    requirements: GpuCapabilityRequirements,
    outputs: Vec<GpuWorkOutput>,
    diagnostics: Vec<GpuPreparedWorkDiagnostic>,
    initial_content: Vec<GpuPreparedInitialContent>,
    retained_seed: Vec<GpuRetainedInitializationSeed>,
    reconstruction_targets: BTreeSet<GpuWorkResourceId>,
    failure_preserved_coverage: BTreeMap<GpuWorkResourceId, GpuInitialCoverage>,
}

impl GpuPreparedWorkGraph {
    pub fn prepare(
        label: GpuResourceLabel,
        fragments: impl IntoIterator<Item = GpuWorkFragment>,
    ) -> Result<Self, GpuWorkGraphError> {
        Self::prepare_with_retained_coverage(label, fragments, &[])
    }

    pub(crate) fn prepare_with_retained_coverage(
        label: GpuResourceLabel,
        fragments: impl IntoIterator<Item = GpuWorkFragment>,
        retained_coverage: &[GpuRetainedInitializationSeed],
    ) -> Result<Self, GpuWorkGraphError> {
        Self::prepare_with_retained_coverage_and_reconstruction(
            label,
            fragments,
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
        let fragments = fragments.into_iter().collect::<Vec<_>>();
        let graph_label = label.as_str();
        let mut declared_resources = BTreeMap::<GpuWorkResourceId, GpuResourceRef>::new();
        let mut storage_resources = BTreeMap::<GpuWorkResourceId, GpuResourceRef>::new();
        let mut prepared_nodes = Vec::new();
        let mut node_locations = BTreeMap::<GpuPreparedWorkNodeId, (usize, usize)>::new();

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

        let output_bindings = collect_output_bindings(graph_label, &fragments)?;
        let (import_bindings, relations) = bind_imports(graph_label, &fragments, &output_bindings)?;
        validate_boundary_access_intents(graph_label, &fragments)?;
        let mut inferred_edges = BTreeMap::<
            (GpuPreparedWorkNodeId, GpuPreparedWorkNodeId),
            BTreeSet<GpuDependencyReason>,
        >::new();
        infer_fragment_hazards(graph_label, &fragments, &mut inferred_edges)?;
        infer_cross_fragment_hazards(graph_label, &fragments, &relations, &mut inferred_edges)?;
        add_explicit_orders(graph_label, &fragments, &mut inferred_edges)?;
        let topological_order =
            topological_node_order(graph_label, &fragments, &node_locations, &inferred_edges)?;
        let fragment_order = topological_fragment_order(graph_label, &fragments, &import_bindings)?;
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
        let mut diagnostics = dependencies
            .iter()
            .cloned()
            .map(GpuPreparedWorkDiagnostic::Dependency)
            .collect::<Vec<_>>();
        diagnostics.extend(initialization_diagnostics);
        diagnostics.extend(
            outputs
                .iter()
                .map(|output| GpuPreparedWorkDiagnostic::Output {
                    export_key: output.relationship().export_key().clone(),
                    resource: storage_identity(output.relationship().resource()),
                }),
        );

        Ok(Self {
            label,
            nodes: prepared_nodes,
            topological_order,
            dependencies,
            initialization,
            requirements,
            outputs,
            diagnostics,
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
        &self.diagnostics
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
