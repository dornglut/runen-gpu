use super::super::{
    GpuAccessError, GpuBufferAccess, GpuBufferAccessKind, GpuCapabilityRequirementError,
    GpuCapabilityRequirements, GpuComputeOperation, GpuExportKey, GpuExportRelationship,
    GpuResourceAccess, GpuResourceAccessIntent, GpuResourceDescriptorError, GpuResourceLabel,
    GpuResourceProvenance, GpuResourceRef, GpuTextureAccess, GpuTextureAccessKind,
    GpuTextureAccessResource, GpuWorkAuthoringCause, GpuWorkAuthoringError,
    GpuWorkAuthoringErrorContext, GpuWorkAuthoringErrorSource, GpuWorkNodeKind, GpuWorkOperation,
    GpuWorkResourceId,
};
use super::{
    coverage::{GpuInitialCoverage, GpuWorkResourceInput},
    dependency::access_intersection,
    identity::GpuWorkNodeId,
};
use core::num::NonZeroU64;
use std::{
    collections::{BTreeMap, btree_map::Entry},
    sync::Arc,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum GpuExecutionPreference {
    #[default]
    Automatic,
    ComputePreferred,
    GraphicsRequired,
    TransferPreferred,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuWorkImport {
    resource: GpuResourceRef,
    producer: GpuExportKey,
    required_initial_access: GpuResourceAccessIntent,
    provenance: GpuResourceProvenance,
}

impl GpuWorkImport {
    pub fn new(
        resource: GpuResourceRef,
        producer: GpuExportKey,
        required_initial_access: GpuResourceAccessIntent,
        provenance: GpuResourceProvenance,
    ) -> Self {
        Self {
            resource,
            producer,
            required_initial_access,
            provenance,
        }
    }

    pub fn resource(&self) -> &GpuResourceRef {
        &self.resource
    }

    pub fn producer(&self) -> &GpuExportKey {
        &self.producer
    }

    pub const fn required_initial_access(&self) -> GpuResourceAccessIntent {
        self.required_initial_access
    }

    pub fn provenance(&self) -> &GpuResourceProvenance {
        &self.provenance
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuWorkOutput {
    relationship: GpuExportRelationship,
    final_initialized_coverage: GpuInitialCoverage,
}

impl GpuWorkOutput {
    pub fn new(
        relationship: GpuExportRelationship,
        final_initialized_coverage: GpuInitialCoverage,
    ) -> Result<Self, GpuWorkAuthoringError> {
        if relationship.resource() != final_initialized_coverage.resource() {
            return Err(GpuWorkAuthoringError::invalid(
                "construct GPU work output",
                GpuWorkAuthoringErrorContext::new(
                    None,
                    None,
                    None,
                    Some(relationship.resource().diagnostic_identity()),
                    Some(relationship.provenance().clone()),
                ),
                GpuWorkAuthoringCause::InvalidCoverage,
                "bind final coverage checked against the exported kind-preserving resource",
            ));
        }
        Ok(Self {
            relationship,
            final_initialized_coverage,
        })
    }

    pub fn relationship(&self) -> &GpuExportRelationship {
        &self.relationship
    }

    pub fn final_initialized_coverage(&self) -> &GpuInitialCoverage {
        &self.final_initialized_coverage
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuExplicitOrder {
    before: GpuWorkNodeId,
    after: GpuWorkNodeId,
    reason: String,
}

impl GpuExplicitOrder {
    pub fn new(
        before: &GpuWorkNodeId,
        after: &GpuWorkNodeId,
        reason: impl Into<String>,
    ) -> Result<Self, GpuWorkAuthoringError> {
        let reason = reason.into();
        let reason = reason.trim();
        if reason.is_empty() || before == after {
            return Err(GpuWorkAuthoringError::invalid(
                "construct explicit GPU work order",
                GpuWorkAuthoringErrorContext::new(None, None, Some(before.clone()), None, None),
                GpuWorkAuthoringCause::InvalidExplicitOrder,
                "provide distinct fragment-local nodes and a nonempty non-data reason",
            ));
        }
        if !Arc::ptr_eq(&before.fragment_identity, &after.fragment_identity) {
            return Err(GpuWorkAuthoringError::invalid(
                "construct explicit GPU work order",
                GpuWorkAuthoringErrorContext::new(None, None, Some(before.clone()), None, None),
                GpuWorkAuthoringCause::ForeignIdentity,
                "order only nodes allocated by the same fragment builder",
            ));
        }
        Ok(Self {
            before: before.clone(),
            after: after.clone(),
            reason: reason.to_string(),
        })
    }

    pub fn before(&self) -> &GpuWorkNodeId {
        &self.before
    }

    pub fn after(&self) -> &GpuWorkNodeId {
        &self.after
    }

    pub fn reason(&self) -> &str {
        &self.reason
    }
}

/// Immutable checked work node. Operation kind is derived from `operation()`;
/// there is no duplicate mutable kind field.
///
/// ```compile_fail
/// use runen_gpu::{GpuWorkNode, GpuWorkNodeKind};
///
/// fn bypass(node: &mut GpuWorkNode) {
///     node.kind = GpuWorkNodeKind::Copy;
/// }
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuWorkNode {
    id: GpuWorkNodeId,
    operation: GpuWorkOperation,
    pub(super) accesses: Vec<GpuResourceAccess>,
    // Initialization requirements belong to caller-declared reads, before the
    // hazard access normalizer can merge them with operation-derived access.
    pub(super) caller_readable_accesses: Vec<GpuResourceAccess>,
    requirements: GpuCapabilityRequirements,
    execution_preference: GpuExecutionPreference,
    label: GpuResourceLabel,
    provenance: GpuResourceProvenance,
}

impl GpuWorkNode {
    pub fn id(&self) -> &GpuWorkNodeId {
        &self.id
    }

    pub fn operation(&self) -> &GpuWorkOperation {
        &self.operation
    }

    pub fn kind(&self) -> GpuWorkNodeKind {
        self.operation.kind()
    }

    pub fn accesses(&self) -> &[GpuResourceAccess] {
        &self.accesses
    }

    pub(super) fn caller_readable_accesses(&self) -> &[GpuResourceAccess] {
        &self.caller_readable_accesses
    }

    pub fn requirements(&self) -> &GpuCapabilityRequirements {
        &self.requirements
    }

    pub const fn execution_preference(&self) -> GpuExecutionPreference {
        self.execution_preference
    }

    pub fn label(&self) -> &GpuResourceLabel {
        &self.label
    }

    pub fn provenance(&self) -> &GpuResourceProvenance {
        &self.provenance
    }
}

/// Immutable authored fragment. Mutation is available only through the
/// checked builder transaction.
///
/// ```compile_fail
/// use runen_gpu::GpuWorkFragment;
///
/// fn mutate(fragment: &mut GpuWorkFragment) {
///     fragment.nodes.clear();
/// }
/// ```
#[derive(Debug, Clone)]
pub struct GpuWorkFragment {
    pub(super) identity: Arc<()>,
    label: GpuResourceLabel,
    resources: Vec<GpuResourceRef>,
    inputs: Vec<GpuWorkResourceInput>,
    imports: Vec<GpuWorkImport>,
    outputs: Vec<GpuWorkOutput>,
    pub(super) nodes: Vec<GpuWorkNode>,
    explicit_orders: Vec<GpuExplicitOrder>,
    provenance: GpuResourceProvenance,
}

impl GpuWorkFragment {
    /// Authors an immutable fragment from complete typed operations.
    ///
    /// ```
    /// use runen_gpu::*;
    /// # fn author_simulation(
    /// #     compute: GpuComputeOperation,
    /// # ) -> Result<GpuWorkFragment, GpuWorkAuthoringError> {
    /// let simulation = GpuWorkFragment::build("simulation.update", |work| {
    ///     work.compute("integrate", compute)?;
    ///     Ok(())
    /// })?;
    /// # Ok(simulation)
    /// # }
    /// ```
    pub fn build<L, F>(label: L, author: F) -> Result<Self, GpuWorkAuthoringError>
    where
        L: AsRef<str>,
        F: FnOnce(&mut GpuWorkFragmentBuilder) -> Result<(), GpuWorkAuthoringError>,
    {
        let label = checked_fragment_label(label.as_ref())?;
        let provenance = GpuResourceProvenance::new(label.clone(), None, None);
        Self::build_with_provenance(label, provenance, author)
    }

    pub fn build_with_provenance<F>(
        label: GpuResourceLabel,
        provenance: GpuResourceProvenance,
        author: F,
    ) -> Result<Self, GpuWorkAuthoringError>
    where
        F: FnOnce(&mut GpuWorkFragmentBuilder) -> Result<(), GpuWorkAuthoringError>,
    {
        let mut builder = GpuWorkFragmentBuilder::new(label, provenance);
        author(&mut builder)?;
        builder.finish()
    }

    pub fn label(&self) -> &GpuResourceLabel {
        &self.label
    }

    pub fn resources(&self) -> &[GpuResourceRef] {
        &self.resources
    }

    pub fn inputs(&self) -> &[GpuWorkResourceInput] {
        &self.inputs
    }

    pub fn imports(&self) -> &[GpuWorkImport] {
        &self.imports
    }
    pub fn outputs(&self) -> &[GpuWorkOutput] {
        &self.outputs
    }

    pub fn nodes(&self) -> &[GpuWorkNode] {
        &self.nodes
    }

    pub fn explicit_orders(&self) -> &[GpuExplicitOrder] {
        &self.explicit_orders
    }

    pub fn provenance(&self) -> &GpuResourceProvenance {
        &self.provenance
    }
}

#[derive(Debug)]
pub struct GpuWorkFragmentBuilder {
    identity: Arc<()>,
    label: GpuResourceLabel,
    resources: BTreeMap<GpuWorkResourceId, GpuResourceRef>,
    inputs: Vec<GpuWorkResourceInput>,
    imports: Vec<GpuWorkImport>,
    outputs: Vec<GpuWorkOutput>,
    nodes: Vec<GpuWorkNode>,
    explicit_orders: Vec<GpuExplicitOrder>,
    next_node: Option<NonZeroU64>,
    provenance: GpuResourceProvenance,
}

impl GpuWorkFragmentBuilder {
    pub fn new(label: GpuResourceLabel, provenance: GpuResourceProvenance) -> Self {
        Self {
            identity: Arc::new(()),
            label,
            resources: BTreeMap::new(),
            inputs: Vec::new(),
            imports: Vec::new(),
            outputs: Vec::new(),
            nodes: Vec::new(),
            explicit_orders: Vec::new(),
            next_node: NonZeroU64::new(1),
            provenance,
        }
    }

    pub fn declare_resource(
        &mut self,
        resource: GpuResourceRef,
    ) -> Result<(), GpuWorkAuthoringError> {
        let identity = resource.diagnostic_identity();
        match self.resources.entry(identity) {
            Entry::Vacant(entry) => {
                entry.insert(resource);
                Ok(())
            }
            Entry::Occupied(_) => Err(GpuWorkAuthoringError::invalid(
                "declare GPU work resource",
                GpuWorkAuthoringErrorContext::new(
                    Some(self.label.as_str().to_string()),
                    None,
                    None,
                    Some(identity),
                    None,
                ),
                GpuWorkAuthoringCause::DuplicateResource,
                "declare each kind-preserving resource once per fragment",
            )),
        }
    }

    pub fn add_input(&mut self, input: GpuWorkResourceInput) -> Result<(), GpuWorkAuthoringError> {
        self.require_resource(input.resource(), "add GPU work-resource input")?;
        let identity = input.resource().diagnostic_identity();
        if self
            .inputs
            .iter()
            .any(|existing| existing.resource().diagnostic_identity() == identity)
        {
            return Err(self.error(
                "add GPU work-resource input",
                None,
                Some(identity),
                GpuWorkAuthoringCause::DuplicateInput,
                "combine exact entry coverage into one input per resource",
                Some(input.provenance().clone()),
            ));
        }
        self.inputs.push(input);
        Ok(())
    }

    pub fn add_import(&mut self, import: GpuWorkImport) -> Result<(), GpuWorkAuthoringError> {
        self.require_resource(import.resource(), "add GPU work import")?;
        if self
            .imports
            .iter()
            .any(|existing| existing.producer() == import.producer())
        {
            return Err(self.error(
                "add GPU work import",
                None,
                Some(import.resource().diagnostic_identity()),
                GpuWorkAuthoringCause::DuplicateImport,
                "bind each producer export key once per fragment",
                Some(import.provenance().clone()),
            ));
        }
        self.imports.push(import);
        Ok(())
    }

    pub fn add_output(&mut self, output: GpuWorkOutput) -> Result<(), GpuWorkAuthoringError> {
        self.require_resource(output.relationship().resource(), "add GPU work output")?;
        let key = output.relationship().export_key();
        if self
            .outputs
            .iter()
            .any(|existing| existing.relationship().export_key() == key)
        {
            return Err(self.error(
                "add GPU work output",
                None,
                Some(output.relationship().resource().diagnostic_identity()),
                GpuWorkAuthoringCause::DuplicateExportKey,
                "use one semantic export key for one fragment output",
                Some(output.relationship().provenance().clone()),
            ));
        }
        self.outputs.push(output);
        Ok(())
    }

    pub fn compute<L>(
        &mut self,
        label: L,
        compute: GpuComputeOperation,
    ) -> Result<GpuWorkNodeId, GpuWorkAuthoringError>
    where
        L: AsRef<str>,
    {
        self.add_checked_lexical_operation(label, GpuWorkOperation::Compute(compute))
    }

    pub(crate) fn add_checked_lexical_operation<L>(
        &mut self,
        label: L,
        operation: GpuWorkOperation,
    ) -> Result<GpuWorkNodeId, GpuWorkAuthoringError>
    where
        L: AsRef<str>,
    {
        let label = checked_node_label(&self.label, label.as_ref())?;
        self.add_lexical_operation(label, operation)
    }

    pub(crate) fn add_lexical_operation(
        &mut self,
        label: GpuResourceLabel,
        operation: GpuWorkOperation,
    ) -> Result<GpuWorkNodeId, GpuWorkAuthoringError> {
        let provenance = GpuResourceProvenance::new(label.clone(), None, None);
        let derived = operation.derived_accesses().map_err(|source| {
            GpuWorkAuthoringError::with_source(
                "derive lexical GPU operation resources",
                GpuWorkAuthoringErrorContext::new(
                    Some(self.label.as_str().to_string()),
                    Some(label.as_str().to_string()),
                    None,
                    source.resource(),
                    Some(provenance.clone()),
                ),
                GpuWorkAuthoringCause::OperationAccessContradiction,
                "use one internally consistent checked operation",
                GpuWorkAuthoringErrorSource::Operation(source),
            )
        })?;

        let mut inserted_resources = Vec::with_capacity(derived.len());
        for access in &derived {
            match self.declare_lexical_resource(
                declared_resource_for_access(access),
                &label,
                &provenance,
            ) {
                Ok(Some(identity)) => inserted_resources.push(identity),
                Ok(None) => {}
                Err(error) => {
                    self.rollback_lexical_resources(&inserted_resources);
                    return Err(error);
                }
            }
        }
        let result = self.add_node(
            label,
            operation,
            [],
            GpuCapabilityRequirements::new(),
            GpuExecutionPreference::Automatic,
            provenance,
        );
        if result.is_err() {
            self.rollback_lexical_resources(&inserted_resources);
        }
        result
    }

    pub fn add_node(
        &mut self,
        label: GpuResourceLabel,
        operation: GpuWorkOperation,
        accesses: impl IntoIterator<Item = GpuResourceAccess>,
        caller_requirements: GpuCapabilityRequirements,
        execution_preference: GpuExecutionPreference,
        provenance: GpuResourceProvenance,
    ) -> Result<GpuWorkNodeId, GpuWorkAuthoringError> {
        operation.validate_shape().map_err(|source| {
            GpuWorkAuthoringError::with_source(
                "add GPU work node",
                GpuWorkAuthoringErrorContext::new(
                    Some(self.label.as_str().to_string()),
                    Some(label.as_str().to_string()),
                    None,
                    source.resource(),
                    Some(provenance.clone()),
                ),
                GpuWorkAuthoringCause::OperationAccessContradiction,
                "construct the operation through its checked operation constructor",
                GpuWorkAuthoringErrorSource::Operation(source),
            )
        })?;
        let derived = operation.derived_accesses().map_err(|source| {
            GpuWorkAuthoringError::with_source(
                "derive GPU work-node access",
                GpuWorkAuthoringErrorContext::new(
                    Some(self.label.as_str().to_string()),
                    Some(label.as_str().to_string()),
                    None,
                    source.resource(),
                    Some(provenance.clone()),
                ),
                GpuWorkAuthoringCause::OperationAccessContradiction,
                "use one internally consistent checked operation",
                GpuWorkAuthoringErrorSource::Operation(source),
            )
        })?;
        let caller = accesses.into_iter().collect::<Vec<_>>();
        for access in derived.iter().chain(&caller) {
            let identity = access.declared_resource_identity();
            if !self.resources.contains_key(&identity) {
                return Err(self.error(
                    "add GPU work-node access",
                    Some(label.as_str().to_string()),
                    Some(identity),
                    GpuWorkAuthoringCause::UnknownIdentity,
                    "declare every operation-derived and caller-declared resource in the fragment",
                    Some(provenance.clone()),
                ));
            }
        }
        let caller_readable_accesses = preserve_caller_readable_accesses(&caller);
        let normalized =
            normalize_node_accesses(&self.label, &label, &provenance, derived, caller)?;
        validate_indexed_draw_access(&self.label, &label, &provenance, &operation, &normalized)?;
        let requirements = merge_node_requirements(
            &self.label,
            &label,
            &provenance,
            &operation,
            &normalized,
            &caller_requirements,
        )?;
        let local = self.next_node.ok_or_else(|| {
            self.error(
                "allocate GPU work-node identity",
                Some(label.as_str().to_string()),
                None,
                GpuWorkAuthoringCause::IdentityExhausted,
                "split work into another fragment before exhausting local node identity",
                Some(provenance.clone()),
            )
        })?;
        self.next_node = local.get().checked_add(1).and_then(NonZeroU64::new);
        let id = GpuWorkNodeId::new(&self.identity, local);
        self.nodes.push(GpuWorkNode {
            id: id.clone(),
            operation,
            accesses: normalized,
            caller_readable_accesses,
            requirements,
            execution_preference,
            label,
            provenance,
        });
        Ok(id)
    }

    pub fn add_explicit_order(
        &mut self,
        order: GpuExplicitOrder,
    ) -> Result<(), GpuWorkAuthoringError> {
        for endpoint in [&order.before, &order.after] {
            if !endpoint.belongs_to(&self.identity) {
                return Err(self.error(
                    "add explicit GPU work order",
                    None,
                    None,
                    GpuWorkAuthoringCause::ForeignIdentity,
                    "use endpoints allocated by this fragment builder",
                    None,
                ));
            }
            if !self.nodes.iter().any(|node| node.id() == endpoint) {
                return Err(self.error(
                    "add explicit GPU work order",
                    None,
                    None,
                    GpuWorkAuthoringCause::UnknownIdentity,
                    "add both endpoint nodes before adding explicit order",
                    None,
                ));
            }
        }
        if self.explicit_orders.iter().any(|existing| {
            existing.before() == order.before() && existing.after() == order.after()
        }) {
            return Err(self.error(
                "add explicit GPU work order",
                None,
                None,
                GpuWorkAuthoringCause::DuplicateExplicitOrder,
                "retain one non-data reason for each explicit directed edge",
                None,
            ));
        }
        self.explicit_orders.push(order);
        Ok(())
    }

    pub fn finish(self) -> Result<GpuWorkFragment, GpuWorkAuthoringError> {
        Ok(GpuWorkFragment {
            identity: self.identity,
            label: self.label,
            resources: self.resources.into_values().collect(),
            inputs: self.inputs,
            imports: self.imports,
            outputs: self.outputs,
            nodes: self.nodes,
            explicit_orders: self.explicit_orders,
            provenance: self.provenance,
        })
    }

    fn declare_lexical_resource(
        &mut self,
        resource: GpuResourceRef,
        node_label: &GpuResourceLabel,
        provenance: &GpuResourceProvenance,
    ) -> Result<Option<GpuWorkResourceId>, GpuWorkAuthoringError> {
        let identity = resource.diagnostic_identity();
        match self.resources.entry(identity) {
            Entry::Vacant(entry) => {
                entry.insert(resource);
                Ok(Some(identity))
            }
            Entry::Occupied(entry) if entry.get() == &resource => Ok(None),
            Entry::Occupied(_) => Err(GpuWorkAuthoringError::invalid(
                "declare lexical GPU work resource",
                GpuWorkAuthoringErrorContext::new(
                    Some(self.label.as_str().to_string()),
                    Some(node_label.as_str().to_string()),
                    None,
                    Some(identity),
                    Some(provenance.clone()),
                ),
                GpuWorkAuthoringCause::InvalidResourceKind,
                "use the exact kind-preserving resource already declared by the fragment",
            )),
        }
    }

    fn rollback_lexical_resources(&mut self, inserted: &[GpuWorkResourceId]) {
        for identity in inserted.iter().rev() {
            let removed = self.resources.remove(identity);
            debug_assert!(removed.is_some());
        }
    }

    fn require_resource(
        &self,
        resource: &GpuResourceRef,
        operation: &'static str,
    ) -> Result<(), GpuWorkAuthoringError> {
        match self.resources.get(&resource.diagnostic_identity()) {
            Some(declared) if declared == resource => Ok(()),
            Some(_) => Err(self.error(
                operation,
                None,
                Some(resource.diagnostic_identity()),
                GpuWorkAuthoringCause::InvalidResourceKind,
                "use the exact kind-preserving resource declared by the fragment",
                Some(resource.common().provenance().clone()),
            )),
            None => Err(self.error(
                operation,
                None,
                Some(resource.diagnostic_identity()),
                GpuWorkAuthoringCause::UnknownIdentity,
                "declare the resource before binding fragment evidence",
                Some(resource.common().provenance().clone()),
            )),
        }
    }

    fn error(
        &self,
        operation: &'static str,
        node_label: Option<String>,
        resource: Option<GpuWorkResourceId>,
        cause: GpuWorkAuthoringCause,
        correction: &'static str,
        provenance: Option<GpuResourceProvenance>,
    ) -> GpuWorkAuthoringError {
        GpuWorkAuthoringError::invalid(
            operation,
            GpuWorkAuthoringErrorContext::new(
                Some(self.label.as_str().to_string()),
                node_label,
                None,
                resource,
                provenance,
            ),
            cause,
            correction,
        )
    }
}

fn declared_resource_for_access(access: &GpuResourceAccess) -> GpuResourceRef {
    match access {
        GpuResourceAccess::Buffer(access) => access.buffer().clone().into(),
        GpuResourceAccess::Texture(access) => match access.resource() {
            GpuTextureAccessResource::Texture(texture) => texture.clone().into(),
            GpuTextureAccessResource::TextureView(view) => view.clone().into(),
        },
        GpuResourceAccess::Query(access) => access.query_set().clone().into(),
        GpuResourceAccess::Sampler(access) => access.sampler().clone().into(),
    }
}

pub(super) fn normalize_node_accesses(
    fragment_label: &GpuResourceLabel,
    node_label: &GpuResourceLabel,
    provenance: &GpuResourceProvenance,
    mut derived: Vec<GpuResourceAccess>,
    mut caller: Vec<GpuResourceAccess>,
) -> Result<Vec<GpuResourceAccess>, GpuWorkAuthoringError> {
    derived.sort();
    derived.dedup();
    caller.sort();
    caller.dedup();
    if let Some(duplicate) = caller.iter().find(|access| derived.contains(access)) {
        return Err(node_authoring_error(
            "normalize GPU work-node access",
            fragment_label,
            node_label,
            Some(duplicate.resource_identity()),
            GpuWorkAuthoringCause::OperationAccessContradiction,
            "remove caller access already derived by the typed operation",
            provenance,
        ));
    }
    let mut normalized = Vec::<GpuResourceAccess>::new();
    for access in derived.into_iter().chain(caller) {
        insert_normalized_access(
            fragment_label,
            node_label,
            provenance,
            &mut normalized,
            access,
        )?;
    }
    normalized.sort();
    Ok(normalized)
}

pub(super) fn preserve_caller_readable_accesses(
    caller: &[GpuResourceAccess],
) -> Vec<GpuResourceAccess> {
    let mut readable = caller
        .iter()
        .filter(|access| access.reads())
        .cloned()
        .collect::<Vec<_>>();
    readable.sort();
    readable.dedup();
    readable
}

fn insert_normalized_access(
    fragment_label: &GpuResourceLabel,
    node_label: &GpuResourceLabel,
    provenance: &GpuResourceProvenance,
    normalized: &mut Vec<GpuResourceAccess>,
    mut access: GpuResourceAccess,
) -> Result<(), GpuWorkAuthoringError> {
    let mut index = 0;
    while index < normalized.len() {
        let existing = &normalized[index];
        if existing == &access {
            return Ok(());
        }
        if access_intersection(existing, &access).is_none() {
            index += 1;
            continue;
        }
        if let Some(merged) = merge_storage_access(existing, &access).map_err(|source| {
            GpuWorkAuthoringError::with_source(
                "normalize GPU storage access",
                GpuWorkAuthoringErrorContext::new(
                    Some(fragment_label.as_str().to_string()),
                    Some(node_label.as_str().to_string()),
                    None,
                    Some(access.resource_identity()),
                    Some(provenance.clone()),
                ),
                GpuWorkAuthoringCause::OperationAccessContradiction,
                "declare compatible exact storage read/write coverage",
                GpuWorkAuthoringErrorSource::Access(source),
            )
        })? {
            normalized.remove(index);
            access = merged;
            index = 0;
            continue;
        }
        if !existing.writes() && !access.writes() {
            index += 1;
            continue;
        }
        return Err(node_authoring_error(
            "normalize GPU work-node access",
            fragment_label,
            node_label,
            Some(access.resource_identity()),
            GpuWorkAuthoringCause::IncompatibleSameNodeAccess,
            "split incompatible overlapping roles into ordered nodes or use exact storage read-write access",
            provenance,
        ));
    }
    normalized.push(access);
    Ok(())
}

fn merge_storage_access(
    left: &GpuResourceAccess,
    right: &GpuResourceAccess,
) -> Result<Option<GpuResourceAccess>, GpuAccessError> {
    match (left, right) {
        (GpuResourceAccess::Buffer(left), GpuResourceAccess::Buffer(right))
            if left.buffer() == right.buffer()
                && left.range() == right.range()
                && storage_buffer_kinds_merge(left.kind(), right.kind()) =>
        {
            GpuBufferAccess::new(
                left.buffer(),
                left.range(),
                GpuBufferAccessKind::StorageReadWrite,
            )
            .map(GpuResourceAccess::Buffer)
            .map(Some)
        }
        (GpuResourceAccess::Texture(left), GpuResourceAccess::Texture(right))
            if left.normalized_texture() == right.normalized_texture()
                && left.normalized_subresources() == right.normalized_subresources()
                && storage_texture_kinds_merge(left.kind(), right.kind()) =>
        {
            GpuTextureAccess::new(
                GpuTextureAccessResource::Texture(left.normalized_texture().clone()),
                left.normalized_subresources(),
                GpuTextureAccessKind::StorageReadWrite,
            )
            .map(GpuResourceAccess::Texture)
            .map(Some)
        }
        _ => Ok(None),
    }
}

fn storage_buffer_kinds_merge(left: GpuBufferAccessKind, right: GpuBufferAccessKind) -> bool {
    use GpuBufferAccessKind::{StorageRead, StorageReadWrite, StorageWrite};
    matches!(
        (left, right),
        (StorageRead, StorageWrite)
            | (StorageWrite, StorageRead)
            | (StorageReadWrite, StorageRead)
            | (StorageReadWrite, StorageWrite)
            | (StorageRead, StorageReadWrite)
            | (StorageWrite, StorageReadWrite)
    )
}
fn storage_texture_kinds_merge(left: GpuTextureAccessKind, right: GpuTextureAccessKind) -> bool {
    use GpuTextureAccessKind::{StorageRead, StorageReadWrite, StorageWrite};
    matches!(
        (left, right),
        (StorageRead, StorageWrite)
            | (StorageWrite, StorageRead)
            | (StorageReadWrite, StorageRead)
            | (StorageReadWrite, StorageWrite)
            | (StorageRead, StorageReadWrite)
            | (StorageWrite, StorageReadWrite)
    )
}

fn validate_indexed_draw_access(
    fragment_label: &GpuResourceLabel,
    node_label: &GpuResourceLabel,
    provenance: &GpuResourceProvenance,
    operation: &GpuWorkOperation,
    accesses: &[GpuResourceAccess],
) -> Result<(), GpuWorkAuthoringError> {
    let GpuWorkOperation::Render(render) = operation else {
        return Ok(());
    };
    if render.draws().iter().any(|draw| draw.draw().is_indexed())
        && !accesses.iter().any(|access| {
            matches!(
                access,
                GpuResourceAccess::Buffer(access)
                    if access.kind() == GpuBufferAccessKind::IndexRead
            )
        })
    {
        return Err(node_authoring_error(
            "validate indexed GPU draw",
            fragment_label,
            node_label,
            None,
            GpuWorkAuthoringCause::OperationAccessContradiction,
            "declare a checked index-buffer access for indexed draw intent",
            provenance,
        ));
    }
    Ok(())
}

fn merge_node_requirements(
    fragment_label: &GpuResourceLabel,
    node_label: &GpuResourceLabel,
    provenance: &GpuResourceProvenance,
    operation: &GpuWorkOperation,
    accesses: &[GpuResourceAccess],
    caller: &GpuCapabilityRequirements,
) -> Result<GpuCapabilityRequirements, GpuWorkAuthoringError> {
    let mut requirements = operation.derived_requirements().map_err(|source| {
        requirement_authoring_error(fragment_label, node_label, provenance, source)
    })?;
    for access in accesses {
        let access_requirements = access.derived_requirements().map_err(|source| {
            requirement_authoring_error(fragment_label, node_label, provenance, source)
        })?;
        requirements = requirements.merge(&access_requirements).map_err(|source| {
            requirement_authoring_error(fragment_label, node_label, provenance, source)
        })?;
    }
    requirements = requirements.merge(caller).map_err(|source| {
        requirement_authoring_error(fragment_label, node_label, provenance, source)
    })?;
    Ok(requirements)
}

fn requirement_authoring_error(
    fragment_label: &GpuResourceLabel,
    node_label: &GpuResourceLabel,
    provenance: &GpuResourceProvenance,
    source: GpuCapabilityRequirementError,
) -> GpuWorkAuthoringError {
    GpuWorkAuthoringError::with_source(
        "merge GPU work-node capability requirements",
        GpuWorkAuthoringErrorContext::new(
            Some(fragment_label.as_str().to_string()),
            Some(node_label.as_str().to_string()),
            None,
            None,
            Some(provenance.clone()),
        ),
        GpuWorkAuthoringCause::MechanicalCapabilityContradiction,
        "remove a caller constraint that disables or ambiguously redefines a mechanically required capability",
        GpuWorkAuthoringErrorSource::Capability(source),
    )
}

fn node_authoring_error(
    operation: &'static str,
    fragment_label: &GpuResourceLabel,
    node_label: &GpuResourceLabel,
    resource: Option<GpuWorkResourceId>,
    cause: GpuWorkAuthoringCause,
    correction: &'static str,
    provenance: &GpuResourceProvenance,
) -> GpuWorkAuthoringError {
    GpuWorkAuthoringError::invalid(
        operation,
        GpuWorkAuthoringErrorContext::new(
            Some(fragment_label.as_str().to_string()),
            Some(node_label.as_str().to_string()),
            None,
            resource,
            Some(provenance.clone()),
        ),
        cause,
        correction,
    )
}

fn checked_fragment_label(value: &str) -> Result<GpuResourceLabel, GpuWorkAuthoringError> {
    GpuResourceLabel::new(value).map_err(|source| {
        label_authoring_error(
            "build GPU work fragment",
            None,
            None,
            source,
            "provide a non-empty fragment label",
        )
    })
}

fn checked_node_label(
    fragment_label: &GpuResourceLabel,
    value: &str,
) -> Result<GpuResourceLabel, GpuWorkAuthoringError> {
    GpuResourceLabel::new(value).map_err(|source| {
        label_authoring_error(
            "author lexical GPU work node",
            Some(fragment_label.as_str().to_string()),
            None,
            source,
            "provide a non-empty node label",
        )
    })
}

fn label_authoring_error(
    operation: &'static str,
    fragment_label: Option<String>,
    node_label: Option<String>,
    source: GpuResourceDescriptorError,
    correction: &'static str,
) -> GpuWorkAuthoringError {
    GpuWorkAuthoringError::with_source(
        operation,
        GpuWorkAuthoringErrorContext::new(fragment_label, node_label, None, None, None),
        GpuWorkAuthoringCause::InvalidLabel,
        correction,
        GpuWorkAuthoringErrorSource::Descriptor(source),
    )
}
