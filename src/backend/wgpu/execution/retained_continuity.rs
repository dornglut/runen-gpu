use crate::api::{
    initial_coverage_contains, initial_coverage_intersection, same_resource_descriptor,
};
use crate::{
    GpuContext, GpuContextAffinity, GpuContextDescriptor, GpuDeviceGenerationReplacementError,
    GpuInitialCoverage, GpuOpaqueContentContinuity, GpuOpaqueContentIndeterminateReason,
    GpuPreparedWorkGraph, GpuRealizationPolicies, GpuReconstruction, GpuResourceLabel,
    GpuResourceRef, GpuRetainedInitializationSeed, GpuRetainedReconstructionRequirement,
    GpuRetainedReconstructionSeed, GpuRetainedResourceContinuity, GpuSubmissionId,
    GpuSubmissionRejectionKind, GpuSubmissionRejectionReason, GpuWorkFragment, GpuWorkGraphError,
    GpuWorkResourceId,
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

#[cfg(test)]
mod explainability_tests;
#[cfg(test)]
mod participation_tests;
#[cfg(test)]
mod quiescence_tests;
#[cfg(test)]
mod storage_scope_tests;
#[cfg(test)]
mod terminal_boundary_tests;
#[cfg(test)]
mod tests;
#[cfg(test)]
mod zeroed_reconstruction_tests;

#[derive(Debug, Clone)]
pub(super) struct PreparedRetainedContinuity {
    resources: BTreeMap<GpuWorkResourceId, PreparedRetainedResource>,
}

#[derive(Debug, Clone, Copy, Default)]
struct PreparedReconstructionEvidence {
    target: bool,
    explicit_write: bool,
    fresh_descriptor_initial_state: bool,
}

#[derive(Debug, Clone)]
struct PreparedRetainedResource {
    resource: GpuResourceRef,
    consumed_lifecycle: bool,
    consumed_seed: Option<GpuInitialCoverage>,
    initial: Option<GpuInitialCoverage>,
    final_coverage: Option<GpuInitialCoverage>,
    failure_preserved_coverage: Option<GpuInitialCoverage>,
    reconstruction: PreparedReconstructionEvidence,
}

fn is_retained_storage_resource(resource: &GpuResourceRef) -> bool {
    resource.common().lifetime().is_retained()
        && matches!(
            resource,
            GpuResourceRef::Buffer(_) | GpuResourceRef::Texture(_) | GpuResourceRef::QuerySet(_)
        )
}

impl PreparedRetainedContinuity {
    pub(super) fn from_graph(graph: &GpuPreparedWorkGraph) -> Self {
        let descriptor_materializations = graph
            .initial_content()
            .iter()
            .map(|candidate| candidate.resource_identity())
            .collect::<BTreeSet<_>>();
        let participating = graph
            .nodes()
            .iter()
            .flat_map(|prepared| prepared.node().accesses().iter())
            .map(|access| access.resource_identity())
            .chain(descriptor_materializations.iter().copied())
            .collect::<BTreeSet<_>>();
        let explicit_writes = graph
            .nodes()
            .iter()
            .flat_map(|prepared| prepared.node().accesses().iter())
            .filter(|access| access.writes())
            .map(|access| access.resource_identity())
            .collect::<BTreeSet<_>>();
        let resources = graph
            .initialization()
            .iter()
            .filter(|summary| {
                is_retained_storage_resource(summary.resource())
                    && participating.contains(&summary.resource().diagnostic_identity())
            })
            .map(|summary| {
                let resource = summary.resource().clone();
                let identity = resource.diagnostic_identity();
                let retained_seed = graph
                    .retained_seed()
                    .iter()
                    .find(|seed| seed.resource_identity() == identity);
                let consumed_lifecycle = retained_seed.is_some();
                let consumed_seed = retained_seed
                    .and_then(|seed| seed.initialized_coverage())
                    .cloned();
                let reconstruction_target = graph.reconstruction_targets().contains(&identity);
                let fresh_descriptor_initial_state = if consumed_lifecycle {
                    false
                } else {
                    match &resource {
                        GpuResourceRef::Buffer(buffer) => {
                            match buffer.descriptor().initialization() {
                                crate::GpuBufferInitialization::Uninitialized => false,
                                crate::GpuBufferInitialization::Zeroed => !reconstruction_target,
                                crate::GpuBufferInitialization::Prepared(_) => {
                                    descriptor_materializations.contains(&identity)
                                }
                            }
                        }
                        GpuResourceRef::Texture(texture) => {
                            match texture.descriptor().initialization() {
                                crate::GpuTextureInitialization::Uninitialized => false,
                                crate::GpuTextureInitialization::Zeroed => !reconstruction_target,
                                crate::GpuTextureInitialization::Prepared(_) => {
                                    descriptor_materializations.contains(&identity)
                                }
                            }
                        }
                        GpuResourceRef::TextureView(_)
                        | GpuResourceRef::Sampler(_)
                        | GpuResourceRef::QuerySet(_) => false,
                    }
                };
                (
                    identity,
                    PreparedRetainedResource {
                        resource,
                        consumed_lifecycle,
                        consumed_seed,
                        initial: summary.initial().cloned(),
                        final_coverage: summary.final_coverage().cloned(),
                        failure_preserved_coverage: graph
                            .failure_preserved_coverage(identity)
                            .cloned(),
                        reconstruction: PreparedReconstructionEvidence {
                            target: reconstruction_target,
                            explicit_write: explicit_writes.contains(&identity),
                            fresh_descriptor_initial_state,
                        },
                    },
                )
            })
            .collect();
        Self { resources }
    }

    pub(super) fn contains(&self, resource: GpuWorkResourceId) -> bool {
        self.resources.contains_key(&resource)
    }

    pub(super) fn is_empty(&self) -> bool {
        self.resources.is_empty()
    }
}

#[derive(Debug, Clone)]
struct RetainedContinuityRecord {
    resource: GpuResourceRef,
    initialized_coverage: Option<GpuInitialCoverage>,
    opaque_content: GpuOpaqueContentContinuity,
    write_pending: bool,
    descriptor_initial_state_matches_required_contents: bool,
}

#[derive(Debug)]
pub(super) struct RetainedContinuityState {
    affinity: GpuContextAffinity,
    records: Mutex<BTreeMap<GpuWorkResourceId, RetainedContinuityRecord>>,
    reconstruction_required: Mutex<BTreeMap<GpuWorkResourceId, GpuRetainedReconstructionSeed>>,
    reserved: Mutex<BTreeSet<GpuWorkResourceId>>,
}

impl RetainedContinuityState {
    pub(super) fn new(affinity: GpuContextAffinity) -> Self {
        Self {
            affinity,
            records: Mutex::new(BTreeMap::new()),
            reconstruction_required: Mutex::new(BTreeMap::new()),
            reserved: Mutex::new(BTreeSet::new()),
        }
    }

    fn install_reconstruction_obligations(
        &self,
        seeds: impl IntoIterator<Item = GpuRetainedReconstructionSeed>,
    ) {
        let mut obligations = self
            .reconstruction_required
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        obligations.clear();
        obligations.extend(seeds.into_iter().filter_map(|seed| {
            is_retained_storage_resource(seed.resource())
                .then(|| (seed.resource().diagnostic_identity(), seed))
        }));
    }

    pub(super) fn reconstruction_seed(&self) -> Vec<GpuRetainedReconstructionSeed> {
        let records = self
            .records
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let obligations = self
            .reconstruction_required
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut resources = obligations.clone();
        for (&identity, record) in records.iter() {
            resources.entry(identity).or_insert_with(|| {
                GpuRetainedReconstructionSeed::new(
                    record.resource.clone(),
                    record.descriptor_initial_state_matches_required_contents,
                )
            });
        }
        resources.into_values().collect()
    }

    pub(super) fn coverage_seed(&self) -> Vec<GpuRetainedInitializationSeed> {
        self.records
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .values()
            .map(|record| {
                GpuRetainedInitializationSeed::new(
                    record.resource.clone(),
                    record.initialized_coverage.clone(),
                )
            })
            .collect()
    }

    pub(super) fn snapshot(
        &self,
        resource: GpuWorkResourceId,
    ) -> Option<GpuRetainedResourceContinuity> {
        self.records
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&resource)
            .map(|record| {
                let (opaque_content, indeterminate_reason) = if record.write_pending {
                    (
                        GpuOpaqueContentContinuity::Unknown,
                        Some(GpuOpaqueContentIndeterminateReason::AcceptedWritePending),
                    )
                } else if record.opaque_content == GpuOpaqueContentContinuity::Unknown {
                    (
                        GpuOpaqueContentContinuity::Unknown,
                        Some(GpuOpaqueContentIndeterminateReason::PossibleWriteFailure),
                    )
                } else {
                    (record.opaque_content, None)
                };
                GpuRetainedResourceContinuity::new(
                    self.affinity,
                    record.resource.clone(),
                    record.initialized_coverage.clone(),
                    opaque_content,
                    indeterminate_reason,
                )
            })
    }

    pub(super) fn reconstruction_requirement(
        &self,
        resource: GpuWorkResourceId,
    ) -> Option<GpuRetainedReconstructionRequirement> {
        self.reconstruction_required
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(&resource)
            .map(|seed| {
                GpuRetainedReconstructionRequirement::new(
                    self.affinity,
                    seed.resource().clone(),
                    seed.descriptor_initial_state_matches_required_contents(),
                )
            })
    }

    pub(super) fn validate_and_reserve(
        &self,
        transition: &PreparedRetainedContinuity,
    ) -> Result<(), GpuSubmissionRejectionReason> {
        if transition.is_empty() {
            return Ok(());
        }

        let records = self
            .records
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let reconstruction_required = self
            .reconstruction_required
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut reserved = self
            .reserved
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        for (&identity, prepared) in &transition.resources {
            if reserved.contains(&identity) {
                return Err(continuity_changed(
                    identity,
                    "another accepted submission still owns retained-state transition authority for this resource",
                ));
            }
            let current = records.get(&identity);
            if let Some(current) = current {
                if !same_resource_descriptor(&current.resource, &prepared.resource) {
                    return Err(continuity_changed(
                        identity,
                        "the retained resource descriptor changed for the same logical identity",
                    ));
                }
            }

            let reconstruction = reconstruction_required.get(&identity);
            if let Some(required) = reconstruction {
                if !same_resource_descriptor(required.resource(), &prepared.resource) {
                    return Err(continuity_changed(
                        identity,
                        "the reconstruction requirement names a different descriptor for this logical identity",
                    ));
                }
            }
            if reconstruction.is_some() && !prepared.reconstruction.target {
                return Err(reconstruction_required_reason(identity));
            }
            if prepared.reconstruction.target {
                let Some(required) = reconstruction else {
                    return Err(continuity_changed(
                        identity,
                        "work was designated as reconstruction but no current reconstruction requirement exists",
                    ));
                };
                match prepared.resource.common().reconstruction() {
                    GpuReconstruction::NonReconstructable => {
                        return Err(reconstruction_unavailable(identity));
                    }
                    GpuReconstruction::ExternallyReconstructed => {
                        if !prepared.reconstruction.explicit_write {
                            return Err(continuity_changed(
                                identity,
                                "external reconstruction must provide a canonical explicit write/reimport effect",
                            ));
                        }
                    }
                    GpuReconstruction::SourceBacked => {
                        let descriptor_source_is_valid = required
                            .descriptor_initial_state_matches_required_contents()
                            && prepared.reconstruction.fresh_descriptor_initial_state;
                        if !prepared.reconstruction.explicit_write && !descriptor_source_is_valid {
                            return Err(continuity_changed(
                                identity,
                                "descriptor initial state cannot reconstruct the required current contents; provide explicit deterministic replay/materialization work",
                            ));
                        }
                    }
                }
            }

            if current.is_some() != prepared.consumed_lifecycle {
                let detail = if current.is_some() {
                    "prepared work did not consume the current retained lifecycle state"
                } else {
                    "retained lifecycle state consumed during graph preparation is no longer current"
                };
                return Err(continuity_changed(identity, detail));
            }
            if let Some(seed) = &prepared.consumed_seed {
                let Some(current_coverage) =
                    current.and_then(|record| record.initialized_coverage.as_ref())
                else {
                    return Err(continuity_changed(
                        identity,
                        "retained coverage used during graph preparation is no longer established",
                    ));
                };
                if !initial_coverage_contains(current_coverage, seed) {
                    return Err(continuity_changed(
                        identity,
                        "retained coverage used during graph preparation is no longer contained by current continuity",
                    ));
                }
            }
            if let Some(current_coverage) =
                current.and_then(|record| record.initialized_coverage.as_ref())
            {
                if !prepared
                    .initial
                    .as_ref()
                    .is_some_and(|initial| initial_coverage_contains(initial, current_coverage))
                {
                    return Err(continuity_changed(
                        identity,
                        "current retained coverage contains state not represented by the prepared graph entry state",
                    ));
                }
            }
        }

        reserved.extend(transition.resources.keys().copied());
        Ok(())
    }

    pub(super) fn mark_may_execute(
        &self,
        transition: &PreparedRetainedContinuity,
        submitted_writes: &BTreeSet<GpuWorkResourceId>,
    ) {
        if submitted_writes.is_empty() {
            return;
        }

        let mut records = self
            .records
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for &identity in submitted_writes {
            let Some(prepared) = transition.resources.get(&identity) else {
                continue;
            };
            let previous = records.get(&identity);
            let preserved = failure_safe_coverage(prepared);
            let previous_opaque = previous
                .map_or(GpuOpaqueContentContinuity::Unestablished, |record| {
                    record.opaque_content
                });
            let descriptor_initial_state_matches_required_contents =
                if prepared.reconstruction.explicit_write {
                    false
                } else {
                    previous.map_or(
                        prepared.reconstruction.fresh_descriptor_initial_state,
                        |record| record.descriptor_initial_state_matches_required_contents,
                    )
                };
            records.insert(
                identity,
                RetainedContinuityRecord {
                    resource: prepared.resource.clone(),
                    initialized_coverage: preserved,
                    opaque_content: previous_opaque,
                    write_pending: true,
                    descriptor_initial_state_matches_required_contents,
                },
            );
        }
    }

    pub(super) fn complete(
        &self,
        submission: GpuSubmissionId,
        transition: &PreparedRetainedContinuity,
        submitted_writes: &BTreeSet<GpuWorkResourceId>,
    ) {
        let mut records = self
            .records
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut obligations = self
            .reconstruction_required
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);

        for (&identity, prepared) in &transition.resources {
            let previous = records.get(&identity);
            let previous_opaque = previous
                .map_or(GpuOpaqueContentContinuity::Unestablished, |record| {
                    record.opaque_content
                });
            let required = obligations.get(&identity);
            let opaque_content = if prepared.reconstruction.target {
                GpuOpaqueContentContinuity::Established {
                    last_completed_write: submission,
                }
            } else if submitted_writes.contains(&identity) {
                match previous_opaque {
                    GpuOpaqueContentContinuity::Unknown => GpuOpaqueContentContinuity::Unknown,
                    GpuOpaqueContentContinuity::Unestablished
                    | GpuOpaqueContentContinuity::Established { .. } => {
                        GpuOpaqueContentContinuity::Established {
                            last_completed_write: submission,
                        }
                    }
                }
            } else {
                previous_opaque
            };
            let descriptor_initial_state_matches_required_contents =
                if prepared.reconstruction.explicit_write {
                    false
                } else if prepared.reconstruction.target {
                    required.is_some_and(|required| {
                        required.descriptor_initial_state_matches_required_contents()
                            && prepared.reconstruction.fresh_descriptor_initial_state
                    })
                } else {
                    previous.map_or(
                        prepared.reconstruction.fresh_descriptor_initial_state,
                        |record| record.descriptor_initial_state_matches_required_contents,
                    )
                };

            records.insert(
                identity,
                RetainedContinuityRecord {
                    resource: prepared.resource.clone(),
                    initialized_coverage: prepared.final_coverage.clone(),
                    opaque_content,
                    write_pending: false,
                    descriptor_initial_state_matches_required_contents,
                },
            );
            if prepared.reconstruction.target {
                obligations.remove(&identity);
            }
        }
        drop(obligations);
        drop(records);
        self.release(transition);
    }

    pub(super) fn fail_after_acceptance(
        &self,
        transition: &PreparedRetainedContinuity,
        submitted_writes: &BTreeSet<GpuWorkResourceId>,
    ) {
        if !submitted_writes.is_empty() {
            let mut records = self
                .records
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            let mut obligations = self
                .reconstruction_required
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            for &identity in submitted_writes {
                let Some(prepared) = transition.resources.get(&identity) else {
                    continue;
                };
                let preserved = failure_safe_coverage(prepared);
                let descriptor_initial_state_matches_required_contents =
                    if prepared.reconstruction.target {
                        obligations.get(&identity).is_some_and(|seed| {
                            seed.descriptor_initial_state_matches_required_contents()
                        })
                    } else {
                        false
                    };
                records.insert(
                    identity,
                    RetainedContinuityRecord {
                        resource: prepared.resource.clone(),
                        initialized_coverage: preserved,
                        opaque_content: GpuOpaqueContentContinuity::Unknown,
                        write_pending: false,
                        descriptor_initial_state_matches_required_contents,
                    },
                );
                obligations.insert(
                    identity,
                    GpuRetainedReconstructionSeed::new(
                        prepared.resource.clone(),
                        descriptor_initial_state_matches_required_contents,
                    ),
                );
            }
        }
        self.release(transition);
    }

    fn release(&self, transition: &PreparedRetainedContinuity) {
        let mut reserved = self
            .reserved
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        for identity in transition.resources.keys() {
            reserved.remove(identity);
        }
    }
}

fn generation_replacement_is_quiescent(
    in_flight_submissions: usize,
    pending_readbacks: usize,
) -> bool {
    in_flight_submissions == 0 && pending_readbacks == 0
}

impl GpuContext {
    /// Requests and atomically installs the next physical device generation for this logical
    /// process-local context identity.
    ///
    /// Submitted GPU work and readback observation must be quiescent before the retained-state
    /// handoff is captured. Merely prepared, unsubmitted work may remain and becomes stale after a
    /// successful replacement. A failed or cancelled successor request leaves this generation
    /// intact. Realization/execution policies are preserved; physical realizations and surfaces are
    /// not migrated. The current private WGPU instance is reused so backend-environment admission
    /// does not silently change. Old initialized coverage and opaque continuity are never reused as
    /// new-generation current state.
    pub async fn replace_device_generation(
        &mut self,
        descriptor: GpuContextDescriptor,
    ) -> Result<(), GpuDeviceGenerationReplacementError> {
        let generation = self
            .generation
            .next()
            .ok_or(GpuDeviceGenerationReplacementError::GenerationExhausted)?;
        let stats = self.progress();
        if !generation_replacement_is_quiescent(
            stats.in_flight_submissions(),
            stats.pending_readbacks(),
        ) {
            return Err(GpuDeviceGenerationReplacementError::ExecutionNotQuiescent {
                in_flight_submissions: stats.in_flight_submissions(),
                pending_readbacks: stats.pending_readbacks(),
            });
        }

        let id = self.id;
        let instance = self.backend.instance.clone();
        let realization_policies = GpuRealizationPolicies::new(
            self.resource_realization_policy(),
            self.program_binding_realization_policy(),
        );
        let execution_policy = self.execution_policy();
        let reconstruction = self.backend.execution.retained.reconstruction_seed();
        let replacement = super::super::device_request::request_generation_with_instance(
            instance,
            descriptor,
            realization_policies,
            execution_policy,
            id,
            generation,
        )
        .await?;
        replacement
            .backend
            .execution
            .retained
            .install_reconstruction_obligations(reconstruction);
        *self = replacement;
        Ok(())
    }

    /// Returns a current-generation reconstruction requirement separately from retained continuity.
    pub fn retained_resource_reconstruction_requirement(
        &self,
        resource: GpuWorkResourceId,
    ) -> Option<GpuRetainedReconstructionRequirement> {
        self.backend
            .execution
            .retained
            .reconstruction_requirement(resource)
    }

    /// Prepares explicit current-state reconstruction work against this generation's canonical
    /// initialization authority.
    ///
    /// Targets remain ordinary retained resources in the one work graph. The caller owns the
    /// deterministic replay/source or external reimport data. RunenGPU only validates that an
    /// outstanding requirement exists and that accepted canonical work can establish the target.
    pub fn prepare_reconstruction_work_graph(
        &self,
        label: GpuResourceLabel,
        fragments: impl IntoIterator<Item = GpuWorkFragment>,
        targets: impl IntoIterator<Item = GpuResourceRef>,
    ) -> Result<GpuPreparedWorkGraph, GpuWorkGraphError> {
        let retained = self.backend.execution.retained.coverage_seed();
        let targets = targets.into_iter().collect::<Vec<_>>();
        GpuPreparedWorkGraph::prepare_with_retained_coverage_and_reconstruction(
            label, fragments, &retained, &targets,
        )
    }
}

fn failure_safe_coverage(prepared: &PreparedRetainedResource) -> Option<GpuInitialCoverage> {
    let initial = prepared.initial.as_ref()?;
    let failure_preserved = prepared.failure_preserved_coverage.as_ref()?;
    initial_coverage_intersection(&prepared.resource, initial, failure_preserved)
}

fn continuity_changed(
    resource: GpuWorkResourceId,
    detail: &'static str,
) -> GpuSubmissionRejectionReason {
    GpuSubmissionRejectionReason::new(
        GpuSubmissionRejectionKind::RetainedContinuityChanged,
        format!(
            "retained resource {resource}: {detail}; prepare the work again against current retained lifecycle state"
        ),
    )
}

fn reconstruction_required_reason(resource: GpuWorkResourceId) -> GpuSubmissionRejectionReason {
    continuity_changed(
        resource,
        "current required contents need explicit reconstruction before ordinary use",
    )
}

fn reconstruction_unavailable(resource: GpuWorkResourceId) -> GpuSubmissionRejectionReason {
    continuity_changed(
        resource,
        "the NonReconstructable contract cannot certify recovery of lost current contents; use a new logical state identity for an explicit reset",
    )
}
