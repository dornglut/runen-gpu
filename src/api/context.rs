//! Backend-neutral context admission contracts.
//!
//! This module is the public namespace only.  Each admission concern has one
//! implementation owner below it; it is not a compatibility layer.

use super::{
    GpuExecutionPolicy, GpuPreparedWorkGraph, GpuPreparedWorkNode, GpuRealizationPolicies,
    GpuRuntimeBindingDeviceFacts, GpuSubmissionPreparationError, GpuWorkNotAdmittedSource,
    GpuWorkOperation,
};
use core::num::NonZeroU64;

mod admission;
mod descriptor;
mod diagnostics;
mod facts;
mod identity;
mod observation;
mod selection;

pub use admission::{
    GpuAdmissionContract, GpuCandidateAdmissionReport, GpuDegradationRecord,
    GpuRejectedCandidateReport,
};
pub use descriptor::{
    GpuAdapterClass, GpuAlignmentKind, GpuBackendFamily, GpuContextDescriptor, GpuFormatRole,
    GpuLimitConstraint, GpuLimitKind, GpuPortabilityPolicy, GpuPowerPreference,
    GpuSoftwareFallbackPolicy,
};
pub use diagnostics::{GpuContextRequestError, GpuContextRequestErrorCategory};
pub use facts::{
    GpuAdapterFacts, GpuAdapterLimits, GpuAdmittedDeviceFacts, GpuAlignmentFacts, GpuDeviceLimits,
    GpuDeviceRequestProfile, GpuFallbackStatus, GpuPortabilityClass, GpuPortabilityEvidence,
    GpuPortabilityReason, GpuSoftwareStatus, GpuWorkloadBudget,
};
pub use identity::{
    GpuContextAffinity, GpuContextAffinityError, GpuContextId, GpuDeviceGeneration,
};
pub use selection::{
    GpuCandidateDisposition, GpuCandidateId, GpuCandidateRankEvidence,
    GpuCandidateSelectionEvidence, GpuCandidateSelectionKind, GpuContextAdmissionReport,
};

pub(crate) use admission::{
    GpuCandidateEnvironmentEvidence, admitted_device_facts, validate_descriptor,
};
pub(crate) use diagnostics::sanitized_diagnostic;
pub(crate) use identity::allocate_context_id;
#[cfg(test)]
pub(crate) use selection::select_candidate_with_host_evidence;
pub(crate) use selection::{
    GpuCandidateInput, GpuCandidateSelection, canonical_candidate_input_key,
    select_candidate_inputs,
};

#[derive(Debug)]
pub struct GpuContext {
    pub(crate) id: GpuContextId,
    pub(crate) generation: GpuDeviceGeneration,
    pub(crate) adapter: GpuAdapterFacts,
    pub(crate) device: GpuAdmittedDeviceFacts,
    pub(crate) report: GpuContextAdmissionReport,
    pub(crate) backend: crate::backend::WgpuContextState,
}

impl GpuContext {
    /// Requests an asynchronous, headless-first context admission.
    pub async fn request(descriptor: GpuContextDescriptor) -> Result<Self, GpuContextRequestError> {
        Self::request_with_policies(
            descriptor,
            GpuRealizationPolicies::default(),
            GpuExecutionPolicy::default(),
        )
        .await
    }

    /// Requests a context with explicit G4C1 and G4C2 realization-record bounds.
    ///
    /// Policies do not alter adapter selection, device admission, or retry identity.
    pub async fn request_with_realization_policies(
        descriptor: GpuContextDescriptor,
        realization_policies: GpuRealizationPolicies,
    ) -> Result<Self, GpuContextRequestError> {
        Self::request_with_policies(
            descriptor,
            realization_policies,
            GpuExecutionPolicy::default(),
        )
        .await
    }

    /// Requests a context with explicit independent realization and execution pressure policies.
    ///
    /// Execution capacities govern prepared/in-flight submissions and transfer staging only; they
    /// do not become resource-realization, logical resource-size, or physical residency budgets.
    pub async fn request_with_policies(
        descriptor: GpuContextDescriptor,
        realization_policies: GpuRealizationPolicies,
        execution_policy: GpuExecutionPolicy,
    ) -> Result<Self, GpuContextRequestError> {
        crate::backend::request_headless(descriptor, realization_policies, execution_policy).await
    }

    pub const fn id(&self) -> GpuContextId {
        self.id
    }

    pub const fn generation(&self) -> GpuDeviceGeneration {
        self.generation
    }

    pub const fn affinity(&self) -> GpuContextAffinity {
        GpuContextAffinity {
            context: self.id,
            generation: self.generation,
        }
    }

    pub fn adapter_facts(&self) -> &GpuAdapterFacts {
        &self.adapter
    }

    pub fn device_facts(&self) -> &GpuAdmittedDeviceFacts {
        &self.device
    }

    pub(crate) fn runtime_binding_device_facts(&self) -> GpuRuntimeBindingDeviceFacts {
        let device_limits = self.device.device_limits();
        let alignments = device_limits.alignments();
        let limits = device_limits.values();
        GpuRuntimeBindingDeviceFacts::new(
            alignments.uniform_dynamic_offset.and_then(NonZeroU64::new),
            alignments.storage_dynamic_offset.and_then(NonZeroU64::new),
            limits.max_bind_groups(),
            limits.max_dynamic_uniform_buffers_per_pipeline_layout(),
            limits.max_dynamic_storage_buffers_per_pipeline_layout(),
            self.adapter.supported().formats(),
        )
    }

    pub(crate) fn validate_prepared_work_device_facts(
        &self,
        graph: &GpuPreparedWorkGraph,
    ) -> Result<(), GpuSubmissionPreparationError> {
        let binding_facts = self.runtime_binding_device_facts();
        let limits = self.device_facts().workload_budget().limits();

        for prepared in graph.nodes() {
            match prepared.node().operation() {
                GpuWorkOperation::Compute(operation) => {
                    operation
                        .bindings()
                        .validate_device_facts(&binding_facts)
                        .map_err(|error| {
                            work_not_admitted(
                                prepared,
                                GpuWorkNotAdmittedSource::ProgramContract(error),
                            )
                        })?;
                    operation
                        .dispatch()
                        .validate_limits(limits)
                        .map_err(|error| {
                            work_not_admitted(prepared, GpuWorkNotAdmittedSource::Operation(error))
                        })?;
                }
                GpuWorkOperation::Render(operation) => {
                    for draw in operation.draws() {
                        draw.bindings()
                            .validate_device_facts(&binding_facts)
                            .map_err(|error| {
                                work_not_admitted(
                                    prepared,
                                    GpuWorkNotAdmittedSource::ProgramContract(error),
                                )
                            })?;
                        draw.validate_limits(limits).map_err(|error| {
                            work_not_admitted(prepared, GpuWorkNotAdmittedSource::Operation(error))
                        })?;
                    }
                }
                GpuWorkOperation::Copy(_)
                | GpuWorkOperation::Clear(_)
                | GpuWorkOperation::Resolve(_)
                | GpuWorkOperation::Present(_)
                | GpuWorkOperation::Upload(_)
                | GpuWorkOperation::Readback(_) => {}
            }
        }
        Ok(())
    }

    pub fn admission_report(&self) -> &GpuContextAdmissionReport {
        &self.report
    }

    pub fn validate_affinity(
        &self,
        affinity: GpuContextAffinity,
    ) -> Result<(), GpuContextAffinityError> {
        identity::validate_affinity(self.affinity(), affinity)
    }
}

fn work_not_admitted(
    prepared: &GpuPreparedWorkNode,
    source: GpuWorkNotAdmittedSource,
) -> GpuSubmissionPreparationError {
    let source_detail = match &source {
        GpuWorkNotAdmittedSource::ProgramContract(error) => error.to_string(),
        GpuWorkNotAdmittedSource::Operation(error) => error.to_string(),
    };
    GpuSubmissionPreparationError::work_not_admitted(
        format!(
            "fragment={:?} node={:?} prepared_node={:?}: {source_detail}",
            prepared.fragment_label().as_str(),
            prepared.node().label().as_str(),
            prepared.id(),
        ),
        source,
    )
}
