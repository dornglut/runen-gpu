use super::{GpuContextAffinity, GpuContextRequestError, GpuResourceRef};
use core::fmt;

/// Outstanding current-state reconstruction requirement for one retained logical storage resource.
///
/// This is lifecycle evidence, not another reconstruction-policy taxonomy. The durable owner
/// contract remains [`super::GpuReconstruction`] on the resource descriptor. Presence means the
/// resource's current-generation contents have not yet been re-established after loss/revocation.
/// It deliberately grants neither initialized coverage nor opaque content continuity.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuRetainedReconstructionRequirement {
    affinity: GpuContextAffinity,
    resource: GpuResourceRef,
    descriptor_initial_state_matches_required_contents: bool,
}

impl GpuRetainedReconstructionRequirement {
    pub(crate) fn new(
        affinity: GpuContextAffinity,
        resource: GpuResourceRef,
        descriptor_initial_state_matches_required_contents: bool,
    ) -> Self {
        Self {
            affinity,
            resource,
            descriptor_initial_state_matches_required_contents,
        }
    }

    /// Context/device generation in which this requirement must be satisfied.
    pub const fn affinity(&self) -> GpuContextAffinity {
        self.affinity
    }

    /// Retained logical storage resource whose current required contents are unavailable.
    pub fn resource(&self) -> &GpuResourceRef {
        &self.resource
    }

    /// Whether accepted prior lifecycle history proves that the required contents at loss still
    /// matched the descriptor's initial-state request.
    ///
    /// `true` only says that descriptor-backed source/materialization may be a valid reconstruction
    /// source. It does not establish bytes, initialized coverage, or continuity in this generation;
    /// reconstruction must still complete through canonical work/materialization authority.
    pub const fn descriptor_initial_state_matches_required_contents(&self) -> bool {
        self.descriptor_initial_state_matches_required_contents
    }
}

/// Private cross-generation handoff for one retained reconstruction requirement.
///
/// This carries only logical resource identity/descriptor authority plus the narrow lifecycle fact
/// needed to decide whether descriptor initial-content materialization can still represent the
/// required state. It never carries old-generation initialized coverage or opaque continuity.
#[derive(Debug, Clone)]
pub(crate) struct GpuRetainedReconstructionSeed {
    resource: GpuResourceRef,
    descriptor_initial_state_matches_required_contents: bool,
}

impl GpuRetainedReconstructionSeed {
    pub(crate) fn new(
        resource: GpuResourceRef,
        descriptor_initial_state_matches_required_contents: bool,
    ) -> Self {
        Self {
            resource,
            descriptor_initial_state_matches_required_contents,
        }
    }

    pub(crate) fn resource(&self) -> &GpuResourceRef {
        &self.resource
    }

    pub(crate) const fn descriptor_initial_state_matches_required_contents(&self) -> bool {
        self.descriptor_initial_state_matches_required_contents
    }
}

/// Failure to replace one physical device generation for an existing logical context.
///
/// Replacement is explicit and transactional: a failed request leaves the current context intact.
#[derive(Debug)]
pub enum GpuDeviceGenerationReplacementError {
    /// Submitted GPU execution or readback observation is still nonterminal, so the retained-state
    /// handoff is not stable yet. Merely prepared, unsubmitted work does not block replacement.
    ExecutionNotQuiescent {
        in_flight_submissions: usize,
        pending_readbacks: usize,
    },
    /// The process-local generation counter cannot advance without wrapping.
    GenerationExhausted,
    /// Fresh adapter/device admission failed; the current generation remains owned by the caller.
    ContextRequest(GpuContextRequestError),
}

impl fmt::Display for GpuDeviceGenerationReplacementError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExecutionNotQuiescent {
                in_flight_submissions,
                pending_readbacks,
            } => write!(
                formatter,
                "GPU device-generation replacement requires quiescent submitted work and readback observation; in-flight submissions: {in_flight_submissions}, pending readbacks: {pending_readbacks}"
            ),
            Self::GenerationExhausted => formatter.write_str(
                "GPU device-generation replacement exhausted the process-local generation identity space",
            ),
            Self::ContextRequest(error) => write!(
                formatter,
                "GPU device-generation replacement could not admit the successor device: {error}"
            ),
        }
    }
}

impl std::error::Error for GpuDeviceGenerationReplacementError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::ContextRequest(error) => Some(error),
            Self::ExecutionNotQuiescent { .. } | Self::GenerationExhausted => None,
        }
    }
}

impl From<GpuContextRequestError> for GpuDeviceGenerationReplacementError {
    fn from(value: GpuContextRequestError) -> Self {
        Self::ContextRequest(value)
    }
}
