//! Immutable GPU work authoring and deterministic graph preparation.
//!
//! The public module remains one authority while private semantic modules keep
//! identity, coverage, authoring, dependency, initialization, hazard, and
//! preparation responsibilities independently maintainable.

mod authoring;
mod composition;
mod coverage;
mod dependency;
mod diagnostics;
mod hazards;
mod identity;
mod initial_content;
mod initialization;
mod preparation;

pub use authoring::{
    GpuExecutionPreference, GpuExplicitOrder, GpuWorkFragment, GpuWorkFragmentBuilder,
    GpuWorkImport, GpuWorkNode, GpuWorkOutput,
};
pub use coverage::{
    GpuBufferCoverage, GpuBufferStridedCoverage, GpuInitialCoverage, GpuInitialCoverageKind,
    GpuWorkResourceInput,
};
pub use dependency::{GpuDependencyReason, GpuDependencyRegion, GpuWorkDependency};
pub use diagnostics::GpuPreparedWorkDiagnostic;
pub use identity::{GpuPreparedWorkNodeId, GpuWorkNodeId};
pub(crate) use initial_content::GpuPreparedInitialContent;
pub use initialization::{
    GpuInitializationExplanation, GpuInitializationExplanationKind,
    GpuPreparedResourceInitialization,
};
pub(crate) use initialization::{initial_coverage_contains, initial_coverage_intersection};
pub use preparation::{GpuPreparedWorkGraph, GpuPreparedWorkNode};

pub(crate) fn same_resource_descriptor(
    left: &super::GpuResourceRef,
    right: &super::GpuResourceRef,
) -> bool {
    match (left, right) {
        (super::GpuResourceRef::Buffer(left), super::GpuResourceRef::Buffer(right)) => {
            left.descriptor() == right.descriptor()
        }
        (super::GpuResourceRef::Texture(left), super::GpuResourceRef::Texture(right)) => {
            left.descriptor() == right.descriptor()
        }
        (super::GpuResourceRef::TextureView(left), super::GpuResourceRef::TextureView(right)) => {
            left.descriptor() == right.descriptor()
                && left.descriptor().texture().descriptor()
                    == right.descriptor().texture().descriptor()
        }
        (super::GpuResourceRef::Sampler(left), super::GpuResourceRef::Sampler(right)) => {
            left.descriptor() == right.descriptor()
        }
        (super::GpuResourceRef::QuerySet(left), super::GpuResourceRef::QuerySet(right)) => {
            left.descriptor() == right.descriptor()
        }
        _ => false,
    }
}

#[cfg(test)]
mod tests;
