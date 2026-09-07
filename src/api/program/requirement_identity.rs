use crate::{GpuCapabilityRequirement, GpuCapabilityRequirements};
use core::hash::Hash;

pub(super) fn hash_capability_requirements<State: core::hash::Hasher>(
    requirements: &GpuCapabilityRequirements,
    state: &mut State,
) {
    requirements.iter().len().hash(state);
    for requirement in requirements.iter() {
        match requirement {
            GpuCapabilityRequirement::Required(feature) => {
                0u8.hash(state);
                feature.hash(state);
            }
            GpuCapabilityRequirement::Preferred { feature, fallback } => {
                1u8.hash(state);
                feature.hash(state);
                fallback.hash(state);
            }
            GpuCapabilityRequirement::Disabled(feature) => {
                2u8.hash(state);
                feature.hash(state);
            }
        }
    }
}
