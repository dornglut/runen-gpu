use super::super::contract_diagnostics::{GpuProgramContractCause, GpuProgramContractError};
use super::super::interface::GpuProgramInterfaceDescriptor;
use super::GpuBindGroupLayoutDescriptor;
use std::collections::BTreeMap;
use std::sync::Arc;

#[derive(Debug)]
struct GpuPipelineLayoutInner {
    groups: Vec<GpuBindGroupLayoutDescriptor>,
}

/// Ordered unique logical bind-group layouts without push constants.
#[derive(Debug, Clone)]
pub struct GpuPipelineLayoutDescriptor(Arc<GpuPipelineLayoutInner>);

impl GpuPipelineLayoutDescriptor {
    pub fn new(
        groups: impl IntoIterator<Item = GpuBindGroupLayoutDescriptor>,
    ) -> Result<Self, GpuProgramContractError> {
        let mut groups = groups.into_iter().collect::<Vec<_>>();
        groups.sort_by_key(GpuBindGroupLayoutDescriptor::group);

        if let Some(duplicate) = groups
            .windows(2)
            .find(|pair| pair[0].group() == pair[1].group())
            .map(|pair| pair[0].group())
        {
            return Err(GpuProgramContractError::invalid(
                "construct GPU pipeline layout",
                format!("group={duplicate}"),
                GpuProgramContractCause::DuplicateBindGroupLayout,
                "declare each bind-group layout exactly once",
            ));
        }

        Ok(Self(Arc::new(GpuPipelineLayoutInner { groups })))
    }

    pub fn from_interface(
        interface: &GpuProgramInterfaceDescriptor,
    ) -> Result<Self, GpuProgramContractError> {
        let mut declarations_by_group = BTreeMap::<u32, Vec<_>>::new();
        for binding in interface.bindings() {
            declarations_by_group
                .entry(binding.key().group())
                .or_default()
                .push(binding.clone());
        }

        let groups = declarations_by_group
            .into_iter()
            .map(|(group, bindings)| GpuBindGroupLayoutDescriptor::new(group, bindings))
            .collect::<Result<Vec<_>, _>>()?;
        Self::new(groups)
    }

    pub fn groups(&self) -> impl ExactSizeIterator<Item = &GpuBindGroupLayoutDescriptor> {
        self.0.groups.iter()
    }

    pub fn group(&self, group: u32) -> Option<&GpuBindGroupLayoutDescriptor> {
        self.0
            .groups
            .binary_search_by_key(&group, GpuBindGroupLayoutDescriptor::group)
            .ok()
            .map(|index| &self.0.groups[index])
    }

    pub fn is_same_record(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl PartialEq for GpuPipelineLayoutDescriptor {
    fn eq(&self, other: &Self) -> bool {
        self.0.groups == other.0.groups
    }
}

impl Eq for GpuPipelineLayoutDescriptor {}

impl PartialOrd for GpuPipelineLayoutDescriptor {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for GpuPipelineLayoutDescriptor {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.groups.cmp(&other.0.groups)
    }
}

impl core::hash::Hash for GpuPipelineLayoutDescriptor {
    fn hash<State: core::hash::Hasher>(&self, state: &mut State) {
        core::hash::Hash::hash(&self.0.groups, state);
    }
}
