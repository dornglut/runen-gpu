use super::super::contract_diagnostics::{GpuProgramContractCause, GpuProgramContractError};
use super::super::interface::{GpuBindingDeclaration, GpuProgramInterfaceDescriptor};
use std::sync::Arc;

#[derive(Debug)]
struct GpuBindGroupLayoutInner {
    group: u32,
    bindings: Vec<GpuBindingDeclaration>,
}

/// Ordered logical bindings for exactly one shader bind group.
#[derive(Debug, Clone)]
pub struct GpuBindGroupLayoutDescriptor(Arc<GpuBindGroupLayoutInner>);

impl GpuBindGroupLayoutDescriptor {
    pub fn new(
        group: u32,
        bindings: impl IntoIterator<Item = GpuBindingDeclaration>,
    ) -> Result<Self, GpuProgramContractError> {
        let mut bindings = bindings.into_iter().collect::<Vec<_>>();
        bindings.sort_by_key(GpuBindingDeclaration::key);

        if let Some(mismatch) = bindings
            .iter()
            .find(|binding| binding.key().group() != group)
        {
            return Err(GpuProgramContractError::invalid(
                "construct GPU bind-group layout",
                mismatch.key().to_string(),
                GpuProgramContractCause::BindGroupLayoutInvalid,
                "place only declarations naming the layout's exact group",
            ));
        }

        if let Some(duplicate) = bindings
            .windows(2)
            .find(|pair| pair[0].key() == pair[1].key())
            .map(|pair| pair[0].key())
        {
            return Err(GpuProgramContractError::invalid(
                "construct GPU bind-group layout",
                duplicate.to_string(),
                GpuProgramContractCause::DuplicateBindingKey,
                "declare each binding in the group exactly once",
            ));
        }

        Ok(Self(Arc::new(GpuBindGroupLayoutInner { group, bindings })))
    }

    pub fn from_interface(
        interface: &GpuProgramInterfaceDescriptor,
        group: u32,
    ) -> Result<Self, GpuProgramContractError> {
        Self::new(
            group,
            interface
                .bindings()
                .filter(|binding| binding.key().group() == group)
                .cloned(),
        )
    }

    pub fn group(&self) -> u32 {
        self.0.group
    }

    pub fn bindings(&self) -> impl ExactSizeIterator<Item = &GpuBindingDeclaration> {
        self.0.bindings.iter()
    }

    pub fn binding(&self, binding: u32) -> Option<&GpuBindingDeclaration> {
        self.0
            .bindings
            .binary_search_by_key(&binding, |declaration| declaration.key().binding())
            .ok()
            .map(|index| &self.0.bindings[index])
    }

    pub fn is_same_record(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl PartialEq for GpuBindGroupLayoutDescriptor {
    fn eq(&self, other: &Self) -> bool {
        self.0.group == other.0.group && self.0.bindings == other.0.bindings
    }
}

impl Eq for GpuBindGroupLayoutDescriptor {}

impl PartialOrd for GpuBindGroupLayoutDescriptor {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for GpuBindGroupLayoutDescriptor {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        (&self.0.group, &self.0.bindings).cmp(&(&other.0.group, &other.0.bindings))
    }
}

impl core::hash::Hash for GpuBindGroupLayoutDescriptor {
    fn hash<State: core::hash::Hasher>(&self, state: &mut State) {
        core::hash::Hash::hash(&self.0.group, state);
        core::hash::Hash::hash(&self.0.bindings, state);
    }
}
