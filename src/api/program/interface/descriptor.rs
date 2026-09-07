use super::super::contract_diagnostics::{GpuProgramContractCause, GpuProgramContractError};
use super::{GpuBindingDeclaration, GpuBindingKey};
use core::hash::Hash;
use std::sync::Arc;

#[derive(Debug)]
struct GpuProgramInterfaceInner {
    bindings: Vec<GpuBindingDeclaration>,
}

/// Ordered shader-visible resource interface.
#[derive(Debug, Clone)]
pub struct GpuProgramInterfaceDescriptor(Arc<GpuProgramInterfaceInner>);

impl GpuProgramInterfaceDescriptor {
    pub fn new(
        bindings: impl IntoIterator<Item = GpuBindingDeclaration>,
    ) -> Result<Self, GpuProgramContractError> {
        let mut bindings = bindings.into_iter().collect::<Vec<_>>();
        bindings.sort_by_key(GpuBindingDeclaration::key);
        if let Some(duplicate) = bindings
            .windows(2)
            .find(|pair| pair[0].key() == pair[1].key())
            .map(|pair| pair[0].key())
        {
            return Err(GpuProgramContractError::invalid(
                "construct GPU program resource interface",
                duplicate.to_string(),
                GpuProgramContractCause::DuplicateBindingKey,
                "declare each typed group/binding key exactly once",
            ));
        }
        Ok(Self(Arc::new(GpuProgramInterfaceInner { bindings })))
    }

    pub fn bindings(&self) -> impl ExactSizeIterator<Item = &GpuBindingDeclaration> {
        self.0.bindings.iter()
    }

    pub fn binding(&self, key: GpuBindingKey) -> Option<&GpuBindingDeclaration> {
        self.0
            .bindings
            .binary_search_by_key(&key, GpuBindingDeclaration::key)
            .ok()
            .map(|index| &self.0.bindings[index])
    }

    pub fn is_same_record(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl PartialEq for GpuProgramInterfaceDescriptor {
    fn eq(&self, other: &Self) -> bool {
        self.0.bindings == other.0.bindings
    }
}

impl Eq for GpuProgramInterfaceDescriptor {}

impl PartialOrd for GpuProgramInterfaceDescriptor {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for GpuProgramInterfaceDescriptor {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.0.bindings.cmp(&other.0.bindings)
    }
}

impl Hash for GpuProgramInterfaceDescriptor {
    fn hash<State: core::hash::Hasher>(&self, state: &mut State) {
        self.0.bindings.hash(state);
    }
}
