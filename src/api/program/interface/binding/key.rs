use super::super::super::contract_diagnostics::{GpuProgramContractCause, GpuProgramContractError};
use core::fmt;

/// Checked shader binding identity ordered by group then binding.
///
/// ```compile_fail
/// use runen_gpu::GpuBindingKey;
/// let _ = GpuBindingKey { group: 0, binding: 0 };
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuBindingKey {
    group: u32,
    binding: u32,
}

impl GpuBindingKey {
    pub fn try_new(group: u64, binding: u64) -> Result<Self, GpuProgramContractError> {
        let group = u32::try_from(group).map_err(|_| {
            GpuProgramContractError::invalid(
                "construct GPU binding key",
                format!("group={group}"),
                GpuProgramContractCause::InvalidBindingKey,
                "provide a group index representable as u32",
            )
        })?;
        let binding = u32::try_from(binding).map_err(|_| {
            GpuProgramContractError::invalid(
                "construct GPU binding key",
                format!("binding={binding}"),
                GpuProgramContractCause::InvalidBindingKey,
                "provide a binding index representable as u32",
            )
        })?;
        Ok(Self { group, binding })
    }

    pub const fn group(self) -> u32 {
        self.group
    }

    pub const fn binding(self) -> u32 {
        self.binding
    }
}

impl fmt::Display for GpuBindingKey {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "({}, {})", self.group, self.binding)
    }
}
