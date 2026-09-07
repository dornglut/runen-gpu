use super::super::contract_diagnostics::{GpuProgramContractCause, GpuProgramContractError};
use core::fmt::Debug;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum GpuVertexInputBuiltin {
    VertexIndex,
    InstanceIndex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum GpuFragmentOutputBuiltin {
    FragDepth,
    SampleMask,
}

pub(super) fn normalize_vertex_input_builtins(
    builtins: impl IntoIterator<Item = GpuVertexInputBuiltin>,
) -> Result<Vec<GpuVertexInputBuiltin>, GpuProgramContractError> {
    normalize_unique_builtins("observed vertex input", builtins)
}

pub(super) fn normalize_fragment_output_builtins(
    builtins: impl IntoIterator<Item = GpuFragmentOutputBuiltin>,
) -> Result<Vec<GpuFragmentOutputBuiltin>, GpuProgramContractError> {
    normalize_unique_builtins("observed fragment output", builtins)
}

fn normalize_unique_builtins<T>(
    role: &'static str,
    builtins: impl IntoIterator<Item = T>,
) -> Result<Vec<T>, GpuProgramContractError>
where
    T: Copy + Debug + Ord,
{
    let mut builtins = builtins.into_iter().collect::<Vec<_>>();
    builtins.sort_unstable();
    if let Some(duplicate) = builtins
        .windows(2)
        .find(|pair| pair[0] == pair[1])
        .map(|pair| pair[0])
    {
        return Err(GpuProgramContractError::invalid(
            "construct GPU shader-stage IO signature",
            format!("{role} duplicate builtin={duplicate:?}"),
            GpuProgramContractCause::StageIoSignatureInvalid,
            "provide each observed shader builtin exactly once",
        ));
    }
    Ok(builtins)
}
