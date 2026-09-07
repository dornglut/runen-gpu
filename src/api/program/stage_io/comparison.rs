use super::super::contract_diagnostics::{GpuProgramContractCause, GpuProgramContractError};
use super::super::entry_point::GpuEntryPointName;
use super::{
    GpuExpectedFragmentOutputSignature, GpuExpectedVertexInputSignature,
    GpuObservedFragmentOutputSignature, GpuObservedVertexInputSignature, GpuShaderIoLocation,
};

pub(crate) fn compare_vertex_input_signatures(
    expected: &GpuExpectedVertexInputSignature,
    observed: &GpuObservedVertexInputSignature,
) -> Result<(), GpuProgramContractError> {
    compare_entry_points(
        "vertex input",
        expected.entry_point(),
        observed.entry_point(),
    )?;
    compare_locations("vertex input", expected.locations(), observed.locations())
}

pub(crate) fn compare_fragment_output_signatures(
    expected: &GpuExpectedFragmentOutputSignature,
    observed: &GpuObservedFragmentOutputSignature,
) -> Result<(), GpuProgramContractError> {
    compare_entry_points(
        "fragment output",
        expected.entry_point(),
        observed.entry_point(),
    )?;
    compare_locations(
        "fragment output",
        expected.locations(),
        observed.locations(),
    )
}

fn compare_entry_points(
    role: &'static str,
    expected: &GpuEntryPointName,
    observed: &GpuEntryPointName,
) -> Result<(), GpuProgramContractError> {
    if expected == observed {
        return Ok(());
    }
    Err(GpuProgramContractError::invalid(
        "compare GPU shader-stage IO signatures",
        format!("{role} expected_entry={expected} observed_entry={observed}"),
        GpuProgramContractCause::PipelineStageIoMismatch,
        "compare pipeline state against observations for the selected entry point",
    ))
}

fn compare_locations<'a>(
    role: &'static str,
    expected: impl Iterator<Item = &'a GpuShaderIoLocation>,
    observed: impl Iterator<Item = &'a GpuShaderIoLocation>,
) -> Result<(), GpuProgramContractError> {
    let mut expected = expected.copied().peekable();
    let mut observed = observed.copied().peekable();

    loop {
        match (expected.peek().copied(), observed.peek().copied()) {
            (None, None) => return Ok(()),
            (Some(expected_location), None) => {
                return Err(mismatch(
                    role,
                    expected_location.location(),
                    "missing observed location",
                ));
            }
            (None, Some(observed_location)) => {
                return Err(mismatch(
                    role,
                    observed_location.location(),
                    "unexpected observed location",
                ));
            }
            (Some(expected_location), Some(observed_location)) => {
                match expected_location
                    .location()
                    .cmp(&observed_location.location())
                {
                    core::cmp::Ordering::Less => {
                        return Err(mismatch(
                            role,
                            expected_location.location(),
                            "missing observed location",
                        ));
                    }
                    core::cmp::Ordering::Greater => {
                        return Err(mismatch(
                            role,
                            observed_location.location(),
                            "unexpected observed location",
                        ));
                    }
                    core::cmp::Ordering::Equal => {
                        if expected_location.value_type() != observed_location.value_type() {
                            return Err(GpuProgramContractError::invalid(
                                "compare GPU shader-stage IO signatures",
                                format!(
                                    "{role} location={} expected={:?} observed={:?}",
                                    expected_location.location(),
                                    expected_location.value_type(),
                                    observed_location.value_type()
                                ),
                                GpuProgramContractCause::PipelineStageIoMismatch,
                                "make the observed scalar class and vector width match the explicit pipeline expectation",
                            ));
                        }
                        expected.next();
                        observed.next();
                    }
                }
            }
        }
    }
}

fn mismatch(role: &'static str, location: u32, reason: &'static str) -> GpuProgramContractError {
    GpuProgramContractError::invalid(
        "compare GPU shader-stage IO signatures",
        format!("{role} location={location}: {reason}"),
        GpuProgramContractCause::PipelineStageIoMismatch,
        "make expected and observed shader locations agree exactly",
    )
}
