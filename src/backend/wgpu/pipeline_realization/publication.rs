use super::PipelineRealizationState;
use crate::{GpuPipelineRealizationError, GpuPipelineRealizationErrorCategory};
use wgpu::ErrorFilter;

pub(super) fn ensure_available(
    realization: &PipelineRealizationState,
    request: impl Into<String>,
) -> Result<(), GpuPipelineRealizationError> {
    let request = request.into();
    match realization.health.terminal_fault() {
        Some(fault) => Err(health_failure(request, &fault)),
        None => Ok(()),
    }
}

/// Pushes all required WGPU scopes, creates one pipeline, pops every scope while the shared
/// attribution gate is held, releases the non-reentrant gate, then awaits before publication.
pub(super) async fn scoped_create<T>(
    device: &wgpu::Device,
    realization: &PipelineRealizationState,
    request: String,
    create: impl FnOnce() -> T,
) -> Result<T, GpuPipelineRealizationError> {
    ensure_available(realization, request.clone())?;
    let (candidate, validation, out_of_memory, internal) = {
        let _gate = realization.error_attribution_gate.acquire();
        let internal_scope = device.push_error_scope(ErrorFilter::Internal);
        let out_of_memory_scope = device.push_error_scope(ErrorFilter::OutOfMemory);
        let validation_scope = device.push_error_scope(ErrorFilter::Validation);
        let candidate = create();
        let validation = validation_scope.pop();
        let out_of_memory = out_of_memory_scope.pop();
        let internal = internal_scope.pop();
        (candidate, validation, out_of_memory, internal)
    };
    let validation = validation.await;
    let out_of_memory = out_of_memory.await;
    let internal = internal.await;
    let validation_detail = validation.map(|error| format!("Validation scope: {error}"));
    let out_of_memory_detail = out_of_memory.map(|error| format!("OutOfMemory scope: {error}"));
    let internal_detail = internal.map(|error| format!("Internal scope: {error}"));
    let health_fault = realization.health.terminal_fault();

    if let Some(fault) = health_fault.as_ref().filter(|fault| {
        fault.class == super::super::health::WgpuDeviceFaultClass::InternalOrDeviceLost
    }) {
        return Err(scoped_failure(
            GpuPipelineRealizationErrorCategory::ContextOrDeviceUnavailableOrLost,
            request,
            format!("shared device health: {}", fault.detail),
            [
                fault.secondary_detail.clone(),
                internal_detail,
                out_of_memory_detail,
                validation_detail,
            ],
        ));
    }
    if let Some(detail) = internal_detail {
        realization
            .health
            .mark_scoped_internal(format!("scoped WGPU internal error: {detail}"));
        return Err(scoped_failure(
            GpuPipelineRealizationErrorCategory::ContextOrDeviceUnavailableOrLost,
            request,
            detail,
            [
                out_of_memory_detail,
                validation_detail,
                health_fault.as_ref().map(shared_health_evidence),
            ],
        ));
    }
    if let Some(fault) = health_fault
        .as_ref()
        .filter(|fault| fault.class == super::super::health::WgpuDeviceFaultClass::OutOfMemory)
    {
        return Err(scoped_failure(
            GpuPipelineRealizationErrorCategory::BackendResourceExhaustion,
            request,
            format!("shared device health: {}", fault.detail),
            [
                fault.secondary_detail.clone(),
                out_of_memory_detail,
                validation_detail,
            ],
        ));
    }
    if let Some(detail) = out_of_memory_detail {
        return Err(scoped_failure(
            GpuPipelineRealizationErrorCategory::BackendResourceExhaustion,
            request,
            detail,
            [
                validation_detail,
                health_fault.as_ref().map(shared_health_evidence),
            ],
        ));
    }
    if let Some(detail) = validation_detail {
        return Err(scoped_failure(
            GpuPipelineRealizationErrorCategory::UnexpectedBackendPipelineValidationRejection,
            request,
            detail,
            [health_fault.as_ref().map(shared_health_evidence)],
        ));
    }
    ensure_available(realization, request)?;
    Ok(candidate)
}

fn health_failure(
    request: String,
    fault: &super::super::health::WgpuDeviceFaultEvidence,
) -> GpuPipelineRealizationError {
    let category = match fault.class {
        super::super::health::WgpuDeviceFaultClass::UnexpectedValidation => {
            GpuPipelineRealizationErrorCategory::UnexpectedBackendPipelineValidationRejection
        }
        super::super::health::WgpuDeviceFaultClass::OutOfMemory => {
            GpuPipelineRealizationErrorCategory::BackendResourceExhaustion
        }
        super::super::health::WgpuDeviceFaultClass::InternalOrDeviceLost => {
            GpuPipelineRealizationErrorCategory::ContextOrDeviceUnavailableOrLost
        }
    };
    let error = GpuPipelineRealizationError::new(category, request, fault.detail.clone());
    match &fault.secondary_detail {
        Some(detail) => error.with_secondary_detail(detail.clone()),
        None => error,
    }
}

fn shared_health_evidence(fault: &super::super::health::WgpuDeviceFaultEvidence) -> String {
    match fault.secondary_detail.as_deref() {
        Some(secondary) => format!("shared device health: {}; {secondary}", fault.detail),
        None => format!("shared device health: {}", fault.detail),
    }
}

fn scoped_failure(
    category: GpuPipelineRealizationErrorCategory,
    request: String,
    detail: impl Into<String>,
    secondary_details: impl IntoIterator<Item = Option<String>>,
) -> GpuPipelineRealizationError {
    let secondary_detail = secondary_details
        .into_iter()
        .flatten()
        .filter(|detail| !detail.trim().is_empty())
        .take(3)
        .collect::<Vec<_>>()
        .join("; ");
    let error = GpuPipelineRealizationError::new(category, request, detail);
    if secondary_detail.is_empty() {
        error
    } else {
        error.with_secondary_detail(secondary_detail)
    }
}
