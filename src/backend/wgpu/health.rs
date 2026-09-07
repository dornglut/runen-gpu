//! Context-wide private WGPU health and error-attribution ownership.
//!
//! WGPU 30 error scopes are thread-local under `std`, and their guards are non-send. Async tasks
//! can still interleave scoped dispatch on the same owner thread, so all current-device operations
//! share one short-lived gate. Device-loss and uncaptured-error observation is installed exactly
//! once by `WgpuContextState`.

use crate::{
    GpuContextAffinity, GpuProgramBindingRealizationError,
    GpuProgramBindingRealizationErrorCategory, GpuResourceRealizationError,
    GpuResourceRealizationErrorCategory, GpuWorkResourceId,
};
use std::sync::{Arc, Mutex, MutexGuard};
use wgpu::{Device, DeviceLostReason};

const MAX_BACKEND_EVIDENCE_CHARS: usize = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WgpuDeviceFaultClass {
    UnexpectedValidation,
    OutOfMemory,
    InternalOrDeviceLost,
}

/// One retained context/device-generation fault fact plus bounded lower-precedence evidence.
///
/// A later device loss or internal fault must displace an earlier validation or OOM observation:
/// the retained class is therefore priority ordered rather than merely first-observed.
#[derive(Debug, Clone)]
pub(crate) struct WgpuDeviceFaultEvidence {
    pub(crate) class: WgpuDeviceFaultClass,
    pub(crate) detail: String,
    pub(crate) secondary_detail: Option<String>,
}

/// The sole context-wide owner of terminal WGPU device-health facts.
///
/// It retains the highest-precedence terminal class plus bounded lower-precedence diagnostic
/// evidence. Owner surfaces translate those backend-neutral facts into their own structured
/// errors.
#[derive(Debug)]
pub(crate) struct WgpuDeviceHealth {
    fault: Mutex<Option<WgpuDeviceFaultEvidence>>,
}

impl WgpuDeviceHealth {
    pub(crate) fn new() -> Self {
        Self {
            fault: Mutex::new(None),
        }
    }

    pub(crate) fn install_observers(self: &Arc<Self>, device: &Device) {
        let lost_health = Arc::clone(self);
        device.set_device_lost_callback(move |reason, detail| {
            lost_health.mark_lost(reason, detail);
        });

        let uncaptured_health = Arc::clone(self);
        device.on_uncaptured_error(Arc::new(move |error| {
            uncaptured_health.mark_uncaptured(error);
        }));
    }

    pub(crate) fn mark_uncaptured(&self, error: wgpu::Error) {
        let class = match error {
            wgpu::Error::Validation { .. } => WgpuDeviceFaultClass::UnexpectedValidation,
            wgpu::Error::OutOfMemory { .. } => WgpuDeviceFaultClass::OutOfMemory,
            wgpu::Error::Internal { .. } => WgpuDeviceFaultClass::InternalOrDeviceLost,
        };
        self.mark_fault(class, format!("uncaptured WGPU backend error: {error}"));
    }

    pub(crate) fn mark_scoped_internal(&self, detail: impl Into<String>) {
        self.mark_fault(WgpuDeviceFaultClass::InternalOrDeviceLost, detail);
    }

    fn mark_lost(&self, reason: DeviceLostReason, detail: String) {
        let detail = bounded(detail);
        let diagnostic = if detail.trim().is_empty() {
            format!("device became unavailable ({reason:?})")
        } else {
            format!("device became unavailable ({reason:?}): {detail}")
        };
        self.mark_fault(WgpuDeviceFaultClass::InternalOrDeviceLost, diagnostic);
    }

    fn mark_fault(&self, class: WgpuDeviceFaultClass, detail: impl Into<String>) {
        let mut retained = self
            .fault
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let incoming = WgpuDeviceFaultEvidence {
            class,
            detail: bounded(detail.into()),
            secondary_detail: None,
        };
        let Some(current) = retained.as_mut() else {
            *retained = Some(incoming);
            return;
        };

        if fault_precedence(incoming.class) > fault_precedence(current.class) {
            let replaced = format!("lower-precedence shared WGPU fault: {}", current.detail);
            let mut replacement = incoming;
            replacement.secondary_detail =
                append_secondary(current.secondary_detail.take(), replaced);
            *current = replacement;
        } else {
            let observation = if incoming.class == current.class {
                "additional shared WGPU fault"
            } else {
                "lower-precedence shared WGPU fault"
            };
            current.secondary_detail = append_secondary(
                current.secondary_detail.take(),
                format!("{observation}: {}", incoming.detail),
            );
        }
    }

    pub(crate) fn ensure_resource(
        &self,
        resource: GpuWorkResourceId,
    ) -> Result<(), GpuResourceRealizationError> {
        let Some(fault) = self.current_fault() else {
            return Ok(());
        };
        let category = match fault.class {
            WgpuDeviceFaultClass::UnexpectedValidation => {
                GpuResourceRealizationErrorCategory::UnexpectedBackendValidationRejection
            }
            WgpuDeviceFaultClass::OutOfMemory => {
                GpuResourceRealizationErrorCategory::BackendResourceExhaustion
            }
            WgpuDeviceFaultClass::InternalOrDeviceLost => {
                GpuResourceRealizationErrorCategory::ContextOrDeviceUnavailableOrLost
            }
        };
        Err(GpuResourceRealizationError::new(
            category,
            Some(resource),
            fault.detail,
        ))
    }

    pub(crate) fn ensure_program_binding(
        &self,
        request: impl Into<String>,
    ) -> Result<(), GpuProgramBindingRealizationError> {
        let Some(fault) = self.current_fault() else {
            return Ok(());
        };
        let category = match fault.class {
            WgpuDeviceFaultClass::UnexpectedValidation => {
                GpuProgramBindingRealizationErrorCategory::UnexpectedBackendProgramOrBindingValidationRejection
            }
            WgpuDeviceFaultClass::OutOfMemory => {
                GpuProgramBindingRealizationErrorCategory::BackendResourceExhaustion
            }
            WgpuDeviceFaultClass::InternalOrDeviceLost => {
                GpuProgramBindingRealizationErrorCategory::ContextOrDeviceUnavailableOrLost
            }
        };
        let error = GpuProgramBindingRealizationError::new(category, request, &fault.detail);
        let error = match fault.secondary_detail {
            Some(detail) => error.with_secondary_detail(detail),
            None => error,
        };
        Err(error)
    }

    /// Returns the retained terminal backend fact for precedence handling after a scoped G4C2
    /// creation attempt. The owning realization still translates it into its public error family.
    pub(crate) fn terminal_fault(&self) -> Option<WgpuDeviceFaultEvidence> {
        self.current_fault()
    }

    fn current_fault(&self) -> Option<WgpuDeviceFaultEvidence> {
        self.fault
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

fn fault_precedence(class: WgpuDeviceFaultClass) -> u8 {
    match class {
        WgpuDeviceFaultClass::InternalOrDeviceLost => 3,
        WgpuDeviceFaultClass::OutOfMemory => 2,
        WgpuDeviceFaultClass::UnexpectedValidation => 1,
    }
}

fn append_secondary(existing: Option<String>, next: impl AsRef<str>) -> Option<String> {
    let next = next.as_ref().trim();
    if next.is_empty() {
        return existing;
    }
    Some(bounded(match existing {
        Some(existing) => format!("{existing}; {next}"),
        None => next.to_owned(),
    }))
}

/// The one private current-device error-scope attribution gate.
///
/// It serializes only synchronous push/create/pop dispatch. Users must release the returned guard
/// before awaiting any popped scope future.
#[derive(Debug, Default)]
pub(crate) struct WgpuErrorAttributionGate {
    mutex: Mutex<()>,
}

impl WgpuErrorAttributionGate {
    pub(crate) fn acquire(&self) -> MutexGuard<'_, ()> {
        self.mutex
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

pub(crate) fn validate_program_affinity(
    expected: GpuContextAffinity,
    request: impl Into<String>,
    observed: GpuContextAffinity,
) -> Result<(), GpuProgramBindingRealizationError> {
    if observed.context() != expected.context() {
        return Err(GpuProgramBindingRealizationError::affinity(
            GpuProgramBindingRealizationErrorCategory::ForeignContext,
            request,
            expected,
            observed,
        ));
    }
    if observed.generation() != expected.generation() {
        return Err(GpuProgramBindingRealizationError::affinity(
            GpuProgramBindingRealizationErrorCategory::StaleDeviceGeneration,
            request,
            expected,
            observed,
        ));
    }
    Ok(())
}

fn bounded(value: impl Into<String>) -> String {
    value
        .into()
        .chars()
        .take(MAX_BACKEND_EVIDENCE_CHARS)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn higher_precedence_faults_replace_lower_shared_health_facts() {
        let health = WgpuDeviceHealth::new();
        health.mark_fault(
            WgpuDeviceFaultClass::UnexpectedValidation,
            "validation fact",
        );
        health.mark_fault(WgpuDeviceFaultClass::OutOfMemory, "OOM fact");
        health.mark_fault(
            WgpuDeviceFaultClass::InternalOrDeviceLost,
            "device-loss fact",
        );
        // A later lower-priority observation cannot displace the terminal loss fact.
        health.mark_fault(
            WgpuDeviceFaultClass::UnexpectedValidation,
            "later validation fact",
        );

        let fault = health
            .terminal_fault()
            .expect("a retained health fact should remain observable");
        assert_eq!(fault.class, WgpuDeviceFaultClass::InternalOrDeviceLost);
        assert_eq!(fault.detail, "device-loss fact");
        assert!(fault.secondary_detail.as_deref().is_some_and(|detail| {
            detail.contains("validation fact")
                && detail.contains("OOM fact")
                && detail.len() <= MAX_BACKEND_EVIDENCE_CHARS
        }));

        let error = health
            .ensure_program_binding("test program")
            .expect_err("terminal loss must reject subsequent G4C2 realization");
        assert_eq!(
            error.category(),
            GpuProgramBindingRealizationErrorCategory::ContextOrDeviceUnavailableOrLost
        );
        assert!(error.secondary_detail().is_some());
    }
}
