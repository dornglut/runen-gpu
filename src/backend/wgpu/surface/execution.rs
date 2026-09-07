mod retirement;

use super::super::WgpuDeviceHealth;
use super::{WgpuSurfaceLease, WgpuSurfaceState, WgpuSurfaceStateInner};
use crate::{
    GpuContextAffinity, GpuSurfaceLeaseDisposition, GpuSurfaceLeaseError,
    GpuSurfaceLeaseErrorCategory, GpuSurfaceLeaseId, GpuSurfaceLeaseOwner, GpuSurfaceResourceLease,
    GpuWorkResourceId,
};
use std::collections::BTreeMap;
use std::sync::{Arc, MutexGuard};
use wgpu::{Queue, Texture, TextureView, TextureViewDescriptor};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum WgpuSurfaceLeaseResource {
    Texture(GpuWorkResourceId),
    TextureView(GpuWorkResourceId),
}

impl WgpuSurfaceLeaseResource {
    const fn identity(self) -> GpuWorkResourceId {
        match self {
            Self::Texture(identity) | Self::TextureView(identity) => identity,
        }
    }
}

/// Short private ownership interval used by G5 submission execution.
///
/// The surface mutex prevents reconfiguration/acquisition/abandon release from mutating physical
/// surface-image ownership while accepted work is encoded and submitted. `pinned_owners` retains
/// only the private acquisition owner for leases actually validated for this interval; logical
/// handles themselves remain non-owning with respect to the physical `SurfaceTexture`.
pub(crate) struct WgpuSurfaceLeaseGuard<'a> {
    affinity: GpuContextAffinity,
    health: &'a WgpuDeviceHealth,
    inner: MutexGuard<'a, WgpuSurfaceStateInner>,
    pinned_owners: BTreeMap<GpuSurfaceLeaseId, Arc<GpuSurfaceLeaseOwner>>,
}

impl WgpuSurfaceState {
    /// Validates one surface-acquired logical resource against the sole G7 physical lease owner.
    ///
    /// This is intentionally separate from G4C1 realization: surface-acquired resources are
    /// transient presentation leases and never become ordinary resource-registry records.
    pub(crate) fn validate_execution_lease(
        &self,
        lease: &GpuSurfaceResourceLease,
        resource: WgpuSurfaceLeaseResource,
        health: &WgpuDeviceHealth,
    ) -> Result<(), GpuSurfaceLeaseError> {
        ensure_lease_health(health, lease)?;
        let mut inner = self.shared.inner.lock().map_err(|_| {
            lease_error(
                GpuSurfaceLeaseErrorCategory::ContextOrDeviceUnavailableOrLost,
                lease,
                "surface lease validation authority is unavailable",
            )
        })?;
        if let Some(record) = inner.records.get_mut(&lease.surface().id()) {
            super::release_abandoned_lease(record);
        }
        validate_lease(self.affinity, &inner, lease, resource)?;
        ensure_lease_health(health, lease)
    }

    pub(crate) fn execution_lease_guard<'a>(
        &'a self,
        representative: &GpuSurfaceResourceLease,
        health: &'a WgpuDeviceHealth,
    ) -> Result<WgpuSurfaceLeaseGuard<'a>, GpuSurfaceLeaseError> {
        ensure_lease_health(health, representative)?;
        validate_affinity(self.affinity, representative)?;
        let mut inner = self.shared.inner.lock().map_err(|_| {
            lease_error(
                GpuSurfaceLeaseErrorCategory::ContextOrDeviceUnavailableOrLost,
                representative,
                "surface lease execution authority is unavailable",
            )
        })?;
        for record in inner.records.values_mut() {
            super::release_abandoned_lease(record);
        }
        ensure_lease_health(health, representative)?;
        Ok(WgpuSurfaceLeaseGuard {
            affinity: self.affinity,
            health,
            inner,
            pinned_owners: BTreeMap::new(),
        })
    }
}

impl WgpuSurfaceLeaseGuard<'_> {
    /// Revalidates one logical resource and pins its private physical owner for this submit interval.
    pub(crate) fn validate_and_pin(
        &mut self,
        lease: &GpuSurfaceResourceLease,
        resource: WgpuSurfaceLeaseResource,
    ) -> Result<(), GpuSurfaceLeaseError> {
        ensure_lease_health(self.health, lease)?;
        let owner = {
            let active = validate_lease(self.affinity, &self.inner, lease, resource)?;
            active.owner.upgrade().ok_or_else(|| {
                lease_error(
                    GpuSurfaceLeaseErrorCategory::InvalidLease,
                    lease,
                    "surface acquisition owner was abandoned before execution acceptance",
                )
            })?
        };
        self.pinned_owners.entry(lease.lease_id()).or_insert(owner);
        ensure_lease_health(self.health, lease)
    }

    /// Resolves an already-pinned surface texture without entering G4C1 realization.
    pub(crate) fn texture(
        &self,
        lease: &GpuSurfaceResourceLease,
        identity: GpuWorkResourceId,
    ) -> Result<&Texture, GpuSurfaceLeaseError> {
        ensure_pinned(self, lease)?;
        let active = validate_lease(
            self.affinity,
            &self.inner,
            lease,
            WgpuSurfaceLeaseResource::Texture(identity),
        )?;
        Ok(&active.texture.texture)
    }

    /// Resolves the explicit G7A default view for an already-pinned surface lease.
    pub(crate) fn create_default_view(
        &self,
        lease: &GpuSurfaceResourceLease,
        identity: GpuWorkResourceId,
    ) -> Result<TextureView, GpuSurfaceLeaseError> {
        ensure_pinned(self, lease)?;
        let active = validate_lease(
            self.affinity,
            &self.inner,
            lease,
            WgpuSurfaceLeaseResource::TextureView(identity),
        )?;
        Ok(active
            .texture
            .texture
            .create_view(&TextureViewDescriptor::default()))
    }

    /// Consumes and presents one already-pinned physical acquisition exactly once.
    pub(crate) fn present(
        &mut self,
        queue: &Queue,
        lease: &GpuSurfaceResourceLease,
        resource: WgpuSurfaceLeaseResource,
    ) -> Result<(), GpuSurfaceLeaseError> {
        ensure_pinned(self, lease)?;
        validate_lease(self.affinity, &self.inner, lease, resource)?;

        let record = self
            .inner
            .records
            .get_mut(&lease.surface().id())
            .ok_or_else(|| {
                lease_error(
                    GpuSurfaceLeaseErrorCategory::UnknownSurface,
                    lease,
                    "surface identity disappeared before Present",
                )
            })?;
        let Some(active) = record.active_lease.take() else {
            return Err(lease_error(
                GpuSurfaceLeaseErrorCategory::InvalidLease,
                lease,
                "active physical surface lease disappeared before Present",
            ));
        };
        if active.lease != *lease {
            record.active_lease = Some(active);
            return Err(lease_error(
                GpuSurfaceLeaseErrorCategory::InvalidLease,
                lease,
                "active physical surface lease changed before Present",
            ));
        }
        if let Err(disposition) = lease.mark_presented() {
            record.active_lease = Some(active);
            return Err(disposition_error(
                lease,
                disposition,
                "surface lease cannot be presented",
            ));
        }
        queue.present(active.texture);
        Ok(())
    }
}

fn ensure_pinned(
    guard: &WgpuSurfaceLeaseGuard<'_>,
    lease: &GpuSurfaceResourceLease,
) -> Result<(), GpuSurfaceLeaseError> {
    ensure_lease_health(guard.health, lease)?;
    if guard.pinned_owners.contains_key(&lease.lease_id()) {
        Ok(())
    } else {
        Err(lease_error(
            GpuSurfaceLeaseErrorCategory::InvalidLease,
            lease,
            "surface lease was not validated and pinned before private execution",
        ))
    }
}

fn validate_lease<'a>(
    expected: GpuContextAffinity,
    inner: &'a WgpuSurfaceStateInner,
    lease: &GpuSurfaceResourceLease,
    resource: WgpuSurfaceLeaseResource,
) -> Result<&'a WgpuSurfaceLease, GpuSurfaceLeaseError> {
    validate_affinity(expected, lease)?;
    let record = inner.records.get(&lease.surface().id()).ok_or_else(|| {
        lease_error(
            GpuSurfaceLeaseErrorCategory::UnknownSurface,
            lease,
            "surface identity is absent from this context-local G7 owner",
        )
    })?;
    if record.generation != lease.surface().generation() {
        return Err(lease_error(
            GpuSurfaceLeaseErrorCategory::StaleGeneration,
            lease,
            "surface-acquired resource belongs to a stale surface generation",
        ));
    }
    validate_disposition(lease)?;
    let active = record.active_lease.as_ref().ok_or_else(|| {
        lease_error(
            GpuSurfaceLeaseErrorCategory::InvalidLease,
            lease,
            "the surface has no active acquired-image lease",
        )
    })?;
    if active.lease != *lease {
        return Err(lease_error(
            GpuSurfaceLeaseErrorCategory::InvalidLease,
            lease,
            "surface-acquired resource does not name the current active lease",
        ));
    }
    let identity_matches = match resource {
        WgpuSurfaceLeaseResource::Texture(identity) => active.texture_identity == identity,
        WgpuSurfaceLeaseResource::TextureView(identity) => active.view_identity == identity,
    };
    if !identity_matches {
        return Err(lease_error(
            GpuSurfaceLeaseErrorCategory::InvalidLease,
            lease,
            format!(
                "logical {:?} identity {:?} is not owned by this surface lease",
                resource,
                resource.identity()
            ),
        ));
    }
    Ok(active)
}

fn validate_affinity(
    expected: GpuContextAffinity,
    lease: &GpuSurfaceResourceLease,
) -> Result<(), GpuSurfaceLeaseError> {
    let observed = lease.surface().affinity();
    if observed.context() != expected.context() {
        return Err(lease_error(
            GpuSurfaceLeaseErrorCategory::ForeignContext,
            lease,
            "surface-acquired resource belongs to a different GPU context",
        ));
    }
    if observed.generation() != expected.generation() {
        return Err(lease_error(
            GpuSurfaceLeaseErrorCategory::StaleGeneration,
            lease,
            "surface-acquired resource belongs to a stale device generation",
        ));
    }
    Ok(())
}

fn validate_disposition(lease: &GpuSurfaceResourceLease) -> Result<(), GpuSurfaceLeaseError> {
    match lease.disposition() {
        GpuSurfaceLeaseDisposition::Active => Ok(()),
        disposition => Err(disposition_error(
            lease,
            disposition,
            "surface lease is no longer active",
        )),
    }
}

fn disposition_error(
    lease: &GpuSurfaceResourceLease,
    disposition: GpuSurfaceLeaseDisposition,
    detail: &'static str,
) -> GpuSurfaceLeaseError {
    let category = match disposition {
        GpuSurfaceLeaseDisposition::Active | GpuSurfaceLeaseDisposition::Abandoned => {
            GpuSurfaceLeaseErrorCategory::InvalidLease
        }
        GpuSurfaceLeaseDisposition::Presented => GpuSurfaceLeaseErrorCategory::AlreadyConsumed,
    };
    lease_error(category, lease, detail)
}

fn ensure_lease_health(
    health: &WgpuDeviceHealth,
    lease: &GpuSurfaceResourceLease,
) -> Result<(), GpuSurfaceLeaseError> {
    if let Some(fault) = health.terminal_fault() {
        Err(lease_error(
            GpuSurfaceLeaseErrorCategory::ContextOrDeviceUnavailableOrLost,
            lease,
            fault.detail,
        ))
    } else {
        Ok(())
    }
}

fn lease_error(
    category: GpuSurfaceLeaseErrorCategory,
    lease: &GpuSurfaceResourceLease,
    detail: impl Into<String>,
) -> GpuSurfaceLeaseError {
    GpuSurfaceLeaseError::new(category, lease.surface().id(), lease.lease_id(), detail)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        GpuContextId, GpuDeviceGeneration, GpuSurfaceGeneration, GpuSurfaceHandle,
        allocate_surface_id, allocate_surface_lease_id,
    };
    use std::num::NonZeroU64;

    fn affinity(context: u64, generation: u64) -> GpuContextAffinity {
        GpuContextAffinity::test_value(
            GpuContextId::test_value(NonZeroU64::new(context).unwrap()),
            GpuDeviceGeneration::test_value(NonZeroU64::new(generation).unwrap()),
        )
    }

    fn lease(affinity: GpuContextAffinity) -> GpuSurfaceResourceLease {
        let surface = GpuSurfaceHandle::new(
            allocate_surface_id().unwrap(),
            affinity,
            GpuSurfaceGeneration::first(),
        );
        let lease_id = allocate_surface_lease_id(surface.id()).unwrap();
        GpuSurfaceResourceLease::new(surface, lease_id)
    }

    #[test]
    fn lease_disposition_distinguishes_abandoned_from_present_consumed() {
        let abandoned = lease(affinity(1, 1));
        abandoned.mark_abandoned();
        let error = validate_disposition(&abandoned).unwrap_err();
        assert_eq!(error.category(), GpuSurfaceLeaseErrorCategory::InvalidLease);

        let presented = lease(affinity(1, 1));
        presented.mark_presented().unwrap();
        let error = validate_disposition(&presented).unwrap_err();
        assert_eq!(
            error.category(),
            GpuSurfaceLeaseErrorCategory::AlreadyConsumed
        );
    }

    #[test]
    fn lease_affinity_distinguishes_foreign_context_and_stale_generation() {
        let expected = affinity(1, 1);

        let foreign_lease = lease(affinity(2, 1));
        let foreign = validate_affinity(expected, &foreign_lease).unwrap_err();
        assert_eq!(
            foreign.category(),
            GpuSurfaceLeaseErrorCategory::ForeignContext
        );

        let stale_lease = lease(affinity(1, 2));
        let stale = validate_affinity(expected, &stale_lease).unwrap_err();
        assert_eq!(
            stale.category(),
            GpuSurfaceLeaseErrorCategory::StaleGeneration
        );
    }
}
