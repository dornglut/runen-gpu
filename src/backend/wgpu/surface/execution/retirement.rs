use super::super::{WgpuSurfaceState, validate_handle};
use crate::{GpuContext, GpuSurfaceConditionalDetachOutcome, GpuSurfaceError, GpuSurfaceHandle};

impl WgpuSurfaceState {
    /// Retires one registered surface and releases its private physical authority.
    ///
    /// Retirement is deliberately independent of device health so callers can always release a
    /// window surface after terminal device failure. The exact context/generation handle is still
    /// validated under the same mutex used by acquisition, reconfiguration, and G5 Present.
    fn detach(&self, handle: GpuSurfaceHandle) -> Result<(), GpuSurfaceError> {
        let mut inner = self.inner(Some(handle.id()))?;
        validate_handle(self.affinity, &inner.records, handle)?;
        let mut record = inner
            .records
            .remove(&handle.id())
            .expect("validated surface remains registered while retirement authority is held");
        if let Some(active) = record.active_lease.take() {
            active.lease.mark_abandoned();
        }
        drop(inner);
        drop(record);
        Ok(())
    }

    /// The active-lease guard and physical removal share the exact mutex used
    /// by acquisition and G5 Present. A busy outcome preserves the surface and
    /// the original lease so its owner can still complete presentation.
    fn detach_if_unleased(
        &self,
        handle: GpuSurfaceHandle,
    ) -> Result<GpuSurfaceConditionalDetachOutcome, GpuSurfaceError> {
        let mut inner = self.inner(Some(handle.id()))?;
        validate_handle(self.affinity, &inner.records, handle)?;
        let record = inner
            .records
            .get(&handle.id())
            .expect("validated surface remains registered while guard is held");
        if record.active_lease.is_some() {
            return Ok(GpuSurfaceConditionalDetachOutcome::ActiveLease);
        }
        let record = inner
            .records
            .remove(&handle.id())
            .expect("validated unleased surface remains registered while guard is held");
        drop(inner);
        drop(record);
        Ok(GpuSurfaceConditionalDetachOutcome::Detached)
    }
}

impl GpuContext {
    /// Retires one context-local presentation surface.
    ///
    /// Any active acquired-image lease is abandoned before the private physical surface is
    /// dropped. Caller-held logical texture/view handles never retain that physical authority and
    /// subsequent use of the retired surface handle is rejected as an unknown surface.
    pub fn detach_surface(&self, surface: GpuSurfaceHandle) -> Result<(), GpuSurfaceError> {
        self.backend.surfaces.detach(surface)
    }

    /// Conditionally retires one exact physical surface without abandoning its
    /// currently acquired image. The lease check and retirement are atomic
    /// against G7 acquisition, configuration, and Present.
    ///
    /// ActiveLease preserves the registered physical surface and original
    /// acquisition; callers may retry only after the lease owner releases it.
    /// Detached does not establish that independent G5 submissions, pending
    /// readbacks, UI publications, or product-owned work are terminal.
    ///
    /// Terminal teardown that must abandon an active lease keeps using
    /// `detach_surface`; unknown/stale/foreign handles return their existing
    /// GpuSurfaceError rather than a successful no-op.
    pub fn detach_surface_if_unleased(
        &self,
        surface: GpuSurfaceHandle,
    ) -> Result<GpuSurfaceConditionalDetachOutcome, GpuSurfaceError> {
        self.backend.surfaces.detach_if_unleased(surface)
    }
}
