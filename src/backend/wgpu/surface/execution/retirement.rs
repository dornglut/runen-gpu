use super::super::{WgpuSurfaceState, validate_handle};
use crate::{GpuContext, GpuSurfaceError, GpuSurfaceHandle};

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
}
