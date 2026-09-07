use super::{
    GpuContextAffinity, GpuSurfaceGeneration, GpuSurfaceHandle, GpuSurfaceId, GpuTextureHandle,
    GpuTextureViewHandle,
};
use core::fmt;
use core::hash::{Hash, Hasher};
use core::num::NonZeroU64;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::sync::{Arc, Weak};

/// Opaque process-local identity for one acquired surface-image lease.
///
/// The value is correlation identity only. It has no persistence, replay, wire, ABI, or cache
/// representation guarantee and cannot be constructed by callers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuSurfaceLeaseId(NonZeroU64);

#[derive(Debug)]
struct GpuSurfaceLeaseIdAllocator {
    next: AtomicU64,
}

impl GpuSurfaceLeaseIdAllocator {
    const fn new(next: u64) -> Self {
        Self {
            next: AtomicU64::new(next),
        }
    }

    fn allocate(&self, surface: GpuSurfaceId) -> Result<GpuSurfaceLeaseId, GpuSurfaceAcquireError> {
        let value = self
            .next
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                (current != 0).then_some(if current == u64::MAX { 0 } else { current + 1 })
            })
            .map_err(|_| {
                GpuSurfaceAcquireError::new(
                    GpuSurfaceAcquireErrorCategory::IdentityExhausted,
                    Some(surface),
                    "surface lease identifier allocator exhausted",
                )
            })?;
        Ok(GpuSurfaceLeaseId(NonZeroU64::new(value).expect(
            "surface lease identifier allocator never returns zero",
        )))
    }
}

static PRODUCTION_SURFACE_LEASE_IDS: GpuSurfaceLeaseIdAllocator =
    GpuSurfaceLeaseIdAllocator::new(1);

pub(crate) fn allocate_surface_lease_id(
    surface: GpuSurfaceId,
) -> Result<GpuSurfaceLeaseId, GpuSurfaceAcquireError> {
    PRODUCTION_SURFACE_LEASE_IDS.allocate(surface)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GpuSurfaceLeaseDisposition {
    Active,
    Abandoned,
    Presented,
}

impl GpuSurfaceLeaseDisposition {
    const ACTIVE: u8 = 0;
    const ABANDONED: u8 = 1;
    const PRESENTED: u8 = 2;

    const fn as_u8(self) -> u8 {
        match self {
            Self::Active => Self::ACTIVE,
            Self::Abandoned => Self::ABANDONED,
            Self::Presented => Self::PRESENTED,
        }
    }

    fn from_u8(value: u8) -> Self {
        match value {
            Self::ACTIVE => Self::Active,
            Self::ABANDONED => Self::Abandoned,
            Self::PRESENTED => Self::Presented,
            _ => unreachable!("surface lease disposition is private and closed"),
        }
    }
}

/// One clone-bounded logical surface-lease record.
///
/// The record carries only backend-neutral identity and terminal lifecycle evidence. It owns no
/// physical surface image and no backend callback. Keeping the entire record behind one `Arc`
/// keeps surface provenance pointer-sized at logical handle sites while preserving exact shared
/// Abandoned/Presented evidence for surviving logical clones.
#[derive(Debug)]
struct GpuSurfaceLeaseState {
    surface: GpuSurfaceHandle,
    lease_id: GpuSurfaceLeaseId,
    disposition: AtomicU8,
}

impl GpuSurfaceLeaseState {
    fn new(surface: GpuSurfaceHandle, lease_id: GpuSurfaceLeaseId) -> Self {
        Self {
            surface,
            lease_id,
            disposition: AtomicU8::new(GpuSurfaceLeaseDisposition::Active.as_u8()),
        }
    }

    fn disposition(&self) -> GpuSurfaceLeaseDisposition {
        GpuSurfaceLeaseDisposition::from_u8(self.disposition.load(Ordering::Acquire))
    }

    fn abandon(&self) {
        let _ = self.disposition.compare_exchange(
            GpuSurfaceLeaseDisposition::Active.as_u8(),
            GpuSurfaceLeaseDisposition::Abandoned.as_u8(),
            Ordering::AcqRel,
            Ordering::Acquire,
        );
    }

    fn present(&self) -> Result<(), GpuSurfaceLeaseDisposition> {
        self.disposition
            .compare_exchange(
                GpuSurfaceLeaseDisposition::Active.as_u8(),
                GpuSurfaceLeaseDisposition::Presented.as_u8(),
                Ordering::AcqRel,
                Ordering::Acquire,
            )
            .map(|_| ())
            .map_err(GpuSurfaceLeaseDisposition::from_u8)
    }
}

/// Non-owning provenance carried by a surface-acquired logical texture lease.
///
/// This value deliberately contains no backend object and no strong physical-owner reference.
/// Logical handle clones retain one shared backend-neutral identity/lifecycle record; retaining it
/// cannot keep a physical swapchain image alive. The private G7 owner remains solely responsible
/// for mapping that identity to the exact active physical acquisition.
#[repr(transparent)]
#[derive(Clone)]
pub(crate) struct GpuSurfaceResourceLease(Arc<GpuSurfaceLeaseState>);

impl GpuSurfaceResourceLease {
    pub(crate) fn new(surface: GpuSurfaceHandle, lease_id: GpuSurfaceLeaseId) -> Self {
        Self(Arc::new(GpuSurfaceLeaseState::new(surface, lease_id)))
    }

    pub(crate) fn surface(&self) -> GpuSurfaceHandle {
        self.0.surface
    }

    pub(crate) fn lease_id(&self) -> GpuSurfaceLeaseId {
        self.0.lease_id
    }

    pub(crate) fn disposition(&self) -> GpuSurfaceLeaseDisposition {
        self.0.disposition()
    }

    pub(crate) fn mark_abandoned(&self) {
        self.0.abandon();
    }

    pub(crate) fn mark_presented(&self) -> Result<(), GpuSurfaceLeaseDisposition> {
        self.0.present()
    }
}

impl fmt::Debug for GpuSurfaceResourceLease {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GpuSurfaceResourceLease")
            .field("surface", &self.surface())
            .field("lease_id", &self.lease_id())
            .field("disposition", &self.disposition())
            .finish()
    }
}

impl PartialEq for GpuSurfaceResourceLease {
    fn eq(&self, other: &Self) -> bool {
        self.surface() == other.surface() && self.lease_id() == other.lease_id()
    }
}

impl Eq for GpuSurfaceResourceLease {}

impl Hash for GpuSurfaceResourceLease {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.surface().hash(state);
        self.lease_id().hash(state);
    }
}

/// Successful backend-neutral surface acquisition quality.
///
/// `Suboptimal` preserves backend evidence that the acquired image is usable but the current
/// surface configuration is no longer ideal. RunenGPU does not turn that evidence into automatic
/// reconfiguration policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuSurfaceAcquisitionStatus {
    Optimal,
    Suboptimal,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuSurfaceAcquireErrorCategory {
    NotConfigured,
    AlreadyAcquired,
    Timeout,
    Occluded,
    Outdated,
    Lost,
    Validation,
    UnknownSurface,
    ForeignContext,
    StaleGeneration,
    ContextOrDeviceUnavailableOrLost,
    IdentityExhausted,
    InternalInvariant,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuSurfaceAcquireError {
    category: GpuSurfaceAcquireErrorCategory,
    surface: Option<GpuSurfaceId>,
    detail: Option<String>,
}

impl GpuSurfaceAcquireError {
    pub(crate) fn new(
        category: GpuSurfaceAcquireErrorCategory,
        surface: Option<GpuSurfaceId>,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            category,
            surface,
            detail: super::context::sanitized_diagnostic(detail.into()),
        }
    }

    pub const fn category(&self) -> GpuSurfaceAcquireErrorCategory {
        self.category
    }

    pub const fn surface(&self) -> Option<GpuSurfaceId> {
        self.surface
    }

    pub fn detail(&self) -> Option<&str> {
        self.detail.as_deref()
    }
}

impl fmt::Display for GpuSurfaceAcquireError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "GPU surface acquisition failed ({:?})",
            self.category
        )?;
        if let Some(detail) = &self.detail {
            write!(formatter, ": {detail}")?;
        }
        Ok(())
    }
}

impl std::error::Error for GpuSurfaceAcquireError {}

/// Structured rejection for using a logical surface-acquired resource after its physical lease is
/// no longer valid for the current context/surface generation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuSurfaceLeaseErrorCategory {
    UnknownSurface,
    InvalidLease,
    ForeignContext,
    StaleGeneration,
    AlreadyConsumed,
    ContextOrDeviceUnavailableOrLost,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuSurfaceLeaseError {
    category: GpuSurfaceLeaseErrorCategory,
    surface: GpuSurfaceId,
    lease_id: GpuSurfaceLeaseId,
    detail: Option<String>,
}

impl GpuSurfaceLeaseError {
    pub(crate) fn new(
        category: GpuSurfaceLeaseErrorCategory,
        surface: GpuSurfaceId,
        lease_id: GpuSurfaceLeaseId,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            category,
            surface,
            lease_id,
            detail: super::context::sanitized_diagnostic(detail.into()),
        }
    }

    pub const fn category(&self) -> GpuSurfaceLeaseErrorCategory {
        self.category
    }

    pub const fn surface(&self) -> GpuSurfaceId {
        self.surface
    }

    pub const fn lease_id(&self) -> GpuSurfaceLeaseId {
        self.lease_id
    }

    pub fn detail(&self) -> Option<&str> {
        self.detail.as_deref()
    }
}

impl fmt::Display for GpuSurfaceLeaseError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "GPU surface lease use failed ({:?})",
            self.category
        )?;
        if let Some(detail) = &self.detail {
            write!(formatter, ": {detail}")?;
        }
        Ok(())
    }
}

impl std::error::Error for GpuSurfaceLeaseError {}

/// Private backend callback used only to release one active physical surface-image lease.
///
/// The public acquired image owns no backend object. Its owner token retains only a weak callback,
/// so dropping the context first remains safe and cannot create an ownership cycle.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) trait GpuSurfaceLeaseReleaser: fmt::Debug + Send + Sync {
    fn release(&self, lease: &GpuSurfaceResourceLease);
}

/// Web surface state is deliberately not forced through native thread-safety requirements.
#[cfg(target_arch = "wasm32")]
pub(crate) trait GpuSurfaceLeaseReleaser: fmt::Debug {
    fn release(&self, lease: &GpuSurfaceResourceLease);
}

/// Private ownership marker for the one active physical surface-image lease.
///
/// Only `GpuAcquiredSurfaceImage` and a short private execution interval may retain this physical
/// owner marker. Logical texture/view handles retain only `GpuSurfaceResourceLease` metadata, so
/// cloning them cannot keep physical swapchain authority alive. Dropping this marker marks an
/// unpresented lease abandoned and asks the owning G7 backend state to release the physical image.
#[derive(Debug)]
pub(crate) struct GpuSurfaceLeaseOwner {
    lease: GpuSurfaceResourceLease,
    releaser: Weak<dyn GpuSurfaceLeaseReleaser>,
}

impl GpuSurfaceLeaseOwner {
    pub(crate) fn new(
        lease: GpuSurfaceResourceLease,
        releaser: Weak<dyn GpuSurfaceLeaseReleaser>,
    ) -> Arc<Self> {
        Arc::new(Self { lease, releaser })
    }
}

impl Drop for GpuSurfaceLeaseOwner {
    fn drop(&mut self) {
        self.lease.mark_abandoned();
        if let Some(releaser) = self.releaser.upgrade() {
            releaser.release(&self.lease);
        }
    }
}

/// One acquired surface image and its logical RunenGPU resource identities.
///
/// This owner is intentionally not `Clone`. Dropping it (or consuming it through `abandon`) ends
/// caller ownership of the acquisition lease. The logical texture/view handles may be cloned for
/// work authoring, but those clones do not retain the private physical surface image.
///
/// ```compile_fail
/// use runen_gpu::GpuAcquiredSurfaceImage;
/// fn require_clone<T: Clone>() {}
/// require_clone::<GpuAcquiredSurfaceImage>();
/// ```
pub struct GpuAcquiredSurfaceImage {
    surface: GpuSurfaceHandle,
    lease_id: GpuSurfaceLeaseId,
    status: GpuSurfaceAcquisitionStatus,
    texture: GpuTextureHandle,
    default_view: GpuTextureViewHandle,
    owner: Arc<GpuSurfaceLeaseOwner>,
}

impl GpuAcquiredSurfaceImage {
    pub(crate) fn new(
        surface: GpuSurfaceHandle,
        lease_id: GpuSurfaceLeaseId,
        status: GpuSurfaceAcquisitionStatus,
        texture: GpuTextureHandle,
        default_view: GpuTextureViewHandle,
        owner: Arc<GpuSurfaceLeaseOwner>,
    ) -> Self {
        Self {
            surface,
            lease_id,
            status,
            texture,
            default_view,
            owner,
        }
    }

    pub const fn surface_id(&self) -> GpuSurfaceId {
        self.surface.id()
    }

    pub const fn surface_generation(&self) -> GpuSurfaceGeneration {
        self.surface.generation()
    }

    pub const fn affinity(&self) -> GpuContextAffinity {
        self.surface.affinity()
    }

    pub const fn lease_id(&self) -> GpuSurfaceLeaseId {
        self.lease_id
    }

    pub const fn status(&self) -> GpuSurfaceAcquisitionStatus {
        self.status
    }

    pub fn texture(&self) -> &GpuTextureHandle {
        &self.texture
    }

    /// Returns the explicit single-mip/single-layer view for ordinary G5 render attachments.
    ///
    /// G5 keeps its explicit `GpuTextureViewHandle` attachment contract; G7A does not weaken that
    /// boundary merely because the parent texture came from a presentation surface.
    pub fn default_view(&self) -> &GpuTextureViewHandle {
        &self.default_view
    }

    /// Explicitly abandons this acquisition without claiming presentation.
    ///
    /// This is semantically equivalent to dropping the owner. Any logical texture/view clones
    /// remain ordinary stale logical references and do not retain the physical lease.
    pub fn abandon(self) {}

    pub(crate) fn owner_weak(&self) -> Weak<GpuSurfaceLeaseOwner> {
        Arc::downgrade(&self.owner)
    }
}

impl fmt::Debug for GpuAcquiredSurfaceImage {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GpuAcquiredSurfaceImage")
            .field("surface", &self.surface)
            .field("lease_id", &self.lease_id)
            .field("status", &self.status)
            .field("texture", &self.texture.diagnostic_identity())
            .field("default_view", &self.default_view.diagnostic_identity())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuContextId, GpuDeviceGeneration, allocate_surface_id};
    use std::sync::atomic::AtomicUsize;

    #[derive(Debug, Default)]
    struct RecordingLeaseReleaser {
        releases: AtomicUsize,
    }

    impl GpuSurfaceLeaseReleaser for RecordingLeaseReleaser {
        fn release(&self, _lease: &GpuSurfaceResourceLease) {
            self.releases.fetch_add(1, Ordering::SeqCst);
        }
    }

    fn test_lease() -> GpuSurfaceResourceLease {
        let context = GpuContextId::test_value(NonZeroU64::new(1).unwrap());
        let affinity = GpuContextAffinity::test_value(context, GpuDeviceGeneration::first());
        let surface = GpuSurfaceHandle::new(
            allocate_surface_id().unwrap(),
            affinity,
            GpuSurfaceGeneration::first(),
        );
        GpuSurfaceResourceLease::new(surface, allocate_surface_lease_id(surface.id()).unwrap())
    }

    fn test_owner(
        lease: &GpuSurfaceResourceLease,
        releaser: &Arc<RecordingLeaseReleaser>,
    ) -> (Arc<GpuSurfaceLeaseOwner>, Arc<dyn GpuSurfaceLeaseReleaser>) {
        let releaser_dyn: Arc<dyn GpuSurfaceLeaseReleaser> = releaser.clone();
        let owner = GpuSurfaceLeaseOwner::new(lease.clone(), Arc::downgrade(&releaser_dyn));
        (owner, releaser_dyn)
    }

    #[test]
    fn acquisition_categories_keep_timeout_and_occlusion_distinct() {
        assert_ne!(
            GpuSurfaceAcquireErrorCategory::Timeout,
            GpuSurfaceAcquireErrorCategory::Occluded
        );
    }

    #[test]
    fn lease_terminal_state_distinguishes_abandonment_from_presentation() {
        let abandoned = test_lease();
        let abandoned_clone = abandoned.clone();
        abandoned.mark_abandoned();
        assert_eq!(
            abandoned_clone.disposition(),
            GpuSurfaceLeaseDisposition::Abandoned
        );
        assert_eq!(
            abandoned_clone.mark_presented(),
            Err(GpuSurfaceLeaseDisposition::Abandoned)
        );

        let presented = test_lease();
        let presented_clone = presented.clone();
        assert_eq!(presented.mark_presented(), Ok(()));
        presented.mark_abandoned();
        assert_eq!(
            presented_clone.disposition(),
            GpuSurfaceLeaseDisposition::Presented
        );
        assert_eq!(
            presented_clone.mark_presented(),
            Err(GpuSurfaceLeaseDisposition::Presented)
        );
    }

    #[test]
    fn lease_owner_drop_abandons_and_releases_exactly_once() {
        let lease = test_lease();
        let releaser = Arc::new(RecordingLeaseReleaser::default());
        let (owner, _releaser_dyn) = test_owner(&lease, &releaser);

        drop(owner);

        assert_eq!(lease.disposition(), GpuSurfaceLeaseDisposition::Abandoned);
        assert_eq!(releaser.releases.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn lease_owner_drop_after_present_preserves_presented_state() {
        let lease = test_lease();
        let releaser = Arc::new(RecordingLeaseReleaser::default());
        let (owner, _releaser_dyn) = test_owner(&lease, &releaser);
        assert_eq!(lease.mark_presented(), Ok(()));

        drop(owner);

        assert_eq!(lease.disposition(), GpuSurfaceLeaseDisposition::Presented);
        assert_eq!(releaser.releases.load(Ordering::SeqCst), 1);
    }
}
