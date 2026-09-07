use super::{
    GpuBufferDescriptor, GpuHandleCause, GpuHandleError, GpuQuerySetDescriptor,
    GpuSamplerDescriptor, GpuSurfaceResourceLease, GpuTextureDescriptor, GpuTextureViewDescriptor,
    GpuWorkResourceId, GpuWorkResourceIdAllocationError, GpuWorkResourceIdAllocator,
};
use core::fmt;
use core::hash::{Hash, Hasher};
use std::sync::Arc;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum GpuResourceKind {
    Buffer,
    Texture,
    TextureView,
    Sampler,
    QuerySet,
}

#[derive(Debug)]
struct GpuLogicalLease {
    id: GpuWorkResourceId,
    kind: GpuResourceKind,
}

impl GpuLogicalLease {
    const fn new(id: GpuWorkResourceId, kind: GpuResourceKind) -> Self {
        Self { id, kind }
    }
}

macro_rules! typed_handle {
    ($name:ident, $descriptor:ty, $kind:ident $(, $surface_field:ident : $surface_type:ty)?) => {
        #[derive(Clone)]
        pub struct $name {
            lease: Arc<GpuLogicalLease>,
            descriptor: Arc<$descriptor>,
            $($surface_field: Option<$surface_type>,)?
        }

        impl $name {
            pub(crate) fn from_descriptor(id: GpuWorkResourceId, descriptor: $descriptor) -> Self {
                Self {
                    lease: Arc::new(GpuLogicalLease::new(id, GpuResourceKind::$kind)),
                    descriptor: Arc::new(descriptor),
                    $($surface_field: None,)?
                }
            }

            /// Returns process-local logical identity for diagnostics only.
            ///
            /// The value has no stable persistence, replay, wire, ABI, or cache
            /// representation guarantee.
            pub fn diagnostic_identity(&self) -> GpuWorkResourceId {
                self.lease.id
            }

            /// Borrows the same process-local diagnostic identity.
            pub fn diagnostic_identity_ref(&self) -> &GpuWorkResourceId {
                &self.lease.id
            }

            pub fn descriptor(&self) -> &$descriptor {
                &self.descriptor
            }

            pub(crate) fn retained_descriptor(&self) -> Arc<$descriptor> {
                Arc::clone(&self.descriptor)
            }

            fn kind(&self) -> GpuResourceKind {
                self.lease.kind
            }
        }

        impl fmt::Debug for $name {
            fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                let mut debug = f.debug_struct(stringify!($name));
                debug
                    .field("diagnostic_identity", &self.lease.id)
                    .field("descriptor", &self.descriptor);
                $(debug.field(stringify!($surface_field), &self.$surface_field);)?
                debug.finish()
            }
        }

        // Equality, ordering, and hashing identify one in-process logical
        // lease by kind and private ID. This deliberately distinguishes two
        // resource references for export relationships without granting the
        // diagnostic ID persistence, replay, wire, cache, or descriptor-
        // semantic authority.
        impl PartialEq for $name {
            fn eq(&self, other: &Self) -> bool {
                self.lease.id == other.lease.id && self.kind() == other.kind()
            }
        }

        impl Eq for $name {}

        impl PartialOrd for $name {
            fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
                Some(self.cmp(other))
            }
        }

        impl Ord for $name {
            fn cmp(&self, other: &Self) -> core::cmp::Ordering {
                self.lease.id.cmp(&other.lease.id)
            }
        }

        impl Hash for $name {
            fn hash<H: Hasher>(&self, state: &mut H) {
                self.lease.id.hash(state);
                core::mem::discriminant(&self.kind()).hash(state);
            }
        }
    };
}

typed_handle!(GpuBufferHandle, GpuBufferDescriptor, Buffer);
typed_handle!(
    GpuTextureHandle,
    GpuTextureDescriptor,
    Texture,
    surface_lease: GpuSurfaceResourceLease
);
typed_handle!(GpuTextureViewHandle, GpuTextureViewDescriptor, TextureView);
typed_handle!(GpuSamplerHandle, GpuSamplerDescriptor, Sampler);
typed_handle!(GpuQuerySetHandle, GpuQuerySetDescriptor, QuerySet);

impl GpuTextureHandle {
    pub(crate) fn from_surface_descriptor(
        id: GpuWorkResourceId,
        descriptor: GpuTextureDescriptor,
        surface_lease: GpuSurfaceResourceLease,
    ) -> Self {
        Self {
            lease: Arc::new(GpuLogicalLease::new(id, GpuResourceKind::Texture)),
            descriptor: Arc::new(descriptor),
            surface_lease: Some(surface_lease),
        }
    }

    pub(crate) fn surface_lease(&self) -> Option<GpuSurfaceResourceLease> {
        self.surface_lease.clone()
    }
}

impl GpuTextureViewHandle {
    pub(crate) fn surface_lease(&self) -> Option<GpuSurfaceResourceLease> {
        self.descriptor.texture().surface_lease()
    }
}

impl GpuWorkResourceIdAllocator {
    pub fn allocate_buffer_handle(
        &mut self,
        descriptor: GpuBufferDescriptor,
    ) -> Result<GpuBufferHandle, GpuWorkResourceIdAllocationError> {
        self.allocate()
            .map(|id| GpuBufferHandle::from_descriptor(id, descriptor))
    }

    pub fn allocate_texture_handle(
        &mut self,
        descriptor: GpuTextureDescriptor,
    ) -> Result<GpuTextureHandle, GpuWorkResourceIdAllocationError> {
        self.allocate()
            .map(|id| GpuTextureHandle::from_descriptor(id, descriptor))
    }

    pub(crate) fn allocate_surface_texture_handle(
        &mut self,
        descriptor: GpuTextureDescriptor,
        surface_lease: GpuSurfaceResourceLease,
    ) -> Result<GpuTextureHandle, GpuWorkResourceIdAllocationError> {
        self.allocate()
            .map(|id| GpuTextureHandle::from_surface_descriptor(id, descriptor, surface_lease))
    }

    pub fn allocate_texture_view_handle(
        &mut self,
        descriptor: GpuTextureViewDescriptor,
    ) -> Result<GpuTextureViewHandle, GpuWorkResourceIdAllocationError> {
        self.allocate()
            .map(|id| GpuTextureViewHandle::from_descriptor(id, descriptor))
    }

    pub fn allocate_sampler_handle(
        &mut self,
        descriptor: GpuSamplerDescriptor,
    ) -> Result<GpuSamplerHandle, GpuWorkResourceIdAllocationError> {
        self.allocate()
            .map(|id| GpuSamplerHandle::from_descriptor(id, descriptor))
    }

    pub fn allocate_query_set_handle(
        &mut self,
        descriptor: GpuQuerySetDescriptor,
    ) -> Result<GpuQuerySetHandle, GpuWorkResourceIdAllocationError> {
        self.allocate()
            .map(|id| GpuQuerySetHandle::from_descriptor(id, descriptor))
    }
}

/// A kind-preserving reference for export relationships and diagnostics.
///
/// Ordinary APIs should accept a kind-specific handle instead.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuResourceRef {
    Buffer(GpuBufferHandle),
    Texture(GpuTextureHandle),
    TextureView(GpuTextureViewHandle),
    Sampler(GpuSamplerHandle),
    QuerySet(GpuQuerySetHandle),
}

impl GpuResourceRef {
    pub fn diagnostic_identity(&self) -> GpuWorkResourceId {
        match self {
            Self::Buffer(handle) => handle.diagnostic_identity(),
            Self::Texture(handle) => handle.diagnostic_identity(),
            Self::TextureView(handle) => handle.diagnostic_identity(),
            Self::Sampler(handle) => handle.diagnostic_identity(),
            Self::QuerySet(handle) => handle.diagnostic_identity(),
        }
    }

    pub fn common(&self) -> &super::GpuResourceCommon {
        match self {
            Self::Buffer(handle) => handle.descriptor().common(),
            Self::Texture(handle) => handle.descriptor().common(),
            Self::TextureView(handle) => handle.descriptor().common(),
            Self::Sampler(handle) => handle.descriptor().common(),
            Self::QuerySet(handle) => handle.descriptor().common(),
        }
    }

    pub fn into_buffer(self, label: impl Into<String>) -> Result<GpuBufferHandle, GpuHandleError> {
        match self {
            Self::Buffer(handle) => Ok(handle),
            _ => Err(wrong_kind("obtain GPU buffer handle", label)),
        }
    }

    pub fn into_texture(
        self,
        label: impl Into<String>,
    ) -> Result<GpuTextureHandle, GpuHandleError> {
        match self {
            Self::Texture(handle) => Ok(handle),
            _ => Err(wrong_kind("obtain GPU texture handle", label)),
        }
    }

    pub fn into_texture_view(
        self,
        label: impl Into<String>,
    ) -> Result<GpuTextureViewHandle, GpuHandleError> {
        match self {
            Self::TextureView(handle) => Ok(handle),
            _ => Err(wrong_kind("obtain GPU texture-view handle", label)),
        }
    }

    pub fn into_sampler(
        self,
        label: impl Into<String>,
    ) -> Result<GpuSamplerHandle, GpuHandleError> {
        match self {
            Self::Sampler(handle) => Ok(handle),
            _ => Err(wrong_kind("obtain GPU sampler handle", label)),
        }
    }

    pub fn into_query_set(
        self,
        label: impl Into<String>,
    ) -> Result<GpuQuerySetHandle, GpuHandleError> {
        match self {
            Self::QuerySet(handle) => Ok(handle),
            _ => Err(wrong_kind("obtain GPU query-set handle", label)),
        }
    }
}

impl From<GpuBufferHandle> for GpuResourceRef {
    fn from(value: GpuBufferHandle) -> Self {
        Self::Buffer(value)
    }
}

impl From<GpuTextureHandle> for GpuResourceRef {
    fn from(value: GpuTextureHandle) -> Self {
        Self::Texture(value)
    }
}

impl From<GpuTextureViewHandle> for GpuResourceRef {
    fn from(value: GpuTextureViewHandle) -> Self {
        Self::TextureView(value)
    }
}

impl From<GpuSamplerHandle> for GpuResourceRef {
    fn from(value: GpuSamplerHandle) -> Self {
        Self::Sampler(value)
    }
}

impl From<GpuQuerySetHandle> for GpuResourceRef {
    fn from(value: GpuQuerySetHandle) -> Self {
        Self::QuerySet(value)
    }
}

fn wrong_kind(operation: &'static str, label: impl Into<String>) -> GpuHandleError {
    GpuHandleError::Invalid {
        operation,
        label: label.into(),
        cause: GpuHandleCause::WrongKind,
        correction: "retain and use the matching kind-specific GPU handle",
    }
}

/// Kind-typed handles cannot be interchanged.
///
/// ```compile_fail
/// use runen_gpu::{GpuBufferHandle, GpuTextureHandle};
/// fn reinterpret(texture: GpuTextureHandle) -> GpuBufferHandle { texture }
/// ```
///
/// Raw identity construction and destroy-by-ID are intentionally absent.
///
/// ```compile_fail
/// use runen_gpu::{GpuBufferHandle, GpuWorkResourceId};
/// fn construct(id: GpuWorkResourceId) -> GpuBufferHandle {
///     GpuBufferHandle::from_raw_id(id)
/// }
/// ```
///
/// ```compile_fail
/// use runen_gpu::{GpuBufferHandle, GpuWorkResourceId};
/// fn destroy(handle: GpuBufferHandle, id: GpuWorkResourceId) {
///     handle.destroy_by_id(id);
/// }
/// ```
///
/// Handles are cloneable but intentionally non-`Copy`:
///
/// ```
/// use runen_gpu::GpuBufferHandle;
/// fn retain(handle: &GpuBufferHandle) -> GpuBufferHandle { handle.clone() }
/// ```
///
/// ```compile_fail
/// use runen_gpu::GpuBufferHandle;
/// fn require_copy<T: Copy>() {}
/// require_copy::<GpuBufferHandle>();
/// ```
const _: () = ();

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        GpuBufferInitialization, GpuBufferUsage, GpuBufferUsages, GpuMemoryIntent,
        GpuReconstruction, GpuResourceCommon, GpuResourceLabel, GpuResourceLifetime,
        GpuResourceProvenance, GpuWorkResourceIdAllocator,
    };
    use std::num::NonZeroU64;

    fn buffer_descriptor(name: &str) -> GpuBufferDescriptor {
        let label = GpuResourceLabel::new(name).unwrap();
        let provenance = GpuResourceProvenance::new(label.clone(), None, None);
        let common = GpuResourceCommon::owned(
            label.clone(),
            GpuResourceLifetime::Transient,
            GpuMemoryIntent::Device,
            GpuReconstruction::SourceBacked,
            provenance,
        )
        .unwrap();
        let usages = GpuBufferUsages::new(&label, [GpuBufferUsage::Storage]).unwrap();
        GpuBufferDescriptor::new(common, 16, usages, GpuBufferInitialization::Uninitialized)
            .unwrap()
    }

    #[test]
    fn scope_free_allocator_constructs_typed_handles() {
        let mut allocator = GpuWorkResourceIdAllocator::new();
        let handle = allocator
            .allocate_buffer_handle(buffer_descriptor("scope-free buffer"))
            .expect("scope-free allocator should allocate a typed buffer handle");

        let (owner_scope, local) = handle.diagnostic_identity().diagnostic_parts();
        assert_ne!(owner_scope, 0);
        assert_eq!(local, 1);
    }

    #[test]
    fn handle_clone_preserves_identity_and_lease() {
        let mut allocator =
            GpuWorkResourceIdAllocator::for_owner_scope(NonZeroU64::new(9).unwrap());
        let handle = allocator
            .allocate_buffer_handle(buffer_descriptor("buffer"))
            .unwrap();
        let clone = handle.clone();
        assert_eq!(clone, handle);
        assert_eq!(clone.diagnostic_identity(), handle.diagnostic_identity());
        assert!(Arc::ptr_eq(&clone.lease, &handle.lease));
    }

    #[test]
    fn generic_reference_reports_wrong_kind_structurally() {
        let mut allocator =
            GpuWorkResourceIdAllocator::for_owner_scope(NonZeroU64::new(9).unwrap());
        let buffer = allocator
            .allocate_buffer_handle(buffer_descriptor("buffer"))
            .unwrap();
        let error = GpuResourceRef::Buffer(buffer)
            .into_texture("render output")
            .unwrap_err();
        assert!(error.to_string().contains("WrongKind"));
        assert!(error.to_string().contains("render output"));
        assert!(error.to_string().contains("correction"));
    }

    #[test]
    fn handles_are_clone_but_not_declared_copy() {
        let source = std::fs::read_to_string(file!()).expect("handle source must be readable");
        let compact = source
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect::<String>();
        for name in [
            "GpuBufferHandle",
            "GpuTextureHandle",
            "GpuTextureViewHandle",
            "GpuSamplerHandle",
            "GpuQuerySetHandle",
        ] {
            let declaration = format!("typed_handle!({name},");
            assert!(compact.contains(&declaration));
        }
        assert!(!source.contains(&["impl Copy", " for Gpu"].concat()));
        assert!(!source.contains(&["pub fn destroy", "_by_id("].concat()));
        assert!(!source.contains(&["pub fn from", "_raw_id("].concat()));
    }
}
