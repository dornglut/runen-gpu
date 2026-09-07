use super::super::surface::execution::{WgpuSurfaceLeaseGuard, WgpuSurfaceLeaseResource};
use crate::{
    GpuContext, GpuRealizedTexture, GpuRealizedTextureView, GpuSubmissionFailure,
    GpuSubmissionFailureKind, GpuSubmissionPreparationError, GpuSubmissionPreparationErrorKind,
    GpuSurfaceResourceLease, GpuTextureAccessResource, GpuTextureHandle, GpuTextureViewHandle,
    GpuWorkResourceId,
};
use std::collections::BTreeMap;
use wgpu::{Texture, TextureView};

#[derive(Debug, Clone)]
pub(super) struct PreparedSurfaceUse {
    lease: GpuSurfaceResourceLease,
    resource: WgpuSurfaceLeaseResource,
    identity: GpuWorkResourceId,
}

impl PreparedSurfaceUse {
    fn new(
        lease: GpuSurfaceResourceLease,
        resource: WgpuSurfaceLeaseResource,
        identity: GpuWorkResourceId,
    ) -> Self {
        Self {
            lease,
            resource,
            identity,
        }
    }

    pub(super) fn lease(&self) -> &GpuSurfaceResourceLease {
        &self.lease
    }

    pub(super) const fn resource(&self) -> WgpuSurfaceLeaseResource {
        self.resource
    }

    pub(super) const fn identity(&self) -> GpuWorkResourceId {
        self.identity
    }
}

#[derive(Debug, Clone)]
pub(super) enum PreparedTexture {
    Realized(GpuRealizedTexture),
    Surface(PreparedSurfaceUse),
}

impl PreparedTexture {
    pub(super) fn surface_use(&self) -> Option<&PreparedSurfaceUse> {
        match self {
            Self::Realized(_) => None,
            Self::Surface(surface) => Some(surface),
        }
    }

    pub(super) fn resolve<'a>(
        &'a self,
        surface_guard: Option<&'a WgpuSurfaceLeaseGuard<'_>>,
    ) -> Result<&'a Texture, GpuSubmissionFailure> {
        match self {
            Self::Realized(realized) => Ok(&realized.record.object),
            Self::Surface(surface) => surface_guard
                .ok_or_else(missing_surface_guard)?
                .texture(surface.lease(), surface.identity())
                .map_err(GpuSubmissionFailure::from_surface_lease),
        }
    }
}

#[derive(Debug, Clone)]
pub(super) enum PreparedTextureView {
    Realized(GpuRealizedTextureView),
    Surface(PreparedSurfaceUse),
}

pub(super) enum ResolvedTextureView<'a> {
    Borrowed(&'a TextureView),
    Owned(TextureView),
}

impl ResolvedTextureView<'_> {
    pub(super) fn as_ref(&self) -> &TextureView {
        match self {
            Self::Borrowed(view) => view,
            Self::Owned(view) => view,
        }
    }
}

impl PreparedTextureView {
    pub(super) fn surface_use(&self) -> Option<&PreparedSurfaceUse> {
        match self {
            Self::Realized(_) => None,
            Self::Surface(surface) => Some(surface),
        }
    }

    pub(super) fn resolve<'a>(
        &'a self,
        surface_guard: Option<&'a WgpuSurfaceLeaseGuard<'_>>,
    ) -> Result<ResolvedTextureView<'a>, GpuSubmissionFailure> {
        match self {
            Self::Realized(realized) => Ok(ResolvedTextureView::Borrowed(&realized.record.object)),
            Self::Surface(surface) => surface_guard
                .ok_or_else(missing_surface_guard)?
                .create_default_view(surface.lease(), surface.identity())
                .map(ResolvedTextureView::Owned)
                .map_err(GpuSubmissionFailure::from_surface_lease),
        }
    }
}

pub(super) fn prepare_texture(
    context: &GpuContext,
    cache: &mut BTreeMap<GpuWorkResourceId, PreparedTexture>,
    handle: &GpuTextureHandle,
) -> Result<PreparedTexture, GpuSubmissionPreparationError> {
    let identity = handle.diagnostic_identity();
    if let Some(prepared) = cache.get(&identity) {
        return Ok(prepared.clone());
    }

    let prepared = if let Some(lease) = handle.surface_lease() {
        let resource = WgpuSurfaceLeaseResource::Texture(identity);
        context
            .backend
            .surfaces
            .validate_execution_lease(&lease, resource, &context.backend.health)
            .map_err(GpuSubmissionPreparationError::from_surface_lease)?;
        PreparedTexture::Surface(PreparedSurfaceUse::new(lease, resource, identity))
    } else {
        let realized = context.realize_texture(handle).map_err(|error| {
            GpuSubmissionPreparationError::new(
                GpuSubmissionPreparationErrorKind::ResourceRealizationFailed,
                error.to_string(),
            )
        })?;
        PreparedTexture::Realized(realized)
    };
    cache.insert(identity, prepared.clone());
    Ok(prepared)
}

pub(super) fn prepare_texture_view(
    context: &GpuContext,
    texture_cache: &mut BTreeMap<GpuWorkResourceId, PreparedTexture>,
    view_cache: &mut BTreeMap<GpuWorkResourceId, PreparedTextureView>,
    handle: &GpuTextureViewHandle,
) -> Result<PreparedTextureView, GpuSubmissionPreparationError> {
    let identity = handle.diagnostic_identity();
    if let Some(prepared) = view_cache.get(&identity) {
        return Ok(prepared.clone());
    }

    let prepared = if let Some(lease) = handle.surface_lease() {
        let resource = WgpuSurfaceLeaseResource::TextureView(identity);
        context
            .backend
            .surfaces
            .validate_execution_lease(&lease, resource, &context.backend.health)
            .map_err(GpuSubmissionPreparationError::from_surface_lease)?;
        PreparedTextureView::Surface(PreparedSurfaceUse::new(lease, resource, identity))
    } else {
        let parent = prepare_texture(context, texture_cache, handle.descriptor().texture())?;
        let PreparedTexture::Realized(parent) = parent else {
            return Err(GpuSubmissionPreparationError::new(
                GpuSubmissionPreparationErrorKind::InternalInvariant,
                "ordinary texture view unexpectedly resolved to a surface acquisition lease",
            ));
        };
        let realized = context
            .realize_texture_view(handle, &parent)
            .map_err(|error| {
                GpuSubmissionPreparationError::new(
                    GpuSubmissionPreparationErrorKind::ResourceRealizationFailed,
                    error.to_string(),
                )
            })?;
        PreparedTextureView::Realized(realized)
    };
    view_cache.insert(identity, prepared.clone());
    Ok(prepared)
}

pub(super) fn prepare_present_source(
    context: &GpuContext,
    source: &GpuTextureAccessResource,
) -> Result<PreparedSurfaceUse, GpuSubmissionPreparationError> {
    let (lease, resource, identity) = match source {
        GpuTextureAccessResource::Texture(texture) => {
            let identity = texture.diagnostic_identity();
            (
                texture.surface_lease(),
                WgpuSurfaceLeaseResource::Texture(identity),
                identity,
            )
        }
        GpuTextureAccessResource::TextureView(view) => {
            let identity = view.diagnostic_identity();
            (
                view.surface_lease(),
                WgpuSurfaceLeaseResource::TextureView(identity),
                identity,
            )
        }
    };
    let lease = lease.ok_or_else(|| {
        GpuSubmissionPreparationError::new(
            GpuSubmissionPreparationErrorKind::UnsupportedOperation,
            "GpuPresentOperation requires an active SurfaceAcquired texture or its explicit acquired default view",
        )
    })?;
    context
        .backend
        .surfaces
        .validate_execution_lease(&lease, resource, &context.backend.health)
        .map_err(GpuSubmissionPreparationError::from_surface_lease)?;
    Ok(PreparedSurfaceUse::new(lease, resource, identity))
}

fn missing_surface_guard() -> GpuSubmissionFailure {
    GpuSubmissionFailure::new(
        GpuSubmissionFailureKind::InternalInvariant,
        "surface-acquired execution resource reached encoding without the validated G7 lease guard",
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        GpuContextAffinity, GpuContextId, GpuDeviceGeneration, GpuSubmissionPreparationErrorKind,
        GpuSurfaceGeneration, GpuSurfaceHandle, GpuSurfaceLeaseErrorCategory,
        GpuSurfaceResourceLease, GpuWorkResourceIdAllocator, allocate_surface_id,
        allocate_surface_lease_id,
    };
    use std::collections::BTreeSet;
    use std::num::NonZeroU64;

    fn surface_use() -> PreparedSurfaceUse {
        let context = GpuContextId::test_value(NonZeroU64::new(1).unwrap());
        let affinity = GpuContextAffinity::test_value(context, GpuDeviceGeneration::first());
        let surface = GpuSurfaceHandle::new(
            allocate_surface_id().unwrap(),
            affinity,
            GpuSurfaceGeneration::first(),
        );
        let lease =
            GpuSurfaceResourceLease::new(surface, allocate_surface_lease_id(surface.id()).unwrap());
        let identity = GpuWorkResourceIdAllocator::new().allocate().unwrap();
        PreparedSurfaceUse::new(lease, WgpuSurfaceLeaseResource::Texture(identity), identity)
    }

    #[test]
    fn present_is_terminal_for_later_prepared_uses_of_the_same_surface_lease() {
        let surface = surface_use();
        let mut uses = Vec::new();
        let mut presented = BTreeSet::new();

        super::super::append_surface_use(&mut uses, &presented, &surface).unwrap();
        presented.insert(surface.lease().lease_id());

        let error = super::super::append_surface_use(&mut uses, &presented, &surface).unwrap_err();
        assert_eq!(
            error.kind(),
            GpuSubmissionPreparationErrorKind::SurfaceLease
        );
        assert_eq!(
            error.surface_error().map(|error| error.category()),
            Some(GpuSurfaceLeaseErrorCategory::AlreadyConsumed)
        );
        assert_eq!(uses.len(), 1);
    }
}
