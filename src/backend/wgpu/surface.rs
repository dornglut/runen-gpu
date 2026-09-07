pub(crate) mod execution;

use super::adapter_mapping::known_formats;
use super::device_request::{enforce_runengpu_instance_flags, request_with_instance};
use super::{WgpuDeviceHealth, WgpuErrorAttributionGate};
use crate::{
    GpuAcquiredSurfaceImage, GpuCapabilityFeature, GpuCapabilityRequirement,
    GpuCapabilityRequirements, GpuContext, GpuContextAffinity, GpuContextDescriptor,
    GpuContextRequestError, GpuContextRequestErrorCategory, GpuExecutionPolicy,
    GpuRealizationPolicies, GpuResourceCommon, GpuResourceLabel, GpuResourceProvenance,
    GpuSurfaceAcquireError, GpuSurfaceAcquireErrorCategory, GpuSurfaceAcquisitionStatus,
    GpuSurfaceAlphaMode, GpuSurfaceCapabilities, GpuSurfaceConfiguration, GpuSurfaceError,
    GpuSurfaceErrorCategory, GpuSurfaceGeneration, GpuSurfaceHandle, GpuSurfaceId,
    GpuSurfaceLeaseOwner, GpuSurfaceLeaseReleaser, GpuSurfacePresentMode, GpuSurfaceResourceLease,
    GpuSurfaceTarget, GpuTextureAspect, GpuTextureDescriptor, GpuTextureDimension,
    GpuTextureExtent, GpuTextureFormat, GpuTextureHandle, GpuTextureInitialization,
    GpuTextureSubresourceRange, GpuTextureUsage, GpuTextureUsages, GpuTextureViewDescriptor,
    GpuTextureViewHandle, GpuWorkResourceId, GpuWorkResourceIdAllocator, allocate_surface_id,
    allocate_surface_lease_id,
};
use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, MutexGuard, Weak};
use wgpu::{
    Adapter, CompositeAlphaMode, CurrentSurfaceTexture, Device, Instance, InstanceDescriptor,
    PresentMode, Surface, SurfaceColorSpace, SurfaceConfiguration, SurfaceTexture, TextureFormat,
    TextureUsages,
};

#[cfg(not(target_arch = "wasm32"))]
struct WgpuSurfaceDisplay<T>(Arc<T>);

#[cfg(not(target_arch = "wasm32"))]
impl<T> core::fmt::Debug for WgpuSurfaceDisplay<T> {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("WgpuSurfaceDisplay")
            .finish_non_exhaustive()
    }
}

#[cfg(not(target_arch = "wasm32"))]
impl<T> raw_window_handle::HasDisplayHandle for WgpuSurfaceDisplay<T>
where
    T: GpuSurfaceTarget,
{
    fn display_handle(
        &self,
    ) -> Result<raw_window_handle::DisplayHandle<'_>, raw_window_handle::HandleError> {
        self.0.display_handle()
    }
}

struct WgpuSurfaceLease {
    lease: GpuSurfaceResourceLease,
    owner: Weak<GpuSurfaceLeaseOwner>,
    texture_identity: GpuWorkResourceId,
    view_identity: GpuWorkResourceId,
    texture: SurfaceTexture,
}

struct WgpuSurfaceRecord {
    generation: GpuSurfaceGeneration,
    capabilities: GpuSurfaceCapabilities,
    configuration: Option<GpuSurfaceConfiguration>,
    active_lease: Option<WgpuSurfaceLease>,
    surface: Surface<'static>,
}

#[derive(Default)]
struct WgpuSurfaceStateInner {
    records: BTreeMap<GpuSurfaceId, WgpuSurfaceRecord>,
    resource_ids: GpuWorkResourceIdAllocator,
}

struct WgpuSurfaceShared {
    inner: Mutex<WgpuSurfaceStateInner>,
}

impl WgpuSurfaceShared {
    fn new() -> Self {
        Self {
            inner: Mutex::new(WgpuSurfaceStateInner::default()),
        }
    }
}

impl core::fmt::Debug for WgpuSurfaceShared {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let surface_count = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .records
            .len();
        formatter
            .debug_struct("WgpuSurfaceShared")
            .field("surface_count", &surface_count)
            .finish()
    }
}

impl GpuSurfaceLeaseReleaser for WgpuSurfaceShared {
    fn release(&self, lease: &GpuSurfaceResourceLease) {
        let mut inner = self
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(record) = inner.records.get_mut(&lease.surface().id()) else {
            return;
        };
        if record.generation != lease.surface().generation() {
            return;
        }
        let matches = record
            .active_lease
            .as_ref()
            .is_some_and(|active| active.lease == *lease);
        if matches {
            record.active_lease.take();
        }
    }
}

pub(crate) struct WgpuSurfaceState {
    affinity: GpuContextAffinity,
    shared: Arc<WgpuSurfaceShared>,
}

impl WgpuSurfaceState {
    pub(crate) fn new(affinity: GpuContextAffinity) -> Self {
        Self {
            affinity,
            shared: Arc::new(WgpuSurfaceShared::new()),
        }
    }

    fn inner(
        &self,
        surface: Option<GpuSurfaceId>,
    ) -> Result<MutexGuard<'_, WgpuSurfaceStateInner>, GpuSurfaceError> {
        self.shared.inner.lock().map_err(|_| {
            GpuSurfaceError::new(
                GpuSurfaceErrorCategory::ContextOrDeviceUnavailableOrLost,
                surface,
                "surface registry was poisoned by an aborted backend operation",
            )
        })
    }

    fn inner_for_acquisition(
        &self,
        surface: GpuSurfaceId,
    ) -> Result<MutexGuard<'_, WgpuSurfaceStateInner>, GpuSurfaceAcquireError> {
        self.shared.inner.lock().map_err(|_| {
            GpuSurfaceAcquireError::new(
                GpuSurfaceAcquireErrorCategory::ContextOrDeviceUnavailableOrLost,
                Some(surface),
                "surface registry was poisoned by an aborted backend operation",
            )
        })
    }

    pub(crate) fn register_surface(
        &self,
        adapter: &Adapter,
        surface: Surface<'static>,
    ) -> Result<GpuSurfaceHandle, GpuSurfaceError> {
        if !adapter.is_surface_supported(&surface) {
            return Err(GpuSurfaceError::new(
                GpuSurfaceErrorCategory::AdapterIncompatible,
                None,
                "the admitted adapter is not compatible with this surface",
            ));
        }
        let capabilities = normalize_surface_capabilities(&surface.get_capabilities(adapter));
        if capabilities.formats().is_empty()
            || capabilities.present_modes().is_empty()
            || capabilities.alpha_modes().is_empty()
            || !capabilities.supports_usage(GpuTextureUsage::ColorAttachment)
        {
            return Err(GpuSurfaceError::new(
                GpuSurfaceErrorCategory::AdapterIncompatible,
                None,
                "the compatible backend surface has no complete normalized G7A presentation baseline",
            ));
        }

        let id = allocate_surface_id()?;
        let generation = GpuSurfaceGeneration::first();
        self.inner(Some(id))?.records.insert(
            id,
            WgpuSurfaceRecord {
                generation,
                capabilities,
                configuration: None,
                active_lease: None,
                surface,
            },
        );
        Ok(GpuSurfaceHandle::new(id, self.affinity, generation))
    }

    pub(crate) fn capabilities(
        &self,
        handle: GpuSurfaceHandle,
    ) -> Result<GpuSurfaceCapabilities, GpuSurfaceError> {
        let inner = self.inner(Some(handle.id()))?;
        let record = validate_handle(self.affinity, &inner.records, handle)?;
        Ok(record.capabilities.clone())
    }

    pub(crate) fn configuration(
        &self,
        handle: GpuSurfaceHandle,
    ) -> Result<Option<GpuSurfaceConfiguration>, GpuSurfaceError> {
        let inner = self.inner(Some(handle.id()))?;
        let record = validate_handle(self.affinity, &inner.records, handle)?;
        Ok(record.configuration.clone())
    }

    pub(crate) fn configure(
        &self,
        handle: GpuSurfaceHandle,
        configuration: GpuSurfaceConfiguration,
        device: &Device,
        health: &WgpuDeviceHealth,
        error_attribution_gate: &WgpuErrorAttributionGate,
    ) -> Result<GpuSurfaceHandle, GpuSurfaceError> {
        ensure_surface_health(health, Some(handle.id()))?;
        let _attribution_gate = error_attribution_gate.acquire();
        let mut inner = self.inner(Some(handle.id()))?;
        let record = validate_handle_mut(self.affinity, &mut inner.records, handle)?;
        validate_configuration(handle.id(), &record.capabilities, &configuration)?;

        let next_generation = surface_generation_after_configuration(
            record.generation,
            record.configuration.is_some(),
            handle.id(),
        )?;
        let native = lower_configuration(&configuration);

        // Reconfiguration is a lease boundary. Drop the exact physical acquired image before the
        // backend is reconfigured; caller-held logical handles never own this object and therefore
        // become stale without prolonging swapchain authority.
        if let Some(active) = record.active_lease.take() {
            active.lease.mark_abandoned();
        }
        record.surface.configure(device, &native);
        ensure_surface_health(health, Some(handle.id()))?;
        record.configuration = Some(configuration);
        record.generation = next_generation;
        Ok(GpuSurfaceHandle::new(
            handle.id(),
            self.affinity,
            next_generation,
        ))
    }

    pub(crate) fn acquire(
        &self,
        handle: GpuSurfaceHandle,
        health: &WgpuDeviceHealth,
        error_attribution_gate: &WgpuErrorAttributionGate,
    ) -> Result<GpuAcquiredSurfaceImage, GpuSurfaceAcquireError> {
        ensure_acquisition_health(health, handle.id())?;
        let _attribution_gate = error_attribution_gate.acquire();
        let mut inner = self.inner_for_acquisition(handle.id())?;

        let (configuration, physical, status) = {
            let record = validate_handle_mut(self.affinity, &mut inner.records, handle)
                .map_err(map_surface_error_to_acquisition)?;
            release_abandoned_lease(record);
            let configuration = record.configuration.clone().ok_or_else(|| {
                GpuSurfaceAcquireError::new(
                    GpuSurfaceAcquireErrorCategory::NotConfigured,
                    Some(handle.id()),
                    "configure the surface explicitly before acquiring an image",
                )
            })?;
            if record.active_lease.is_some() {
                return Err(GpuSurfaceAcquireError::new(
                    GpuSurfaceAcquireErrorCategory::AlreadyAcquired,
                    Some(handle.id()),
                    "the surface already has one active acquired-image lease",
                ));
            }

            let (physical, status) = match record.surface.get_current_texture() {
                CurrentSurfaceTexture::Success(texture) => {
                    (texture, GpuSurfaceAcquisitionStatus::Optimal)
                }
                CurrentSurfaceTexture::Suboptimal(texture) => {
                    (texture, GpuSurfaceAcquisitionStatus::Suboptimal)
                }
                CurrentSurfaceTexture::Timeout => {
                    return Err(GpuSurfaceAcquireError::new(
                        GpuSurfaceAcquireErrorCategory::Timeout,
                        Some(handle.id()),
                        "surface acquisition timed out",
                    ));
                }
                CurrentSurfaceTexture::Occluded => {
                    return Err(GpuSurfaceAcquireError::new(
                        GpuSurfaceAcquireErrorCategory::Occluded,
                        Some(handle.id()),
                        "surface acquisition reports an occluded presentation target",
                    ));
                }
                CurrentSurfaceTexture::Outdated => {
                    return Err(GpuSurfaceAcquireError::new(
                        GpuSurfaceAcquireErrorCategory::Outdated,
                        Some(handle.id()),
                        "surface acquisition reports an outdated configuration",
                    ));
                }
                CurrentSurfaceTexture::Lost => {
                    return Err(GpuSurfaceAcquireError::new(
                        GpuSurfaceAcquireErrorCategory::Lost,
                        Some(handle.id()),
                        "surface acquisition reports a lost surface",
                    ));
                }
                CurrentSurfaceTexture::Validation => {
                    return Err(GpuSurfaceAcquireError::new(
                        GpuSurfaceAcquireErrorCategory::Validation,
                        Some(handle.id()),
                        "surface acquisition failed backend validation",
                    ));
                }
            };
            (configuration, physical, status)
        };

        if let Some(fault) = health.terminal_fault() {
            drop(physical);
            return Err(GpuSurfaceAcquireError::new(
                GpuSurfaceAcquireErrorCategory::ContextOrDeviceUnavailableOrLost,
                Some(handle.id()),
                fault.detail,
            ));
        }

        let releaser: Arc<dyn GpuSurfaceLeaseReleaser> = self.shared.clone();
        let resources =
            build_acquired_surface_resources(handle, &configuration, &mut inner.resource_ids)?;
        let record = inner
            .records
            .get_mut(&handle.id())
            .expect("validated surface record remains present while acquisition authority is held");
        let owner = GpuSurfaceLeaseOwner::new(resources.lease.clone(), Arc::downgrade(&releaser));
        let image = GpuAcquiredSurfaceImage::new(
            handle,
            resources.lease.lease_id(),
            status,
            resources.texture,
            resources.default_view,
            owner,
        );
        record.active_lease = Some(WgpuSurfaceLease {
            lease: resources.lease,
            owner: image.owner_weak(),
            texture_identity: image.texture().diagnostic_identity(),
            view_identity: image.default_view().diagnostic_identity(),
            texture: physical,
        });
        Ok(image)
    }
}

impl core::fmt::Debug for WgpuSurfaceState {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let count = self
            .shared
            .inner
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .records
            .len();
        formatter
            .debug_struct("WgpuSurfaceState")
            .field("affinity", &self.affinity)
            .field("surface_count", &count)
            .finish()
    }
}

fn release_abandoned_lease(record: &mut WgpuSurfaceRecord) {
    let abandoned = record
        .active_lease
        .as_ref()
        .is_some_and(|lease| lease.owner.upgrade().is_none());
    if abandoned {
        if let Some(active) = record.active_lease.take() {
            active.lease.mark_abandoned();
        }
    }
}

struct AcquiredSurfaceResources {
    lease: GpuSurfaceResourceLease,
    texture: GpuTextureHandle,
    default_view: GpuTextureViewHandle,
}

fn build_acquired_surface_resources(
    surface: GpuSurfaceHandle,
    configuration: &GpuSurfaceConfiguration,
    resource_ids: &mut GpuWorkResourceIdAllocator,
) -> Result<AcquiredSurfaceResources, GpuSurfaceAcquireError> {
    let lease_id = allocate_surface_lease_id(surface.id())?;
    let surface_lease = GpuSurfaceResourceLease::new(surface, lease_id);
    let texture_label = GpuResourceLabel::new("surface acquired image")
        .map_err(|error| acquisition_invariant(surface.id(), error.to_string()))?;
    let provenance = GpuResourceProvenance::new(texture_label.clone(), None, None);
    let texture_common =
        GpuResourceCommon::surface_acquired(texture_label.clone(), provenance.clone());
    let extent = GpuTextureExtent::new(
        &texture_label,
        GpuTextureDimension::D2,
        configuration.width(),
        configuration.height(),
        1,
    )
    .map_err(|error| acquisition_invariant(surface.id(), error.to_string()))?;
    let usages = GpuTextureUsages::new(&texture_label, configuration.usages().iter().copied())
        .map_err(|error| acquisition_invariant(surface.id(), error.to_string()))?;
    let texture_descriptor = GpuTextureDescriptor::new(
        texture_common,
        GpuTextureDimension::D2,
        extent,
        1,
        1,
        configuration.format(),
        usages,
        GpuTextureInitialization::Uninitialized,
    )
    .map_err(|error| acquisition_invariant(surface.id(), error.to_string()))?;
    let texture = resource_ids
        .allocate_surface_texture_handle(texture_descriptor, surface_lease.clone())
        .map_err(|_| acquisition_identity_exhausted(surface.id()))?;

    let view_label = GpuResourceLabel::new("surface acquired default view")
        .map_err(|error| acquisition_invariant(surface.id(), error.to_string()))?;
    let view_common = GpuResourceCommon::surface_acquired(view_label.clone(), provenance);
    let subresources =
        GpuTextureSubresourceRange::new(&view_label, 0, 1, 0, 1, GpuTextureAspect::Color)
            .map_err(|error| acquisition_invariant(surface.id(), error.to_string()))?;
    let view_descriptor = GpuTextureViewDescriptor::new(
        view_common,
        &texture,
        None,
        GpuTextureDimension::D2,
        subresources,
    )
    .map_err(|error| acquisition_invariant(surface.id(), error.to_string()))?;
    let default_view = resource_ids
        .allocate_texture_view_handle(view_descriptor)
        .map_err(|_| acquisition_identity_exhausted(surface.id()))?;

    Ok(AcquiredSurfaceResources {
        lease: surface_lease,
        texture,
        default_view,
    })
}

fn acquisition_identity_exhausted(surface: GpuSurfaceId) -> GpuSurfaceAcquireError {
    GpuSurfaceAcquireError::new(
        GpuSurfaceAcquireErrorCategory::IdentityExhausted,
        Some(surface),
        "surface-acquired logical resource identity space is exhausted",
    )
}

fn acquisition_invariant(surface: GpuSurfaceId, detail: String) -> GpuSurfaceAcquireError {
    GpuSurfaceAcquireError::new(
        GpuSurfaceAcquireErrorCategory::InternalInvariant,
        Some(surface),
        detail,
    )
}

fn map_surface_error_to_acquisition(error: GpuSurfaceError) -> GpuSurfaceAcquireError {
    let category = match error.category() {
        GpuSurfaceErrorCategory::UnknownSurface => GpuSurfaceAcquireErrorCategory::UnknownSurface,
        GpuSurfaceErrorCategory::ForeignContext => GpuSurfaceAcquireErrorCategory::ForeignContext,
        GpuSurfaceErrorCategory::StaleGeneration => GpuSurfaceAcquireErrorCategory::StaleGeneration,
        GpuSurfaceErrorCategory::ContextOrDeviceUnavailableOrLost => {
            GpuSurfaceAcquireErrorCategory::ContextOrDeviceUnavailableOrLost
        }
        GpuSurfaceErrorCategory::IdentityExhausted
        | GpuSurfaceErrorCategory::GenerationExhausted => {
            GpuSurfaceAcquireErrorCategory::IdentityExhausted
        }
        _ => GpuSurfaceAcquireErrorCategory::InternalInvariant,
    };
    let surface = error.surface();
    let detail = error
        .detail()
        .unwrap_or("surface handle validation failed before acquisition")
        .to_string();
    GpuSurfaceAcquireError::new(category, surface, detail)
}

fn ensure_acquisition_health(
    health: &WgpuDeviceHealth,
    surface: GpuSurfaceId,
) -> Result<(), GpuSurfaceAcquireError> {
    if let Some(fault) = health.terminal_fault() {
        Err(GpuSurfaceAcquireError::new(
            GpuSurfaceAcquireErrorCategory::ContextOrDeviceUnavailableOrLost,
            Some(surface),
            fault.detail,
        ))
    } else {
        Ok(())
    }
}

fn validate_handle(
    expected: GpuContextAffinity,
    records: &BTreeMap<GpuSurfaceId, WgpuSurfaceRecord>,
    handle: GpuSurfaceHandle,
) -> Result<&WgpuSurfaceRecord, GpuSurfaceError> {
    validate_surface_affinity(expected, handle)?;
    let record = records.get(&handle.id()).ok_or_else(|| {
        GpuSurfaceError::new(
            GpuSurfaceErrorCategory::UnknownSurface,
            Some(handle.id()),
            "surface identity is absent from this context-local surface owner",
        )
    })?;
    validate_surface_generation(record.generation, handle)?;
    Ok(record)
}

fn validate_handle_mut(
    expected: GpuContextAffinity,
    records: &mut BTreeMap<GpuSurfaceId, WgpuSurfaceRecord>,
    handle: GpuSurfaceHandle,
) -> Result<&mut WgpuSurfaceRecord, GpuSurfaceError> {
    validate_surface_affinity(expected, handle)?;
    let record = records.get_mut(&handle.id()).ok_or_else(|| {
        GpuSurfaceError::new(
            GpuSurfaceErrorCategory::UnknownSurface,
            Some(handle.id()),
            "surface identity is absent from this context-local surface owner",
        )
    })?;
    validate_surface_generation(record.generation, handle)?;
    Ok(record)
}

fn validate_surface_generation(
    current: GpuSurfaceGeneration,
    handle: GpuSurfaceHandle,
) -> Result<(), GpuSurfaceError> {
    if current != handle.generation() {
        return Err(GpuSurfaceError::new(
            GpuSurfaceErrorCategory::StaleGeneration,
            Some(handle.id()),
            "surface reference names a stale configuration generation",
        ));
    }
    Ok(())
}

fn surface_generation_after_configuration(
    current: GpuSurfaceGeneration,
    already_configured: bool,
    surface: GpuSurfaceId,
) -> Result<GpuSurfaceGeneration, GpuSurfaceError> {
    if already_configured {
        current.next(surface)
    } else {
        Ok(current)
    }
}

fn validate_surface_affinity(
    expected: GpuContextAffinity,
    handle: GpuSurfaceHandle,
) -> Result<(), GpuSurfaceError> {
    if handle.affinity().context() != expected.context() {
        return Err(GpuSurfaceError::new(
            GpuSurfaceErrorCategory::ForeignContext,
            Some(handle.id()),
            "surface reference belongs to a different GPU context",
        ));
    }
    if handle.affinity().generation() != expected.generation() {
        return Err(GpuSurfaceError::new(
            GpuSurfaceErrorCategory::StaleGeneration,
            Some(handle.id()),
            "surface reference belongs to a stale device generation",
        ));
    }
    Ok(())
}

fn ensure_surface_health(
    health: &WgpuDeviceHealth,
    surface: Option<GpuSurfaceId>,
) -> Result<(), GpuSurfaceError> {
    if let Some(fault) = health.terminal_fault() {
        Err(GpuSurfaceError::new(
            GpuSurfaceErrorCategory::ContextOrDeviceUnavailableOrLost,
            surface,
            fault.detail,
        ))
    } else {
        Ok(())
    }
}

fn normalize_surface_capabilities(native: &wgpu::SurfaceCapabilities) -> GpuSurfaceCapabilities {
    let mut formats = Vec::new();
    for native_format in &native.formats {
        if let Some(format) = normalize_texture_format(*native_format) {
            if !formats.contains(&format) {
                formats.push(format);
            }
        }
    }

    let mut usages = Vec::new();
    for (native_usage, normalized) in [
        (
            TextureUsages::RENDER_ATTACHMENT,
            GpuTextureUsage::ColorAttachment,
        ),
        (TextureUsages::COPY_SRC, GpuTextureUsage::CopySource),
        (TextureUsages::COPY_DST, GpuTextureUsage::CopyDestination),
    ] {
        if native.usages.contains(native_usage) {
            usages.push(normalized);
        }
    }

    let mut present_modes = Vec::new();
    for native_mode in &native.present_modes {
        if let Some(mode) = normalize_present_mode(*native_mode) {
            if !present_modes.contains(&mode) {
                present_modes.push(mode);
            }
        }
    }

    let mut alpha_modes = Vec::new();
    for native_mode in &native.alpha_modes {
        if let Some(mode) = normalize_alpha_mode(*native_mode) {
            if !alpha_modes.contains(&mode) {
                alpha_modes.push(mode);
            }
        }
    }

    GpuSurfaceCapabilities::from_normalized_facts(formats, usages, present_modes, alpha_modes)
}

fn validate_configuration(
    surface: GpuSurfaceId,
    capabilities: &GpuSurfaceCapabilities,
    configuration: &GpuSurfaceConfiguration,
) -> Result<(), GpuSurfaceError> {
    if !capabilities.supports_format(configuration.format()) {
        return Err(GpuSurfaceError::new(
            GpuSurfaceErrorCategory::UnsupportedFormat,
            Some(surface),
            "configured surface format is absent from the normalized surface capability set",
        ));
    }
    if let Some(usage) = configuration
        .usages()
        .iter()
        .find(|usage| !capabilities.supports_usage(**usage))
    {
        return Err(GpuSurfaceError::new(
            GpuSurfaceErrorCategory::UnsupportedUsage,
            Some(surface),
            format!("configured surface usage is unsupported: {usage:?}"),
        ));
    }
    if !capabilities.supports_present_mode(configuration.present_mode()) {
        return Err(GpuSurfaceError::new(
            GpuSurfaceErrorCategory::UnsupportedPresentMode,
            Some(surface),
            "configured present mode is absent from the normalized surface capability set",
        ));
    }
    if !capabilities.supports_alpha_mode(configuration.alpha_mode()) {
        return Err(GpuSurfaceError::new(
            GpuSurfaceErrorCategory::UnsupportedAlphaMode,
            Some(surface),
            "configured alpha mode is absent from the normalized surface capability set",
        ));
    }
    if configuration
        .view_formats()
        .iter()
        .any(|format| map_texture_format(*format).is_none())
    {
        return Err(GpuSurfaceError::new(
            GpuSurfaceErrorCategory::UnsupportedViewFormat,
            Some(surface),
            "configured surface view format is absent from the normalized backend format vocabulary",
        ));
    }
    Ok(())
}

fn lower_configuration(configuration: &GpuSurfaceConfiguration) -> SurfaceConfiguration {
    let usage = configuration
        .usages()
        .iter()
        .fold(TextureUsages::empty(), |native, usage| {
            native
                | match usage {
                    GpuTextureUsage::ColorAttachment => TextureUsages::RENDER_ATTACHMENT,
                    GpuTextureUsage::CopySource => TextureUsages::COPY_SRC,
                    GpuTextureUsage::CopyDestination => TextureUsages::COPY_DST,
                    _ => TextureUsages::empty(),
                }
        });
    SurfaceConfiguration {
        usage,
        format: map_texture_format(configuration.format())
            .expect("surface configuration construction accepts only normalized backend formats"),
        color_space: SurfaceColorSpace::Auto,
        width: configuration.width(),
        height: configuration.height(),
        present_mode: map_present_mode(configuration.present_mode()),
        desired_maximum_frame_latency: configuration.desired_maximum_frame_latency(),
        alpha_mode: map_alpha_mode(configuration.alpha_mode()),
        view_formats: configuration
            .view_formats()
            .iter()
            .copied()
            .map(|format| {
                map_texture_format(format).expect(
                    "surface view format construction accepts only normalized backend formats",
                )
            })
            .collect(),
    }
}

fn normalize_texture_format(native: TextureFormat) -> Option<GpuTextureFormat> {
    known_formats()
        .into_iter()
        .find_map(|(normalized, candidate)| (candidate == native).then_some(normalized))
}

fn map_texture_format(format: GpuTextureFormat) -> Option<TextureFormat> {
    known_formats()
        .into_iter()
        .find_map(|(candidate, native)| (candidate == format).then_some(native))
}

const fn normalize_present_mode(native: PresentMode) -> Option<GpuSurfacePresentMode> {
    match native {
        PresentMode::Fifo => Some(GpuSurfacePresentMode::Fifo),
        PresentMode::FifoRelaxed => Some(GpuSurfacePresentMode::FifoRelaxed),
        PresentMode::Immediate => Some(GpuSurfacePresentMode::Immediate),
        PresentMode::Mailbox => Some(GpuSurfacePresentMode::Mailbox),
        PresentMode::AutoVsync | PresentMode::AutoNoVsync => None,
    }
}

const fn map_present_mode(mode: GpuSurfacePresentMode) -> PresentMode {
    match mode {
        GpuSurfacePresentMode::Fifo => PresentMode::Fifo,
        GpuSurfacePresentMode::FifoRelaxed => PresentMode::FifoRelaxed,
        GpuSurfacePresentMode::Immediate => PresentMode::Immediate,
        GpuSurfacePresentMode::Mailbox => PresentMode::Mailbox,
    }
}

const fn normalize_alpha_mode(native: CompositeAlphaMode) -> Option<GpuSurfaceAlphaMode> {
    match native {
        CompositeAlphaMode::Opaque => Some(GpuSurfaceAlphaMode::Opaque),
        CompositeAlphaMode::PreMultiplied => Some(GpuSurfaceAlphaMode::PreMultiplied),
        CompositeAlphaMode::PostMultiplied => Some(GpuSurfaceAlphaMode::PostMultiplied),
        CompositeAlphaMode::Inherit => Some(GpuSurfaceAlphaMode::Inherit),
        CompositeAlphaMode::Auto => None,
    }
}

const fn map_alpha_mode(mode: GpuSurfaceAlphaMode) -> CompositeAlphaMode {
    match mode {
        GpuSurfaceAlphaMode::Opaque => CompositeAlphaMode::Opaque,
        GpuSurfaceAlphaMode::PreMultiplied => CompositeAlphaMode::PreMultiplied,
        GpuSurfaceAlphaMode::PostMultiplied => CompositeAlphaMode::PostMultiplied,
        GpuSurfaceAlphaMode::Inherit => CompositeAlphaMode::Inherit,
    }
}

fn descriptor_with_required_presentation(
    descriptor: GpuContextDescriptor,
) -> Result<GpuContextDescriptor, GpuContextRequestError> {
    let label = descriptor.label().map(str::to_owned);
    let provenance = descriptor.provenance().map(str::to_owned);

    let mut requirements = GpuCapabilityRequirements::new();
    requirements
        .insert(GpuCapabilityRequirement::Required(
            GpuCapabilityFeature::Presentation,
        ))
        .map_err(|error| {
            GpuContextRequestError::new(
                GpuContextRequestErrorCategory::ContradictoryRequest,
                error.to_string(),
            )
        })?;
    let surface_constraint = GpuContextDescriptor::new(requirements);
    let mut merged = descriptor.merge(&surface_constraint)?;

    if let Some(label) = label {
        merged = merged.with_label(label);
    }
    if let Some(provenance) = provenance {
        merged = merged.with_provenance(provenance);
    }
    Ok(merged)
}

pub(crate) async fn request_for_surface<T>(
    descriptor: GpuContextDescriptor,
    realization_policies: GpuRealizationPolicies,
    execution_policy: GpuExecutionPolicy,
    target: T,
) -> Result<(GpuContext, GpuSurfaceHandle), GpuContextRequestError>
where
    T: GpuSurfaceTarget,
{
    let descriptor = descriptor_with_required_presentation(descriptor)?;
    let target = Arc::new(target);
    let instance = Instance::new(surface_instance_descriptor(&target));
    let surface = instance
        .create_surface(Arc::clone(&target))
        .map_err(|error| {
            GpuContextRequestError::new(
                GpuContextRequestErrorCategory::SurfaceCreationFailure,
                error.to_string(),
            )
        })?;
    let context = request_with_instance(
        instance,
        descriptor,
        Some(&surface),
        realization_policies,
        execution_policy,
    )
    .await?;
    let handle = context
        .backend
        .surfaces
        .register_surface(&context.backend.adapter, surface)
        .map_err(map_surface_registration_error)?;
    Ok((context, handle))
}

fn surface_instance_descriptor<T: GpuSurfaceTarget>(target: &Arc<T>) -> InstanceDescriptor {
    #[cfg(not(target_arch = "wasm32"))]
    {
        enforce_runengpu_instance_flags(InstanceDescriptor::new_with_display_handle_from_env(
            Box::new(WgpuSurfaceDisplay(Arc::clone(target))),
        ))
    }
    #[cfg(target_arch = "wasm32")]
    {
        let _ = target;
        enforce_runengpu_instance_flags(InstanceDescriptor::new_without_display_handle_from_env())
    }
}

fn map_surface_registration_error(error: GpuSurfaceError) -> GpuContextRequestError {
    let category = match error.category() {
        GpuSurfaceErrorCategory::IdentityExhausted => {
            GpuContextRequestErrorCategory::IdentityExhausted
        }
        GpuSurfaceErrorCategory::ContextOrDeviceUnavailableOrLost => {
            GpuContextRequestErrorCategory::BackendDeviceRequestFailure
        }
        _ => GpuContextRequestErrorCategory::SurfaceCompatibilityFailure,
    };
    GpuContextRequestError::new(category, error.to_string())
}

impl GpuContext {
    pub async fn request_for_surface<T>(
        descriptor: GpuContextDescriptor,
        target: T,
    ) -> Result<(Self, GpuSurfaceHandle), GpuContextRequestError>
    where
        T: GpuSurfaceTarget,
    {
        Self::request_for_surface_with_policies(
            descriptor,
            GpuRealizationPolicies::default(),
            GpuExecutionPolicy::default(),
            target,
        )
        .await
    }

    pub async fn request_for_surface_with_policies<T>(
        descriptor: GpuContextDescriptor,
        realization_policies: GpuRealizationPolicies,
        execution_policy: GpuExecutionPolicy,
        target: T,
    ) -> Result<(Self, GpuSurfaceHandle), GpuContextRequestError>
    where
        T: GpuSurfaceTarget,
    {
        request_for_surface(descriptor, realization_policies, execution_policy, target).await
    }

    pub fn attach_surface<T>(&self, target: T) -> Result<GpuSurfaceHandle, GpuSurfaceError>
    where
        T: GpuSurfaceTarget,
    {
        ensure_surface_health(&self.backend.health, None)?;
        let surface = self
            .backend
            .instance
            .create_surface(Arc::new(target))
            .map_err(|error| {
                GpuSurfaceError::new(
                    GpuSurfaceErrorCategory::BackendCreationFailure,
                    None,
                    error.to_string(),
                )
            })?;
        self.backend
            .surfaces
            .register_surface(&self.backend.adapter, surface)
    }

    pub fn surface_capabilities(
        &self,
        surface: GpuSurfaceHandle,
    ) -> Result<GpuSurfaceCapabilities, GpuSurfaceError> {
        ensure_surface_health(&self.backend.health, Some(surface.id()))?;
        self.backend.surfaces.capabilities(surface)
    }

    pub fn surface_configuration(
        &self,
        surface: GpuSurfaceHandle,
    ) -> Result<Option<GpuSurfaceConfiguration>, GpuSurfaceError> {
        ensure_surface_health(&self.backend.health, Some(surface.id()))?;
        self.backend.surfaces.configuration(surface)
    }

    pub fn configure_surface(
        &self,
        surface: GpuSurfaceHandle,
        configuration: GpuSurfaceConfiguration,
    ) -> Result<GpuSurfaceHandle, GpuSurfaceError> {
        self.backend.surfaces.configure(
            surface,
            configuration,
            &self.backend.device,
            &self.backend.health,
            &self.backend.error_attribution_gate,
        )
    }

    pub fn acquire_surface_image(
        &self,
        surface: GpuSurfaceHandle,
    ) -> Result<GpuAcquiredSurfaceImage, GpuSurfaceAcquireError> {
        self.backend.surfaces.acquire(
            surface,
            &self.backend.health,
            &self.backend.error_attribution_gate,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        GpuContextId, GpuDeviceGeneration, GpuPreferredFallback, GpuResourceOwnership,
        GpuSurfaceLeaseDisposition,
    };
    use std::num::NonZeroU64;

    #[derive(Debug)]
    struct TestLeaseReleaser;

    impl GpuSurfaceLeaseReleaser for TestLeaseReleaser {
        fn release(&self, _lease: &GpuSurfaceResourceLease) {}
    }

    fn capabilities() -> GpuSurfaceCapabilities {
        GpuSurfaceCapabilities::from_normalized_facts(
            vec![
                GpuTextureFormat::Bgra8Unorm,
                GpuTextureFormat::Bgra8UnormSrgb,
            ],
            vec![
                GpuTextureUsage::ColorAttachment,
                GpuTextureUsage::CopySource,
            ],
            vec![GpuSurfacePresentMode::Fifo],
            vec![GpuSurfaceAlphaMode::Opaque],
        )
    }

    fn configuration() -> GpuSurfaceConfiguration {
        GpuSurfaceConfiguration::new(
            640,
            480,
            GpuTextureFormat::Bgra8Unorm,
            [
                GpuTextureUsage::ColorAttachment,
                GpuTextureUsage::CopySource,
            ],
            GpuSurfacePresentMode::Fifo,
            GpuSurfaceAlphaMode::Opaque,
            2,
            [GpuTextureFormat::Bgra8UnormSrgb],
        )
        .unwrap()
    }

    fn test_surface_handle() -> GpuSurfaceHandle {
        let context = GpuContextId::test_value(NonZeroU64::new(1).unwrap());
        let affinity = GpuContextAffinity::test_value(context, GpuDeviceGeneration::first());
        GpuSurfaceHandle::new(
            allocate_surface_id().unwrap(),
            affinity,
            GpuSurfaceGeneration::first(),
        )
    }

    fn test_image(
        surface: GpuSurfaceHandle,
        resource_ids: &mut GpuWorkResourceIdAllocator,
    ) -> GpuAcquiredSurfaceImage {
        let releaser: Arc<dyn GpuSurfaceLeaseReleaser> = Arc::new(TestLeaseReleaser);
        let resources =
            build_acquired_surface_resources(surface, &configuration(), resource_ids).unwrap();
        let owner = GpuSurfaceLeaseOwner::new(resources.lease.clone(), Arc::downgrade(&releaser));
        GpuAcquiredSurfaceImage::new(
            surface,
            resources.lease.lease_id(),
            GpuSurfaceAcquisitionStatus::Optimal,
            resources.texture,
            resources.default_view,
            owner,
        )
    }

    #[test]
    fn capability_normalization_keeps_only_explicit_g7a_vocabulary() {
        let native = wgpu::SurfaceCapabilities {
            formats: vec![TextureFormat::Bgra8UnormSrgb, TextureFormat::Rgba16Float],
            format_capabilities: Vec::new(),
            present_modes: vec![PresentMode::AutoVsync, PresentMode::Fifo],
            alpha_modes: vec![CompositeAlphaMode::Auto, CompositeAlphaMode::Opaque],
            usages: TextureUsages::RENDER_ATTACHMENT
                | TextureUsages::COPY_SRC
                | TextureUsages::TEXTURE_BINDING,
        };
        let normalized = normalize_surface_capabilities(&native);
        assert_eq!(normalized.formats(), &[GpuTextureFormat::Bgra8UnormSrgb]);
        assert_eq!(
            normalized.usages(),
            &[
                GpuTextureUsage::ColorAttachment,
                GpuTextureUsage::CopySource
            ]
        );
        assert_eq!(normalized.present_modes(), &[GpuSurfacePresentMode::Fifo]);
        assert_eq!(normalized.alpha_modes(), &[GpuSurfaceAlphaMode::Opaque]);
    }

    #[test]
    fn configuration_admission_uses_surface_local_capability_evidence() {
        let surface = allocate_surface_id().unwrap();
        let accepted = configuration();
        assert!(validate_configuration(surface, &capabilities(), &accepted).is_ok());

        let unsupported = GpuSurfaceConfiguration::new(
            640,
            480,
            GpuTextureFormat::Bgra8Unorm,
            [
                GpuTextureUsage::ColorAttachment,
                GpuTextureUsage::CopyDestination,
            ],
            GpuSurfacePresentMode::Fifo,
            GpuSurfaceAlphaMode::Opaque,
            2,
            [],
        )
        .unwrap();
        assert!(matches!(
            validate_configuration(surface, &capabilities(), &unsupported),
            Err(error) if error.category() == GpuSurfaceErrorCategory::UnsupportedUsage
        ));
    }

    #[test]
    fn presentation_first_descriptor_contributes_required_presentation_and_keeps_diagnostics() {
        let descriptor = GpuContextDescriptor::new(GpuCapabilityRequirements::new())
            .with_label("surface-first")
            .with_provenance("g7a1-test");
        let descriptor = descriptor_with_required_presentation(descriptor).unwrap();
        assert_eq!(
            descriptor
                .requirements()
                .get(GpuCapabilityFeature::Presentation),
            Some(GpuCapabilityRequirement::Required(
                GpuCapabilityFeature::Presentation
            ))
        );
        assert_eq!(descriptor.label(), Some("surface-first"));
        assert_eq!(descriptor.provenance(), Some("g7a1-test"));

        let mut preferred = GpuCapabilityRequirements::new();
        preferred
            .insert(GpuCapabilityRequirement::Preferred {
                feature: GpuCapabilityFeature::Presentation,
                fallback: GpuPreferredFallback::ContinueWithoutFeature,
            })
            .unwrap();
        let descriptor =
            descriptor_with_required_presentation(GpuContextDescriptor::new(preferred)).unwrap();
        assert_eq!(
            descriptor
                .requirements()
                .get(GpuCapabilityFeature::Presentation),
            Some(GpuCapabilityRequirement::Required(
                GpuCapabilityFeature::Presentation
            ))
        );
    }

    #[test]
    fn presentation_first_descriptor_rejects_disabled_presentation_before_backend_action() {
        let mut requirements = GpuCapabilityRequirements::new();
        requirements
            .insert(GpuCapabilityRequirement::Disabled(
                GpuCapabilityFeature::Presentation,
            ))
            .unwrap();
        let error = descriptor_with_required_presentation(GpuContextDescriptor::new(requirements))
            .unwrap_err();
        assert_eq!(
            error.category(),
            GpuContextRequestErrorCategory::ContradictoryRequest
        );
    }

    #[test]
    fn surface_reconfiguration_advances_generation_and_rejects_the_previous_handle() {
        let surface = allocate_surface_id().unwrap();
        let context = GpuContextId::test_value(NonZeroU64::new(1).unwrap());
        let affinity = GpuContextAffinity::test_value(context, GpuDeviceGeneration::first());
        let first_generation = GpuSurfaceGeneration::first();
        let first_handle = GpuSurfaceHandle::new(surface, affinity, first_generation);

        assert_eq!(
            surface_generation_after_configuration(first_generation, false, surface).unwrap(),
            first_generation
        );
        let second_generation =
            surface_generation_after_configuration(first_generation, true, surface).unwrap();
        assert_ne!(second_generation, first_generation);
        assert!(matches!(
            validate_surface_generation(second_generation, first_handle),
            Err(error) if error.category() == GpuSurfaceErrorCategory::StaleGeneration
        ));

        let current_handle = GpuSurfaceHandle::new(surface, affinity, second_generation);
        assert!(validate_surface_generation(second_generation, current_handle).is_ok());
    }

    #[test]
    fn surface_handle_affinity_rejects_foreign_context_and_stale_device_generation() {
        let surface = allocate_surface_id().unwrap();
        let context_one = GpuContextId::test_value(NonZeroU64::new(1).unwrap());
        let context_two = GpuContextId::test_value(NonZeroU64::new(2).unwrap());
        let generation_one = GpuDeviceGeneration::first();
        let generation_two = GpuDeviceGeneration::test_value(NonZeroU64::new(2).unwrap());
        let expected = GpuContextAffinity::test_value(context_one, generation_one);

        let foreign = GpuSurfaceHandle::new(
            surface,
            GpuContextAffinity::test_value(context_two, generation_one),
            GpuSurfaceGeneration::first(),
        );
        assert!(matches!(
            validate_surface_affinity(expected, foreign),
            Err(error) if error.category() == GpuSurfaceErrorCategory::ForeignContext
        ));

        let stale = GpuSurfaceHandle::new(
            surface,
            GpuContextAffinity::test_value(context_one, generation_two),
            GpuSurfaceGeneration::first(),
        );
        assert!(matches!(
            validate_surface_affinity(expected, stale),
            Err(error) if error.category() == GpuSurfaceErrorCategory::StaleGeneration
        ));
    }

    #[test]
    fn acquired_surface_image_uses_transient_surface_owned_texture_and_explicit_view() {
        let surface = test_surface_handle();
        let mut resource_ids = GpuWorkResourceIdAllocator::new();
        let image = test_image(surface, &mut resource_ids);

        assert_eq!(
            image.texture().descriptor().common().ownership(),
            GpuResourceOwnership::SurfaceAcquired
        );
        assert_eq!(
            image.default_view().descriptor().common().ownership(),
            GpuResourceOwnership::SurfaceAcquired
        );
        assert_eq!(
            image
                .default_view()
                .descriptor()
                .texture()
                .diagnostic_identity(),
            image.texture().diagnostic_identity()
        );
        assert!(
            image
                .texture()
                .descriptor()
                .usages()
                .contains(GpuTextureUsage::ColorAttachment)
        );
        assert!(
            image
                .texture()
                .descriptor()
                .usages()
                .contains(GpuTextureUsage::CopySource)
        );
        let texture_lease = image
            .texture()
            .surface_lease()
            .expect("surface-acquired texture must retain lease provenance");
        assert_eq!(texture_lease.surface(), surface);
        assert_eq!(texture_lease.lease_id(), image.lease_id());
        assert_eq!(image.default_view().surface_lease(), Some(texture_lease));
    }

    #[test]
    fn logical_handle_clones_do_not_retain_acquired_surface_owner() {
        let surface = test_surface_handle();
        let mut resource_ids = GpuWorkResourceIdAllocator::new();
        let image = test_image(surface, &mut resource_ids);
        let owner = image.owner_weak();
        let texture = image.texture().clone();
        let view = image.default_view().clone();
        let lease = texture.surface_lease().unwrap();

        drop(image);

        assert!(owner.upgrade().is_none());
        assert_eq!(lease.disposition(), GpuSurfaceLeaseDisposition::Abandoned);
        assert_eq!(
            view.descriptor().texture().diagnostic_identity(),
            texture.diagnostic_identity()
        );
    }
}
