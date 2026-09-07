//! Context/device-generation-bound private WGPU resource realization.

mod lowering;
mod records;
mod registry;

pub(crate) use records::{
    BufferRealizationRecord, QuerySetRealizationRecord, SamplerRealizationRecord,
    TextureRealizationRecord, TextureViewRealizationRecord,
};

use super::{WgpuContextState, WgpuDeviceHealth, WgpuErrorAttributionGate};
use crate::{
    GpuBufferHandle, GpuContext, GpuContextAffinity, GpuContextAffinityError, GpuQuerySetHandle,
    GpuRealizedBuffer, GpuRealizedQuerySet, GpuRealizedSampler, GpuRealizedTexture,
    GpuRealizedTextureView, GpuResourceRealizationError, GpuResourceRealizationErrorCategory,
    GpuResourceRealizationPolicy, GpuResourceRealizationStats, GpuSamplerHandle, GpuTextureAspect,
    GpuTextureHandle, GpuTextureViewHandle, GpuWorkResourceId,
};
use registry::{ResourceKind, ResourceRegistries};
use std::sync::{Arc, Mutex, MutexGuard};
use wgpu::{
    BufferDescriptor, QuerySetDescriptor, SamplerDescriptor, TextureDescriptor,
    TextureViewDescriptor,
};

pub(super) const fn map_texture_aspect(aspect: GpuTextureAspect) -> wgpu::TextureAspect {
    lowering::map_texture_aspect(aspect)
}

/// The sole authoritative owner of G4C1 lookup state for one context/device generation.
pub(crate) struct ResourceRealizationState {
    affinity: GpuContextAffinity,
    policy: GpuResourceRealizationPolicy,
    registries: Mutex<ResourceRegistries>,
    health: Arc<WgpuDeviceHealth>,
    error_attribution_gate: Arc<WgpuErrorAttributionGate>,
}

impl ResourceRealizationState {
    pub(crate) fn new(
        affinity: GpuContextAffinity,
        policy: GpuResourceRealizationPolicy,
        health: Arc<WgpuDeviceHealth>,
        error_attribution_gate: Arc<WgpuErrorAttributionGate>,
    ) -> Self {
        Self {
            affinity,
            policy,
            registries: Mutex::new(ResourceRegistries::default()),
            health,
            error_attribution_gate,
        }
    }

    pub(crate) const fn policy(&self) -> GpuResourceRealizationPolicy {
        self.policy
    }

    pub(crate) fn stats(&self) -> GpuResourceRealizationStats {
        self.registries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .stats(self.policy)
    }

    fn ensure_available(
        &self,
        resource: GpuWorkResourceId,
    ) -> Result<(), GpuResourceRealizationError> {
        self.health.ensure_resource(resource)
    }

    fn registries(
        &self,
        resource: GpuWorkResourceId,
    ) -> Result<MutexGuard<'_, ResourceRegistries>, GpuResourceRealizationError> {
        self.registries.lock().map_err(|_| {
            GpuResourceRealizationError::new(
                GpuResourceRealizationErrorCategory::ContextOrDeviceUnavailableOrLost,
                Some(resource),
                "resource-realization authority was poisoned by an aborted backend operation",
            )
        })
    }

    /// Acquires the shared WGPU attribution gate before this G4C1 owner takes its registry
    /// mutex. The execution bridge takes those locks in the same order while a raw operation loan
    /// is live, so reversing them would permit a cross-thread gate/registry deadlock.
    fn attributed_registries(
        &self,
        resource: GpuWorkResourceId,
    ) -> Result<(MutexGuard<'_, ()>, MutexGuard<'_, ResourceRegistries>), GpuResourceRealizationError>
    {
        self.ensure_available(resource)?;
        let gate = self.error_attribution_gate.acquire();
        let registries = self.registries(resource)?;
        self.ensure_available(resource)?;
        Ok((gate, registries))
    }

    fn create_backend_object<Object>(
        &self,
        resource: GpuWorkResourceId,
        _attribution_gate: &MutexGuard<'_, ()>,
        create: impl FnOnce() -> Object,
    ) -> Result<Object, GpuResourceRealizationError> {
        self.ensure_available(resource)?;
        // A G4C1 creation can synchronously trigger an uncaptured WGPU callback. The caller
        // retains the one shared gate from creation through authoritative publication.
        let created = create();
        self.ensure_available(resource)?;
        // A synchronously delivered WGPU callback is observed before publication. WGPU may also
        // deliver a backend fault later; in that case the context becomes unavailable and every
        // later realization/execution-bridge access rejects it rather than attributing the fault
        // to another constructor call.
        self.ensure_available(resource)?;
        Ok(created)
    }
}

impl core::fmt::Debug for ResourceRealizationState {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("ResourceRealizationState")
            .field("affinity", &self.affinity)
            .field("stats", &self.stats())
            .finish_non_exhaustive()
    }
}

impl GpuContext {
    /// Returns the operational record policy chosen when this context was requested.
    pub fn resource_realization_policy(&self) -> GpuResourceRealizationPolicy {
        self.backend.resource_realization.policy()
    }

    /// Observes authoritative lookup counts. Per-kind counts are observations, not quotas.
    pub fn resource_realization_stats(&self) -> GpuResourceRealizationStats {
        self.backend.resource_realization.stats()
    }

    pub fn realize_buffer(
        &self,
        handle: &GpuBufferHandle,
    ) -> Result<GpuRealizedBuffer, GpuResourceRealizationError> {
        let identity = handle.diagnostic_identity();
        lowering::validate_resource_ownership(identity, handle.descriptor().common())?;
        self.backend
            .resource_realization
            .ensure_available(identity)?;
        let native_usage = lowering::lower_buffer(self, identity, handle.descriptor())?;
        let descriptor = handle.retained_descriptor();
        {
            let registries = self.backend.resource_realization.registries(identity)?;
            if let Some(record) = registries.buffers.lookup(identity, &descriptor)? {
                return Ok(GpuRealizedBuffer::from_record(record));
            }
            registries.reject_other_kind(ResourceKind::Buffer, identity)?;
        }
        let (attribution_gate, mut registries) = self
            .backend
            .resource_realization
            .attributed_registries(identity)?;
        if let Some(record) = registries.buffers.lookup(identity, &descriptor)? {
            return Ok(GpuRealizedBuffer::from_record(record));
        }
        registries.reject_other_kind(ResourceKind::Buffer, identity)?;
        registries.ensure_capacity(identity, self.backend.resource_realization.policy)?;

        let object = self.backend.resource_realization.create_backend_object(
            identity,
            &attribution_gate,
            || {
                self.backend.device.create_buffer(&BufferDescriptor {
                    label: Some(descriptor.common().label().as_str()),
                    size: descriptor.size_bytes(),
                    usage: native_usage,
                    mapped_at_creation: false,
                })
            },
        )?;
        let record = Arc::new(BufferRealizationRecord::new(
            self.affinity(),
            identity,
            descriptor,
            object,
        ));
        registries.buffers.insert(identity, Arc::clone(&record));
        Ok(GpuRealizedBuffer::from_record(record))
    }

    pub fn realize_texture(
        &self,
        handle: &GpuTextureHandle,
    ) -> Result<GpuRealizedTexture, GpuResourceRealizationError> {
        let identity = handle.diagnostic_identity();
        lowering::validate_resource_ownership(identity, handle.descriptor().common())?;
        self.backend
            .resource_realization
            .ensure_available(identity)?;
        let lowered = lowering::lower_texture(self, identity, handle.descriptor())?;
        let descriptor = handle.retained_descriptor();
        {
            let registries = self.backend.resource_realization.registries(identity)?;
            if let Some(record) = registries.textures.lookup(identity, &descriptor)? {
                return Ok(GpuRealizedTexture::from_record(record));
            }
            registries.reject_other_kind(ResourceKind::Texture, identity)?;
        }
        let (attribution_gate, mut registries) = self
            .backend
            .resource_realization
            .attributed_registries(identity)?;
        if let Some(record) = registries.textures.lookup(identity, &descriptor)? {
            return Ok(GpuRealizedTexture::from_record(record));
        }
        registries.reject_other_kind(ResourceKind::Texture, identity)?;
        registries.ensure_capacity(identity, self.backend.resource_realization.policy)?;

        let paired = lowered.paired_view_format.into_iter().collect::<Vec<_>>();
        let object = self.backend.resource_realization.create_backend_object(
            identity,
            &attribution_gate,
            || {
                self.backend.device.create_texture(&TextureDescriptor {
                    label: Some(descriptor.common().label().as_str()),
                    size: lowered.size,
                    mip_level_count: descriptor.mip_level_count(),
                    sample_count: descriptor.sample_count(),
                    dimension: lowered.dimension,
                    format: lowered.format,
                    usage: lowered.usage,
                    view_formats: &paired,
                })
            },
        )?;
        let record = Arc::new(TextureRealizationRecord::new(
            self.affinity(),
            identity,
            descriptor,
            object,
            lowered.permits_format_reinterpretation,
        ));
        registries.textures.insert(identity, Arc::clone(&record));
        Ok(GpuRealizedTexture::from_record(record))
    }

    /// Realizes a view only within the exact already-realized parent texture.
    pub fn realize_texture_view(
        &self,
        handle: &GpuTextureViewHandle,
        parent: &GpuRealizedTexture,
    ) -> Result<GpuRealizedTextureView, GpuResourceRealizationError> {
        let identity = handle.diagnostic_identity();
        lowering::validate_resource_ownership(identity, handle.descriptor().common())?;
        validate_realized_input_affinity(self.affinity(), identity, parent.affinity())?;
        self.backend
            .resource_realization
            .ensure_available(identity)?;
        lowering::validate_texture_view(self, identity, handle.descriptor(), &parent.record)?;
        let descriptor = handle.retained_descriptor();
        {
            let registries = self.backend.resource_realization.registries(identity)?;
            let authoritative_parent = registries
                .textures
                .lookup(parent.logical_identity(), parent.descriptor())?
                .ok_or_else(|| {
                    GpuResourceRealizationError::new(
                        GpuResourceRealizationErrorCategory::UnknownLogicalResource,
                        Some(parent.logical_identity()),
                        "the parent texture is absent from this context's authoritative registry",
                    )
                })?;
            if !Arc::ptr_eq(&authoritative_parent, &parent.record) {
                return Err(GpuResourceRealizationError::new(
                    GpuResourceRealizationErrorCategory::UnknownLogicalResource,
                    Some(parent.logical_identity()),
                    "the supplied parent is not the authoritative texture realization record",
                ));
            }
            if let Some(record) = registries.texture_views.lookup(identity, &descriptor)? {
                return Ok(GpuRealizedTextureView::from_record(record));
            }
            registries.reject_other_kind(ResourceKind::TextureView, identity)?;
        }
        let (attribution_gate, mut registries) = self
            .backend
            .resource_realization
            .attributed_registries(identity)?;

        let authoritative_parent = registries
            .textures
            .lookup(parent.logical_identity(), parent.descriptor())?
            .ok_or_else(|| {
                GpuResourceRealizationError::new(
                    GpuResourceRealizationErrorCategory::UnknownLogicalResource,
                    Some(parent.logical_identity()),
                    "the parent texture is absent from this context's authoritative registry",
                )
            })?;
        if !Arc::ptr_eq(&authoritative_parent, &parent.record) {
            return Err(GpuResourceRealizationError::new(
                GpuResourceRealizationErrorCategory::UnknownLogicalResource,
                Some(parent.logical_identity()),
                "the supplied parent is not the authoritative texture realization record",
            ));
        }
        if let Some(record) = registries.texture_views.lookup(identity, &descriptor)? {
            return Ok(GpuRealizedTextureView::from_record(record));
        }
        registries.reject_other_kind(ResourceKind::TextureView, identity)?;
        registries.ensure_capacity(identity, self.backend.resource_realization.policy)?;

        let subresources = descriptor.subresources();
        let object = self.backend.resource_realization.create_backend_object(
            identity,
            &attribution_gate,
            || {
                authoritative_parent
                    .object
                    .create_view(&TextureViewDescriptor {
                        label: Some(descriptor.common().label().as_str()),
                        format: descriptor.format().map(lowering::map_texture_format),
                        dimension: Some(lowering::map_texture_view_dimension(&descriptor)),
                        usage: None,
                        aspect: map_texture_aspect(subresources.aspect()),
                        base_mip_level: subresources.base_mip_level(),
                        mip_level_count: Some(subresources.mip_level_count()),
                        base_array_layer: subresources.base_array_layer(),
                        array_layer_count: Some(subresources.array_layer_count()),
                    })
            },
        )?;
        let record = Arc::new(TextureViewRealizationRecord {
            affinity: self.affinity(),
            logical_identity: identity,
            descriptor,
            object,
            parent: authoritative_parent,
        });
        registries
            .texture_views
            .insert(identity, Arc::clone(&record));
        Ok(GpuRealizedTextureView::from_record(record))
    }

    pub fn realize_sampler(
        &self,
        handle: &GpuSamplerHandle,
    ) -> Result<GpuRealizedSampler, GpuResourceRealizationError> {
        let identity = handle.diagnostic_identity();
        lowering::validate_resource_ownership(identity, handle.descriptor().common())?;
        self.backend
            .resource_realization
            .ensure_available(identity)?;
        lowering::validate_sampler(identity, handle.descriptor())?;
        let descriptor = handle.retained_descriptor();
        {
            let registries = self.backend.resource_realization.registries(identity)?;
            if let Some(record) = registries.samplers.lookup(identity, &descriptor)? {
                return Ok(GpuRealizedSampler::from_record(record));
            }
            registries.reject_other_kind(ResourceKind::Sampler, identity)?;
        }
        let (attribution_gate, mut registries) = self
            .backend
            .resource_realization
            .attributed_registries(identity)?;
        if let Some(record) = registries.samplers.lookup(identity, &descriptor)? {
            return Ok(GpuRealizedSampler::from_record(record));
        }
        registries.reject_other_kind(ResourceKind::Sampler, identity)?;
        registries.ensure_capacity(identity, self.backend.resource_realization.policy)?;

        let (address_u, address_v, address_w) = descriptor.address_modes();
        let (mag_filter, min_filter, mipmap_filter) = descriptor.filters();
        let (lod_min_clamp, lod_max_clamp) = descriptor.lod_range();
        let object = self.backend.resource_realization.create_backend_object(
            identity,
            &attribution_gate,
            || {
                self.backend.device.create_sampler(&SamplerDescriptor {
                    label: Some(descriptor.common().label().as_str()),
                    address_mode_u: lowering::map_address_mode(address_u),
                    address_mode_v: lowering::map_address_mode(address_v),
                    address_mode_w: lowering::map_address_mode(address_w),
                    mag_filter: lowering::map_filter_mode(mag_filter),
                    min_filter: lowering::map_filter_mode(min_filter),
                    mipmap_filter: lowering::map_mipmap_filter_mode(mipmap_filter),
                    lod_min_clamp,
                    lod_max_clamp,
                    compare: descriptor.compare().map(lowering::map_compare_function),
                    anisotropy_clamp: 1,
                    border_color: None,
                })
            },
        )?;
        let record = Arc::new(SamplerRealizationRecord {
            affinity: self.affinity(),
            logical_identity: identity,
            descriptor,
            object,
        });
        registries.samplers.insert(identity, Arc::clone(&record));
        Ok(GpuRealizedSampler::from_record(record))
    }

    pub fn realize_query_set(
        &self,
        handle: &GpuQuerySetHandle,
    ) -> Result<GpuRealizedQuerySet, GpuResourceRealizationError> {
        let identity = handle.diagnostic_identity();
        lowering::validate_resource_ownership(identity, handle.descriptor().common())?;
        self.backend
            .resource_realization
            .ensure_available(identity)?;
        lowering::validate_query_set(self, identity, handle.descriptor())?;
        let descriptor = handle.retained_descriptor();
        {
            let registries = self.backend.resource_realization.registries(identity)?;
            if let Some(record) = registries.query_sets.lookup(identity, &descriptor)? {
                return Ok(GpuRealizedQuerySet::from_record(record));
            }
            registries.reject_other_kind(ResourceKind::QuerySet, identity)?;
        }
        let (attribution_gate, mut registries) = self
            .backend
            .resource_realization
            .attributed_registries(identity)?;
        if let Some(record) = registries.query_sets.lookup(identity, &descriptor)? {
            return Ok(GpuRealizedQuerySet::from_record(record));
        }
        registries.reject_other_kind(ResourceKind::QuerySet, identity)?;
        registries.ensure_capacity(identity, self.backend.resource_realization.policy)?;

        let object = self.backend.resource_realization.create_backend_object(
            identity,
            &attribution_gate,
            || {
                self.backend.device.create_query_set(&QuerySetDescriptor {
                    label: Some(descriptor.common().label().as_str()),
                    ty: lowering::map_query_kind(descriptor.kind()),
                    count: descriptor.count(),
                })
            },
        )?;
        let record = Arc::new(QuerySetRealizationRecord {
            affinity: self.affinity(),
            logical_identity: identity,
            descriptor,
            object,
        });
        registries.query_sets.insert(identity, Arc::clone(&record));
        Ok(GpuRealizedQuerySet::from_record(record))
    }
}

pub(super) fn validate_realized_input_affinity(
    expected: GpuContextAffinity,
    resource: GpuWorkResourceId,
    observed: GpuContextAffinity,
) -> Result<(), GpuResourceRealizationError> {
    match if observed.context() != expected.context() {
        Err(GpuContextAffinityError::ForeignContext)
    } else if observed.generation() != expected.generation() {
        Err(GpuContextAffinityError::StaleGeneration)
    } else {
        Ok(())
    } {
        Ok(()) => Ok(()),
        Err(GpuContextAffinityError::ForeignContext) => Err(GpuResourceRealizationError::affinity(
            GpuResourceRealizationErrorCategory::ForeignContext,
            Some(resource),
            expected,
            observed,
        )),
        Err(GpuContextAffinityError::StaleGeneration) => {
            Err(GpuResourceRealizationError::affinity(
                GpuResourceRealizationErrorCategory::StaleDeviceGeneration,
                Some(resource),
                expected,
                observed,
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{GpuContextId, GpuDeviceGeneration};
    use std::num::NonZeroU64;

    fn backend_source() -> Box<dyn std::error::Error + Send + Sync> {
        Box::new(std::io::Error::other("synthetic backend evidence"))
    }

    fn test_resource() -> GpuWorkResourceId {
        let mut allocator = crate::GpuWorkResourceIdAllocator::new();
        allocator.allocate().unwrap()
    }

    #[test]
    fn affinity_failure_order_distinguishes_foreign_context_and_stale_generation() {
        let context_one = GpuContextId::test_value(NonZeroU64::new(1).unwrap());
        let context_two = GpuContextId::test_value(NonZeroU64::new(2).unwrap());
        let generation_one = GpuDeviceGeneration::first();
        let generation_two = GpuDeviceGeneration::test_value(NonZeroU64::new(2).unwrap());
        let resource = test_resource();
        let expected = GpuContextAffinity::test_value(context_one, generation_one);

        let foreign = validate_realized_input_affinity(
            expected,
            resource,
            GpuContextAffinity::test_value(context_two, generation_two),
        )
        .unwrap_err();
        assert_eq!(
            foreign.category(),
            GpuResourceRealizationErrorCategory::ForeignContext
        );

        let stale = validate_realized_input_affinity(
            expected,
            resource,
            GpuContextAffinity::test_value(context_one, generation_two),
        )
        .unwrap_err();
        assert_eq!(
            stale.category(),
            GpuResourceRealizationErrorCategory::StaleDeviceGeneration
        );
    }

    #[test]
    fn device_health_keeps_uncaptured_validation_oom_and_internal_outcomes_distinct() {
        let cases = [
            (
                wgpu::Error::Validation {
                    source: backend_source(),
                    description: "validation".to_string(),
                },
                GpuResourceRealizationErrorCategory::UnexpectedBackendValidationRejection,
            ),
            (
                wgpu::Error::OutOfMemory {
                    source: backend_source(),
                },
                GpuResourceRealizationErrorCategory::BackendResourceExhaustion,
            ),
            (
                wgpu::Error::Internal {
                    source: backend_source(),
                    description: "backend internal failure".to_string(),
                },
                GpuResourceRealizationErrorCategory::ContextOrDeviceUnavailableOrLost,
            ),
        ];

        for (backend_error, expected) in cases {
            let health = WgpuDeviceHealth::new();
            let resource = test_resource();
            health.mark_uncaptured(backend_error);
            let error = health.ensure_resource(resource).unwrap_err();
            assert_eq!(error.category(), expected);
            assert!(
                error.detail().is_some_and(|detail| detail.len() <= 256),
                "backend evidence must remain bounded"
            );
        }
    }

    #[test]
    fn synchronous_observed_backend_fault_rejects_constructor_candidate() {
        let context = GpuContextId::test_value(NonZeroU64::new(1).unwrap());
        let affinity = GpuContextAffinity::test_value(context, GpuDeviceGeneration::first());
        let health = Arc::new(WgpuDeviceHealth::new());
        let state = ResourceRealizationState::new(
            affinity,
            GpuResourceRealizationPolicy::default(),
            Arc::clone(&health),
            Arc::new(WgpuErrorAttributionGate::default()),
        );
        let resource = test_resource();
        let attribution_gate = state.error_attribution_gate.acquire();

        let error = state
            .create_backend_object(resource, &attribution_gate, || {
                health.mark_uncaptured(wgpu::Error::Validation {
                    source: backend_source(),
                    description: "synchronous validation".to_string(),
                });
                7_u32
            })
            .unwrap_err();
        assert_eq!(
            error.category(),
            GpuResourceRealizationErrorCategory::UnexpectedBackendValidationRejection
        );
    }
}
