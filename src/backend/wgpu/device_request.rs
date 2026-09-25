use super::adapter_mapping::adapter_facts;
use super::{
    PipelineRealizationState, ProgramBindingRealizationState, ResourceRealizationState,
    WgpuContextState, WgpuDeviceHealth, WgpuErrorAttributionGate, WgpuExecutionState,
    WgpuSurfaceState,
};
#[cfg(not(target_arch = "wasm32"))]
use crate::GpuAdapterFacts;
use crate::api::texture_format;
use crate::{
    GpuAlignmentFacts, GpuCandidateEnvironmentEvidence, GpuCandidateId, GpuCandidateInput,
    GpuCandidateSelection, GpuCandidateSelectionKind, GpuCapabilityFeature, GpuContext,
    GpuContextAdmissionReport, GpuContextAffinity, GpuContextDescriptor, GpuContextId,
    GpuContextRequestError, GpuContextRequestErrorCategory, GpuDeviceGeneration, GpuDeviceLimits,
    GpuDeviceRequestProfile, GpuExecutionPolicy, GpuFallbackStatus, GpuLimits,
    GpuRealizationPolicies, GpuSoftwareFallbackPolicy, GpuTextureFormat, admitted_device_facts,
    allocate_context_id, select_candidate_inputs,
};
use std::sync::Arc;
#[cfg(not(target_arch = "wasm32"))]
use wgpu::Backends;
use wgpu::{
    Adapter, DeviceDescriptor, ExperimentalFeatures, Features, Instance, InstanceDescriptor,
    InstanceFlags, Limits, MemoryHints, RequestAdapterError, RequestAdapterOptions, Surface, Trace,
};

#[cfg(not(target_arch = "wasm32"))]
struct NativeAdapterCandidate<T> {
    id: GpuCandidateId,
    facts: GpuAdapterFacts,
    adapter: T,
    environment: GpuCandidateEnvironmentEvidence,
}

struct ContextGenerationSeed {
    id: GpuContextId,
    generation: GpuDeviceGeneration,
}

pub(crate) async fn request_headless(
    descriptor: GpuContextDescriptor,
    realization_policies: GpuRealizationPolicies,
    execution_policy: GpuExecutionPolicy,
) -> Result<GpuContext, GpuContextRequestError> {
    request_with_instance_generation(
        Instance::new(runengpu_instance_descriptor()),
        descriptor,
        None,
        realization_policies,
        execution_policy,
        None,
    )
    .await
}

/// Requests one successor device generation through an already-owned private instance.
///
/// The caller supplies the existing logical context identity and checked successor generation.
/// No old physical realization, retained coverage, content continuity, or surface state enters this
/// admission path; lifecycle owners install any reconstruction requirements only after successful
/// fresh-generation admission.
pub(crate) async fn request_generation_with_instance(
    instance: Instance,
    descriptor: GpuContextDescriptor,
    realization_policies: GpuRealizationPolicies,
    execution_policy: GpuExecutionPolicy,
    id: GpuContextId,
    generation: GpuDeviceGeneration,
) -> Result<GpuContext, GpuContextRequestError> {
    request_with_instance_generation(
        instance,
        descriptor,
        None,
        realization_policies,
        execution_policy,
        Some(ContextGenerationSeed { id, generation }),
    )
    .await
}

fn runengpu_instance_descriptor() -> InstanceDescriptor {
    enforce_runengpu_instance_flags(InstanceDescriptor::new_without_display_handle_from_env())
}

pub(super) fn enforce_runengpu_instance_flags(
    mut descriptor: InstanceDescriptor,
) -> InstanceDescriptor {
    // IndirectExecution has defined portable runtime-invalid no-op semantics. Environment/debug
    // configuration may not weaken that RunenGPU contract on the private WGPU backend.
    descriptor
        .flags
        .insert(InstanceFlags::VALIDATION_INDIRECT_CALL);
    descriptor
}

pub(super) async fn request_with_instance(
    instance: Instance,
    descriptor: GpuContextDescriptor,
    compatible_surface: Option<&Surface<'_>>,
    realization_policies: GpuRealizationPolicies,
    execution_policy: GpuExecutionPolicy,
) -> Result<GpuContext, GpuContextRequestError> {
    request_with_instance_generation(
        instance,
        descriptor,
        compatible_surface,
        realization_policies,
        execution_policy,
        None,
    )
    .await
}

async fn request_with_instance_generation(
    instance: Instance,
    descriptor: GpuContextDescriptor,
    compatible_surface: Option<&Surface<'_>>,
    realization_policies: GpuRealizationPolicies,
    execution_policy: GpuExecutionPolicy,
    generation_seed: Option<ContextGenerationSeed>,
) -> Result<GpuContext, GpuContextRequestError> {
    crate::validate_descriptor(&descriptor)?;
    let (adapter, selection, selection_kind) =
        select_backend_adapter(&instance, &descriptor, compatible_surface).await?;
    let candidate = selection.candidate;
    let requested_limits = requested_limits(&candidate)?;
    if !requested_limits.check_limits(&adapter.limits()) {
        return Err(GpuContextRequestError::new(
            GpuContextRequestErrorCategory::DeviceRequestProfileUnsupported,
            "selected adapter cannot satisfy the complete requested device profile",
        ));
    }
    let requested_features = requested_features(&candidate);
    let (device, queue) = adapter
        .request_device(&DeviceDescriptor {
            label: descriptor.label(),
            required_features: requested_features,
            required_limits: requested_limits.clone(),
            experimental_features: ExperimentalFeatures::disabled(),
            memory_hints: MemoryHints::Performance,
            trace: Trace::Off,
        })
        .await
        .map_err(|error| {
            GpuContextRequestError::new(
                GpuContextRequestErrorCategory::BackendDeviceRequestFailure,
                error.to_string(),
            )
        })?;
    verify_requested_features(requested_features, device.features())?;
    let actual_native_limits = device.limits();
    if !requested_limits.check_limits(&actual_native_limits) {
        return Err(GpuContextRequestError::new(
            GpuContextRequestErrorCategory::BackendDeviceRequestFailure,
            "created device did not expose the complete admitted request profile",
        ));
    }
    let device_facts = admitted_device_facts(
        &candidate,
        map_device_limits(&actual_native_limits),
        selection.dispositions.clone(),
    )?;
    let ContextGenerationSeed { id, generation } = match generation_seed {
        Some(seed) => seed,
        None => ContextGenerationSeed {
            id: allocate_context_id()?,
            generation: GpuDeviceGeneration::first(),
        },
    };
    let affinity = GpuContextAffinity::from_context_generation(id, generation);
    let health = Arc::new(WgpuDeviceHealth::new());
    health.install_observers(&device);
    let error_attribution_gate = Arc::new(WgpuErrorAttributionGate::default());
    let resource_realization = ResourceRealizationState::new(
        affinity,
        realization_policies.resource(),
        Arc::clone(&health),
        Arc::clone(&error_attribution_gate),
    );
    let program_binding_realization = ProgramBindingRealizationState::new(
        affinity,
        realization_policies.program_binding(),
        Arc::clone(&health),
        Arc::clone(&error_attribution_gate),
    );
    let pipeline_realization = PipelineRealizationState::new(
        affinity,
        Arc::clone(&health),
        Arc::clone(&error_attribution_gate),
    );
    let execution = Arc::new(WgpuExecutionState::new(affinity, execution_policy));
    let surfaces = WgpuSurfaceState::new(affinity);
    let adapter_facts = candidate.adapter().clone();
    Ok(GpuContext {
        id,
        generation,
        adapter: adapter_facts,
        device: device_facts,
        report: GpuContextAdmissionReport {
            selected: selection_kind,
            candidate,
            candidate_dispositions: selection.dispositions,
            selection_evidence: selection.evidence,
        },
        backend: WgpuContextState {
            instance,
            adapter,
            device: Arc::new(device),
            queue: Arc::new(queue),
            health,
            error_attribution_gate,
            resource_realization,
            program_binding_realization,
            pipeline_realization,
            execution,
            surfaces,
        },
    })
}

#[cfg(not(target_arch = "wasm32"))]
async fn select_backend_adapter(
    instance: &Instance,
    descriptor: &GpuContextDescriptor,
    compatible_surface: Option<&Surface<'_>>,
) -> Result<(Adapter, GpuCandidateSelection, GpuCandidateSelectionKind), GpuContextRequestError> {
    if native_selection_route(descriptor.fallback_policy())
        == NativeAdapterSelectionRoute::ForcedFallback
    {
        return select_backend_selected_adapter(instance, descriptor, compatible_surface, true)
            .await;
    }
    let candidates = instance
        .enumerate_adapters(Backends::all())
        .await
        .into_iter()
        .map(|adapter| -> Result<_, GpuContextRequestError> {
            let surface_supported =
                compatible_surface.is_some_and(|surface| adapter.is_surface_supported(surface));
            let environment = if compatible_surface.is_some() {
                GpuCandidateEnvironmentEvidence::current_host(surface_supported)
            } else {
                GpuCandidateEnvironmentEvidence::headless()
            };
            Ok(NativeAdapterCandidate {
                id: GpuCandidateId::allocate()?,
                facts: adapter_facts(
                    &adapter,
                    surface_supported,
                    ordinary_enumeration_fallback_status(),
                ),
                adapter,
                environment,
            })
        })
        .collect::<Result<Vec<_>, GpuContextRequestError>>()?;
    select_enumerated_adapter(descriptor, candidates).map(|(adapter, selection)| {
        (
            adapter,
            selection,
            GpuCandidateSelectionKind::DeterministicallyRanked,
        )
    })
}

#[cfg(not(target_arch = "wasm32"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NativeAdapterSelectionRoute {
    Enumerated,
    ForcedFallback,
}

#[cfg(not(target_arch = "wasm32"))]
const fn native_selection_route(policy: GpuSoftwareFallbackPolicy) -> NativeAdapterSelectionRoute {
    match policy {
        GpuSoftwareFallbackPolicy::Require => NativeAdapterSelectionRoute::ForcedFallback,
        GpuSoftwareFallbackPolicy::Allow | GpuSoftwareFallbackPolicy::Forbid => {
            NativeAdapterSelectionRoute::Enumerated
        }
    }
}

const fn ordinary_enumeration_fallback_status() -> GpuFallbackStatus {
    GpuFallbackStatus::Unknown
}

#[cfg(not(target_arch = "wasm32"))]
fn select_enumerated_adapter<T>(
    descriptor: &GpuContextDescriptor,
    mut candidates: Vec<NativeAdapterCandidate<T>>,
) -> Result<(T, GpuCandidateSelection), GpuContextRequestError> {
    candidates.sort_by_key(|candidate| {
        crate::canonical_candidate_input_key(&candidate.facts, candidate.environment)
    });
    let selection = select_candidate_inputs(
        descriptor,
        candidates.iter().map(|candidate| GpuCandidateInput {
            id: candidate.id,
            adapter: candidate.facts.clone(),
            environment: candidate.environment,
        }),
    )?;
    let selected = selection.backend_candidate_id;
    let adapter = candidates
        .into_iter()
        .find(|candidate| candidate.id == selected)
        .map(|candidate| candidate.adapter)
        .ok_or_else(|| {
            GpuContextRequestError::new(
                GpuContextRequestErrorCategory::BackendAdapterRequestFailure,
                "selected candidate ID is absent from the native candidate set",
            )
        })?;
    Ok((adapter, selection))
}

#[cfg(target_arch = "wasm32")]
async fn select_backend_adapter(
    instance: &Instance,
    descriptor: &GpuContextDescriptor,
    compatible_surface: Option<&Surface<'_>>,
) -> Result<(Adapter, GpuCandidateSelection, GpuCandidateSelectionKind), GpuContextRequestError> {
    let fallback_required = matches!(
        descriptor.fallback_policy(),
        GpuSoftwareFallbackPolicy::Require
    );
    select_backend_selected_adapter(instance, descriptor, compatible_surface, fallback_required)
        .await
}

async fn select_backend_selected_adapter(
    instance: &Instance,
    descriptor: &GpuContextDescriptor,
    compatible_surface: Option<&Surface<'_>>,
    force_fallback_adapter: bool,
) -> Result<(Adapter, GpuCandidateSelection, GpuCandidateSelectionKind), GpuContextRequestError> {
    let adapter = instance
        .request_adapter(&RequestAdapterOptions {
            power_preference: map_power_preference(descriptor.power_preference()),
            force_fallback_adapter,
            compatible_surface,
            apply_limit_buckets: false,
        })
        .await
        .map_err(map_request_adapter_error)?;
    let surface_supported =
        compatible_surface.is_some_and(|surface| adapter.is_surface_supported(surface));
    let environment = if compatible_surface.is_some() {
        GpuCandidateEnvironmentEvidence::current_host(surface_supported)
    } else {
        GpuCandidateEnvironmentEvidence::headless()
    };
    let selection = select_candidate_inputs(
        descriptor,
        [GpuCandidateInput {
            id: GpuCandidateId::allocate()?,
            adapter: adapter_facts(
                &adapter,
                surface_supported,
                backend_selected_fallback_status(force_fallback_adapter),
            ),
            environment,
        }],
    )?;
    Ok((
        adapter,
        selection,
        GpuCandidateSelectionKind::BackendSelectedCandidate,
    ))
}

const fn backend_selected_fallback_status(force_fallback_adapter: bool) -> GpuFallbackStatus {
    if force_fallback_adapter {
        GpuFallbackStatus::ConfirmedFallback
    } else {
        GpuFallbackStatus::Unknown
    }
}

fn map_request_adapter_error(error: RequestAdapterError) -> GpuContextRequestError {
    let category = match error {
        RequestAdapterError::NotFound { .. } => GpuContextRequestErrorCategory::NoAdapterAvailable,
        RequestAdapterError::EnvNotSet => {
            GpuContextRequestErrorCategory::BackendAdapterRequestFailure
        }
        _ => GpuContextRequestErrorCategory::BackendAdapterRequestFailure,
    };
    GpuContextRequestError::new(category, error.to_string())
}

fn map_power_preference(preference: crate::GpuPowerPreference) -> wgpu::PowerPreference {
    match preference {
        crate::GpuPowerPreference::HighPerformance => wgpu::PowerPreference::HighPerformance,
        crate::GpuPowerPreference::LowPower => wgpu::PowerPreference::LowPower,
        crate::GpuPowerPreference::NoPreference => wgpu::PowerPreference::None,
    }
}

fn requested_features(candidate: &crate::GpuCandidateAdmissionReport) -> Features {
    let mut features = candidate
        .enabled_features()
        .fold(Features::empty(), |features, feature| {
            features | wgpu_features_for(feature)
        });
    for (format, role) in candidate.contract().format_roles() {
        if matches!(
            texture_format::compression_family(format),
            Some(texture_format::GpuTextureCompressionFamily::Bc)
        ) {
            features |= Features::TEXTURE_COMPRESSION_BC;
        }
        match (format, role) {
            (GpuTextureFormat::Depth32FloatStencil8, _) => {
                features |= Features::DEPTH32FLOAT_STENCIL8;
            }
            (GpuTextureFormat::Rg11b10Ufloat, crate::GpuFormatRole::ColorAttachment) => {
                features |= Features::RG11B10UFLOAT_RENDERABLE;
            }
            _ => {}
        }
    }
    features
}

fn wgpu_features_for(feature: GpuCapabilityFeature) -> Features {
    match feature {
        GpuCapabilityFeature::TimestampQuery => Features::TIMESTAMP_QUERY,
        GpuCapabilityFeature::TextureBindingArray => Features::TEXTURE_BINDING_ARRAY,
        GpuCapabilityFeature::BufferBindingArray => Features::BUFFER_BINDING_ARRAY,
        GpuCapabilityFeature::StorageResourceBindingArray => {
            Features::STORAGE_RESOURCE_BINDING_ARRAY
        }
        GpuCapabilityFeature::UniformBufferBindingArray => Features::UNIFORM_BUFFER_BINDING_ARRAYS,
        GpuCapabilityFeature::ShaderF16 => Features::SHADER_F16,
        GpuCapabilityFeature::DepthBiasClamp => Features::empty(),
        _ => Features::empty(),
    }
}

fn verify_requested_features(
    requested: Features,
    actual: Features,
) -> Result<(), GpuContextRequestError> {
    if actual.contains(requested) {
        Ok(())
    } else {
        Err(GpuContextRequestError::new(
            GpuContextRequestErrorCategory::BackendDeviceRequestFailure,
            "created device did not expose every admitted WGPU feature",
        ))
    }
}

pub(super) const fn profile_limits(profile: GpuDeviceRequestProfile) -> Limits {
    match profile {
        GpuDeviceRequestProfile::ModernPortable | GpuDeviceRequestProfile::BrowserWebGpu => {
            Limits::defaults()
        }
        GpuDeviceRequestProfile::Downlevel => Limits::downlevel_defaults(),
        GpuDeviceRequestProfile::DownlevelWebGl2 => Limits::downlevel_webgl2_defaults(),
    }
}

fn requested_limits(
    candidate: &crate::GpuCandidateAdmissionReport,
) -> Result<Limits, GpuContextRequestError> {
    let contract = candidate.contract();
    let budget = contract.workload_budget().limits();
    let mut limits = profile_limits(contract.device_request_profile());
    limits.max_uniform_buffer_binding_size = budget.max_uniform_buffer_binding_size();
    limits.max_storage_buffer_binding_size = budget.max_storage_buffer_binding_size();
    limits.max_color_attachments = budget.max_color_attachments();
    limits.max_vertex_buffers = budget.max_vertex_buffers();
    limits.max_bindings_per_bind_group = budget.max_bindings_per_group();
    limits.max_texture_dimension_2d = budget.max_texture_dimension_2d();
    limits.max_bind_groups = budget.max_bind_groups();
    limits.max_bind_groups_plus_vertex_buffers = budget.max_bind_groups_plus_vertex_buffers();
    limits.max_dynamic_uniform_buffers_per_pipeline_layout =
        budget.max_dynamic_uniform_buffers_per_pipeline_layout();
    limits.max_dynamic_storage_buffers_per_pipeline_layout =
        budget.max_dynamic_storage_buffers_per_pipeline_layout();
    limits.max_compute_workgroups_per_dimension = budget.max_compute_workgroups_per_dimension();
    limits.max_buffer_size = budget.max_buffer_size();
    limits.max_texture_dimension_1d = budget.max_texture_dimension_1d();
    limits.max_texture_dimension_3d = budget.max_texture_dimension_3d();
    limits.max_texture_array_layers = budget.max_texture_array_layers();
    limits.max_vertex_attributes = budget.max_vertex_attributes();
    limits.max_vertex_buffer_array_stride = budget.max_vertex_buffer_array_stride();
    limits.max_binding_array_elements_per_shader_stage =
        budget.max_binding_array_elements_per_shader_stage();
    limits.max_binding_array_sampler_elements_per_shader_stage =
        budget.max_binding_array_sampler_elements_per_shader_stage();
    let alignments = contract.selected_alignments();
    limits.min_uniform_buffer_offset_alignment =
        requested_alignment(alignments.uniform_dynamic_offset, "uniform dynamic offset")?;
    limits.min_storage_buffer_offset_alignment =
        requested_alignment(alignments.storage_dynamic_offset, "storage dynamic offset")?;
    Ok(limits)
}

fn requested_alignment(
    value: Option<u64>,
    label: &'static str,
) -> Result<u32, GpuContextRequestError> {
    let value = value.ok_or_else(|| {
        GpuContextRequestError::new(
            GpuContextRequestErrorCategory::AlignmentIncompatibility,
            format!("{label} has no requestable admitted value"),
        )
    })?;
    u32::try_from(value).map_err(|_| {
        GpuContextRequestError::new(
            GpuContextRequestErrorCategory::AlignmentIncompatibility,
            format!("{label} exceeds the pinned WGPU alignment domain"),
        )
    })
}

fn map_device_limits(native: &Limits) -> GpuDeviceLimits {
    GpuDeviceLimits::new(
        GpuLimits::from_validated_adapter_facts(
            native.max_uniform_buffer_binding_size,
            native.max_storage_buffer_binding_size,
            native.max_color_attachments,
            native.max_vertex_buffers,
            native.max_bindings_per_bind_group,
            native.max_texture_dimension_2d,
            native.max_bind_groups,
            native.max_bind_groups_plus_vertex_buffers,
            native.max_dynamic_uniform_buffers_per_pipeline_layout,
            native.max_dynamic_storage_buffers_per_pipeline_layout,
            native.max_compute_workgroups_per_dimension,
            native.max_buffer_size,
            native.max_texture_dimension_1d,
            native.max_texture_dimension_3d,
            native.max_texture_array_layers,
            native.max_vertex_attributes,
            native.max_vertex_buffer_array_stride,
        )
        .with_binding_array_limits(
            native.max_binding_array_elements_per_shader_stage,
            native.max_binding_array_sampler_elements_per_shader_stage,
        ),
        GpuAlignmentFacts {
            uniform_dynamic_offset: Some(u64::from(native.min_uniform_buffer_offset_alignment)),
            storage_dynamic_offset: Some(u64::from(native.min_storage_buffer_offset_alignment)),
            copy_buffer_offset: Some(wgpu::COPY_BUFFER_ALIGNMENT),
            bytes_per_row: Some(u64::from(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)),
            query_resolve_destination: Some(wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT),
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        GpuAdapterClass, GpuAdapterLimits, GpuBackendFamily, GpuCapabilities,
        GpuCapabilityRequirement, GpuCapabilityRequirements, GpuContextDescriptor,
        GpuFallbackStatus, GpuFormatRole, GpuSoftwareStatus, GpuTextureFormatCapabilities,
        select_candidate_with_host_evidence,
    };

    fn candidate() -> crate::GpuCandidateAdmissionReport {
        candidate_with_enabled_features([])
    }

    fn test_gpu_limits() -> GpuLimits {
        let native = Limits::defaults();
        GpuLimits::new(
            256 * 1024,
            512 * 1024 * 1024,
            8,
            16,
            128,
            native.max_texture_dimension_2d,
            native.max_bind_groups,
            native.max_bind_groups_plus_vertex_buffers,
            native.max_dynamic_uniform_buffers_per_pipeline_layout,
            native.max_dynamic_storage_buffers_per_pipeline_layout,
            native.max_compute_workgroups_per_dimension,
            256 * 1024 * 1024,
            8192,
            2048,
            256,
            16,
            2048,
        )
        .unwrap()
    }

    fn candidate_with_enabled_features(
        enabled_features: impl IntoIterator<Item = GpuCapabilityFeature>,
    ) -> crate::GpuCandidateAdmissionReport {
        let enabled_features = enabled_features.into_iter().collect::<Vec<_>>();
        let limits = test_gpu_limits().with_binding_array_limits(
            if enabled_features.contains(&GpuCapabilityFeature::TextureBindingArray)
                || enabled_features.contains(&GpuCapabilityFeature::BufferBindingArray)
            {
                500_000
            } else {
                0
            },
            if enabled_features.contains(&GpuCapabilityFeature::TextureBindingArray) {
                1_000
            } else {
                0
            },
        );
        let facts = GpuAdapterFacts::new(
            GpuBackendFamily::Vulkan,
            GpuAdapterClass::Discrete,
            GpuSoftwareStatus::Hardware,
            GpuFallbackStatus::ConfirmedNotFallback,
            GpuCapabilities::from_normalized_facts(enabled_features.iter().copied(), limits, []),
            GpuAdapterLimits::new(limits),
            GpuAlignmentFacts {
                uniform_dynamic_offset: Some(256),
                storage_dynamic_offset: Some(256),
                copy_buffer_offset: Some(4),
                bytes_per_row: Some(256),
                query_resolve_destination: Some(256),
            },
        );
        let mut requirements = GpuCapabilityRequirements::new();
        for feature in enabled_features {
            requirements
                .insert(GpuCapabilityRequirement::Required(feature))
                .expect("test feature requirements should not conflict");
        }
        select_candidate_with_host_evidence(
            &GpuContextDescriptor::new(requirements),
            [(facts, GpuCandidateEnvironmentEvidence::headless())],
        )
        .unwrap()
        .candidate
    }

    #[test]
    fn runengpu_instance_flags_restore_indirect_runtime_validity() {
        let mut descriptor = InstanceDescriptor::new_without_display_handle();
        descriptor
            .flags
            .remove(InstanceFlags::VALIDATION_INDIRECT_CALL);
        let descriptor = enforce_runengpu_instance_flags(descriptor);
        assert!(
            descriptor
                .flags
                .contains(InstanceFlags::VALIDATION_INDIRECT_CALL)
        );
    }

    #[test]
    fn every_profile_is_complete_and_minimal_budget_does_not_request_adapter_maxima() {
        assert_eq!(
            profile_limits(GpuDeviceRequestProfile::ModernPortable),
            Limits::defaults()
        );
        assert_eq!(
            profile_limits(GpuDeviceRequestProfile::BrowserWebGpu),
            Limits::defaults()
        );
        assert_eq!(
            profile_limits(GpuDeviceRequestProfile::Downlevel),
            Limits::downlevel_defaults()
        );
        assert_eq!(
            profile_limits(GpuDeviceRequestProfile::DownlevelWebGl2),
            Limits::downlevel_webgl2_defaults()
        );
        let requested = requested_limits(&candidate()).unwrap();
        let budget = candidate().contract().workload_budget().limits();
        assert_eq!(requested.max_uniform_buffer_binding_size, 64 * 1024);
        assert_eq!(requested.max_storage_buffer_binding_size, 128 * 1024 * 1024);
        assert_eq!(requested.max_color_attachments, 1);
        assert_eq!(requested.max_vertex_buffers, 8);
        assert_eq!(requested.max_bindings_per_bind_group, 16);
        assert_eq!(
            requested.max_texture_dimension_2d,
            budget.max_texture_dimension_2d()
        );
        assert_eq!(requested.max_bind_groups, budget.max_bind_groups());
        assert_eq!(
            requested.max_bind_groups_plus_vertex_buffers,
            budget.max_bind_groups_plus_vertex_buffers()
        );
        assert_eq!(
            requested.max_dynamic_uniform_buffers_per_pipeline_layout,
            budget.max_dynamic_uniform_buffers_per_pipeline_layout()
        );
        assert_eq!(
            requested.max_dynamic_storage_buffers_per_pipeline_layout,
            budget.max_dynamic_storage_buffers_per_pipeline_layout()
        );
        assert_eq!(
            requested.max_compute_workgroups_per_dimension,
            budget.max_compute_workgroups_per_dimension()
        );
        assert_eq!(requested.max_buffer_size, budget.max_buffer_size());
        assert_eq!(
            requested.max_texture_dimension_1d,
            budget.max_texture_dimension_1d()
        );
        assert_eq!(
            requested.max_texture_dimension_3d,
            budget.max_texture_dimension_3d()
        );
        assert_eq!(
            requested.max_texture_array_layers,
            budget.max_texture_array_layers()
        );
        assert_eq!(
            requested.max_vertex_attributes,
            budget.max_vertex_attributes()
        );
        assert_eq!(
            requested.max_vertex_buffer_array_stride,
            budget.max_vertex_buffer_array_stride()
        );
        assert_eq!(requested.max_binding_array_elements_per_shader_stage, 0);
        assert_eq!(
            requested.max_binding_array_sampler_elements_per_shader_stage,
            0
        );
        assert_eq!(requested.min_uniform_buffer_offset_alignment, 256);
        assert_eq!(requested.min_storage_buffer_offset_alignment, 256);
    }

    #[test]
    fn actual_device_mapping_records_only_actual_native_facts() {
        let mut native = Limits::defaults();
        native.max_vertex_buffers = 12;
        native.max_compute_workgroups_per_dimension = 1234;
        native.max_buffer_size = 123_456_789;
        native.max_texture_dimension_1d = 4096;
        native.max_texture_dimension_3d = 1024;
        native.max_texture_array_layers = 128;
        native.max_vertex_attributes = 12;
        native.max_vertex_buffer_array_stride = 1024;
        native.max_binding_array_elements_per_shader_stage = 123_456;
        native.max_binding_array_sampler_elements_per_shader_stage = 789;
        native.min_uniform_buffer_offset_alignment = 512;
        let facts = map_device_limits(&native);
        assert_eq!(facts.values().max_vertex_buffers(), 12);
        assert_eq!(facts.values().max_compute_workgroups_per_dimension(), 1234);
        assert_eq!(facts.values().max_buffer_size(), 123_456_789);
        assert_eq!(facts.values().max_texture_dimension_1d(), 4096);
        assert_eq!(facts.values().max_texture_dimension_3d(), 1024);
        assert_eq!(facts.values().max_texture_array_layers(), 128);
        assert_eq!(facts.values().max_vertex_attributes(), 12);
        assert_eq!(facts.values().max_vertex_buffer_array_stride(), 1024);
        assert_eq!(
            facts.values().max_binding_array_elements_per_shader_stage(),
            123_456
        );
        assert_eq!(
            facts
                .values()
                .max_binding_array_sampler_elements_per_shader_stage(),
            789
        );
        assert_eq!(facts.alignments().uniform_dynamic_offset, Some(512));
    }

    #[test]
    fn created_device_must_verify_every_requested_wgpu_feature() {
        assert!(verify_requested_features(Features::TIMESTAMP_QUERY, Features::empty()).is_err());
        assert!(
            verify_requested_features(Features::TIMESTAMP_QUERY, Features::TIMESTAMP_QUERY).is_ok()
        );
        assert!(
            verify_requested_features(Features::DEPTH32FLOAT_STENCIL8, Features::empty()).is_err()
        );
        assert!(
            verify_requested_features(
                Features::DEPTH32FLOAT_STENCIL8,
                Features::DEPTH32FLOAT_STENCIL8,
            )
            .is_ok()
        );
        assert!(
            verify_requested_features(Features::RG11B10UFLOAT_RENDERABLE, Features::empty())
                .is_err()
        );
        assert!(
            verify_requested_features(
                Features::RG11B10UFLOAT_RENDERABLE,
                Features::RG11B10UFLOAT_RENDERABLE,
            )
            .is_ok()
        );
    }

    fn candidate_with_depth32float_stencil8_role(
        role: GpuFormatRole,
    ) -> crate::GpuCandidateAdmissionReport {
        let limits = test_gpu_limits();
        let mut format = GpuTextureFormatCapabilities::none();
        match role {
            GpuFormatRole::Sampled => format.sampled = true,
            GpuFormatRole::DepthStencil => format.depth_stencil = true,
            GpuFormatRole::CopySource => format.copy_source = true,
            GpuFormatRole::CopyDestination => format.copy_destination = true,
            other => panic!("unsupported focused test role: {other:?}"),
        }
        let facts = GpuAdapterFacts::new(
            GpuBackendFamily::Vulkan,
            GpuAdapterClass::Discrete,
            GpuSoftwareStatus::Hardware,
            GpuFallbackStatus::ConfirmedNotFallback,
            GpuCapabilities::from_normalized_facts(
                [],
                limits,
                [(GpuTextureFormat::Depth32FloatStencil8, format)],
            ),
            GpuAdapterLimits::new(limits),
            GpuAlignmentFacts {
                uniform_dynamic_offset: Some(256),
                storage_dynamic_offset: Some(256),
                copy_buffer_offset: Some(4),
                bytes_per_row: Some(256),
                query_resolve_destination: Some(256),
            },
        );
        select_candidate_with_host_evidence(
            &GpuContextDescriptor::new(GpuCapabilityRequirements::new())
                .require_format_role(GpuTextureFormat::Depth32FloatStencil8, role),
            [(facts, GpuCandidateEnvironmentEvidence::headless())],
        )
        .unwrap()
        .candidate
    }

    #[test]
    fn depth32float_stencil8_roles_request_only_the_private_backend_prerequisite() {
        assert!(!requested_features(&candidate()).contains(Features::DEPTH32FLOAT_STENCIL8));
        for role in [
            GpuFormatRole::Sampled,
            GpuFormatRole::DepthStencil,
            GpuFormatRole::CopySource,
            GpuFormatRole::CopyDestination,
        ] {
            let requested = requested_features(&candidate_with_depth32float_stencil8_role(role));
            assert!(
                requested.contains(Features::DEPTH32FLOAT_STENCIL8),
                "{role:?}"
            );
        }
    }

    fn candidate_with_rg11b10_role(role: GpuFormatRole) -> crate::GpuCandidateAdmissionReport {
        let limits = test_gpu_limits();
        let mut format = GpuTextureFormatCapabilities::none();
        match role {
            GpuFormatRole::Sampled => format.sampled = true,
            GpuFormatRole::ColorAttachment => format.color_attachment = true,
            GpuFormatRole::CopySource => format.copy_source = true,
            GpuFormatRole::CopyDestination => format.copy_destination = true,
            other => panic!("unsupported focused packed role: {other:?}"),
        }
        let facts = GpuAdapterFacts::new(
            GpuBackendFamily::Vulkan,
            GpuAdapterClass::Discrete,
            GpuSoftwareStatus::Hardware,
            GpuFallbackStatus::ConfirmedNotFallback,
            GpuCapabilities::from_normalized_facts(
                [],
                limits,
                [(GpuTextureFormat::Rg11b10Ufloat, format)],
            ),
            GpuAdapterLimits::new(limits),
            GpuAlignmentFacts {
                uniform_dynamic_offset: Some(256),
                storage_dynamic_offset: Some(256),
                copy_buffer_offset: Some(4),
                bytes_per_row: Some(256),
                query_resolve_destination: Some(256),
            },
        );
        select_candidate_with_host_evidence(
            &GpuContextDescriptor::new(GpuCapabilityRequirements::new())
                .require_format_role(GpuTextureFormat::Rg11b10Ufloat, role),
            [(facts, GpuCandidateEnvironmentEvidence::headless())],
        )
        .unwrap()
        .candidate
    }

    #[test]
    fn rg11b10_render_role_requests_only_its_private_backend_prerequisite() {
        assert!(!requested_features(&candidate()).contains(Features::RG11B10UFLOAT_RENDERABLE));
        for role in [
            GpuFormatRole::Sampled,
            GpuFormatRole::CopySource,
            GpuFormatRole::CopyDestination,
        ] {
            assert!(
                !requested_features(&candidate_with_rg11b10_role(role))
                    .contains(Features::RG11B10UFLOAT_RENDERABLE),
                "{role:?}"
            );
        }
        assert!(
            requested_features(&candidate_with_rg11b10_role(GpuFormatRole::ColorAttachment))
                .contains(Features::RG11B10UFLOAT_RENDERABLE)
        );
    }

    #[test]
    fn shader_f16_requests_the_exact_wgpu_feature() {
        assert_eq!(
            wgpu_features_for(GpuCapabilityFeature::ShaderF16),
            Features::SHADER_F16
        );
    }

    #[test]
    fn depth_bias_clamp_is_an_admission_fact_not_a_wgpu_feature_request() {
        assert_eq!(
            wgpu_features_for(GpuCapabilityFeature::DepthBiasClamp),
            Features::empty()
        );
    }

    #[test]
    fn binding_array_capabilities_request_their_exact_backend_feature_bits() {
        let candidate = candidate_with_enabled_features([
            GpuCapabilityFeature::TextureBindingArray,
            GpuCapabilityFeature::BufferBindingArray,
            GpuCapabilityFeature::StorageResourceBindingArray,
        ]);

        assert_eq!(
            requested_features(&candidate),
            Features::TEXTURE_BINDING_ARRAY
                | Features::BUFFER_BINDING_ARRAY
                | Features::STORAGE_RESOURCE_BINDING_ARRAY
        );
        let requested = requested_limits(&candidate).unwrap();
        assert_eq!(
            requested.max_binding_array_elements_per_shader_stage,
            500_000
        );
        assert_eq!(
            requested.max_binding_array_sampler_elements_per_shader_stage,
            1_000
        );
        assert_eq!(
            wgpu_features_for(GpuCapabilityFeature::UniformBufferBindingArray),
            Features::UNIFORM_BUFFER_BINDING_ARRAYS
        );
    }

    fn candidate_with_bc_role(
        format: GpuTextureFormat,
        role: GpuFormatRole,
    ) -> crate::GpuCandidateAdmissionReport {
        let limits = test_gpu_limits();
        let mut facts_for_format = GpuTextureFormatCapabilities::none();
        match role {
            GpuFormatRole::Sampled => facts_for_format.sampled = true,
            GpuFormatRole::CopySource => facts_for_format.copy_source = true,
            GpuFormatRole::CopyDestination => facts_for_format.copy_destination = true,
            other => panic!("unsupported focused BC role: {other:?}"),
        }
        let facts = GpuAdapterFacts::new(
            GpuBackendFamily::Vulkan,
            GpuAdapterClass::Discrete,
            GpuSoftwareStatus::Hardware,
            GpuFallbackStatus::ConfirmedNotFallback,
            GpuCapabilities::from_normalized_facts([], limits, [(format, facts_for_format)]),
            GpuAdapterLimits::new(limits),
            GpuAlignmentFacts {
                uniform_dynamic_offset: Some(256),
                storage_dynamic_offset: Some(256),
                copy_buffer_offset: Some(4),
                bytes_per_row: Some(256),
                query_resolve_destination: Some(256),
            },
        );
        select_candidate_with_host_evidence(
            &GpuContextDescriptor::new(GpuCapabilityRequirements::new())
                .require_format_role(format, role),
            [(facts, GpuCandidateEnvironmentEvidence::headless())],
        )
        .unwrap()
        .candidate
    }

    #[test]
    fn bc_format_roles_request_the_private_compression_feature_without_public_duplication() {
        assert!(!requested_features(&candidate()).contains(Features::TEXTURE_COMPRESSION_BC));
        let formats = [
            GpuTextureFormat::Bc1RgbaUnorm,
            GpuTextureFormat::Bc1RgbaUnormSrgb,
            GpuTextureFormat::Bc2RgbaUnorm,
            GpuTextureFormat::Bc2RgbaUnormSrgb,
            GpuTextureFormat::Bc3RgbaUnorm,
            GpuTextureFormat::Bc3RgbaUnormSrgb,
            GpuTextureFormat::Bc4RUnorm,
            GpuTextureFormat::Bc4RSnorm,
            GpuTextureFormat::Bc5RgUnorm,
            GpuTextureFormat::Bc5RgSnorm,
            GpuTextureFormat::Bc6hRgbUfloat,
            GpuTextureFormat::Bc6hRgbFloat,
            GpuTextureFormat::Bc7RgbaUnorm,
            GpuTextureFormat::Bc7RgbaUnormSrgb,
        ];
        for format in formats {
            for role in [
                GpuFormatRole::Sampled,
                GpuFormatRole::CopySource,
                GpuFormatRole::CopyDestination,
            ] {
                assert!(
                    requested_features(&candidate_with_bc_role(format, role))
                        .contains(Features::TEXTURE_COMPRESSION_BC),
                    "{format:?} {role:?}"
                );
            }
        }
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn native_fallback_policies_have_only_the_explicit_selection_routes() {
        assert_eq!(
            native_selection_route(GpuSoftwareFallbackPolicy::Allow),
            NativeAdapterSelectionRoute::Enumerated
        );
        assert_eq!(
            native_selection_route(GpuSoftwareFallbackPolicy::Forbid),
            NativeAdapterSelectionRoute::Enumerated
        );
        assert_eq!(
            native_selection_route(GpuSoftwareFallbackPolicy::Require),
            NativeAdapterSelectionRoute::ForcedFallback
        );
        assert_eq!(
            ordinary_enumeration_fallback_status(),
            GpuFallbackStatus::Unknown,
            "native enumeration must not claim non-fallback evidence without backend proof"
        );
        assert_eq!(
            backend_selected_fallback_status(true),
            GpuFallbackStatus::ConfirmedFallback
        );
        assert_eq!(
            backend_selected_fallback_status(false),
            GpuFallbackStatus::Unknown
        );
    }
}
