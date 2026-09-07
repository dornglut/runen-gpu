//! Context/device-generation-bound G4C2 program, layout, and bind-group realization.

mod lowering;
mod records;
mod registry;

pub(crate) use records::{
    BindGroupLayoutRealizationRecord, BindGroupRealizationRecord, PipelineLayoutRealizationRecord,
    ProgramRealizationRecord,
};

use super::{WgpuDeviceHealth, WgpuErrorAttributionGate};
use crate::{
    GpuBindGroupLayoutDescriptor, GpuBindingDeclaration, GpuContext, GpuContextAffinity,
    GpuPipelineLayoutDescriptor, GpuProgramBindingRealizationError,
    GpuProgramBindingRealizationErrorCategory, GpuProgramBindingRealizationPolicy,
    GpuProgramBindingRealizationStats, GpuProgramDescriptor, GpuRealizedBindGroup,
    GpuRealizedBindGroupLayout, GpuRealizedPipelineLayout, GpuRealizedProgram,
    GpuRuntimeBindingResource, GpuRuntimeBindingValue, GpuValidatedBindGroupBindings,
};
use records::{
    BindGroupLayoutRealizationRecord as LayoutRecord, BindGroupRealizationRecord as GroupRecord,
    BindGroupResourceDependency, PipelineLayoutRealizationRecord as PipelineRecord,
    ProgramRealizationRecord as ProgramRecord, static_bind_group_values,
};
use registry::{
    BindGroupLayoutRequestKey, BindGroupRequestKey, InFlightOutcome, PipelineLayoutRequestKey,
    ProgramBindingRegistries, ProgramRequestKey, Reservation,
};
use std::borrow::Cow;
use std::sync::{Arc, Mutex};
use wgpu::{
    BindGroupDescriptor, BindGroupEntry, BindingResource, BufferBinding, ErrorFilter,
    PipelineLayoutDescriptor, ShaderModuleDescriptor, ShaderSource,
};

/// The sole authoritative private G4C2 owner for one admitted WGPU device generation.
pub(crate) struct ProgramBindingRealizationState {
    affinity: GpuContextAffinity,
    policy: GpuProgramBindingRealizationPolicy,
    registries: Arc<Mutex<ProgramBindingRegistries>>,
    health: Arc<WgpuDeviceHealth>,
    error_attribution_gate: Arc<WgpuErrorAttributionGate>,
}

impl ProgramBindingRealizationState {
    pub(crate) fn new(
        affinity: GpuContextAffinity,
        policy: GpuProgramBindingRealizationPolicy,
        health: Arc<WgpuDeviceHealth>,
        error_attribution_gate: Arc<WgpuErrorAttributionGate>,
    ) -> Self {
        Self {
            affinity,
            policy,
            registries: Arc::new(Mutex::new(ProgramBindingRegistries::default())),
            health,
            error_attribution_gate,
        }
    }

    pub(crate) const fn policy(&self) -> GpuProgramBindingRealizationPolicy {
        self.policy
    }

    pub(crate) fn stats(&self) -> GpuProgramBindingRealizationStats {
        self.registries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .stats(self.policy)
    }

    fn ensure_available(
        &self,
        request: impl Into<String>,
    ) -> Result<(), GpuProgramBindingRealizationError> {
        self.health.ensure_program_binding(request)
    }

    pub(crate) fn validate_execution_program(
        &self,
        record: &Arc<ProgramRealizationRecord>,
    ) -> Result<(), GpuProgramBindingRealizationError> {
        self.validate_execution_record("program", record.affinity(), |registries| {
            registries.contains_program(record)
        })
    }

    pub(crate) fn validate_execution_pipeline_layout(
        &self,
        record: &Arc<PipelineLayoutRealizationRecord>,
    ) -> Result<(), GpuProgramBindingRealizationError> {
        self.validate_execution_record("pipeline layout", record.affinity(), |registries| {
            registries.contains_pipeline_layout(record)
        })
    }

    pub(crate) fn validate_execution_bind_group(
        &self,
        record: &Arc<BindGroupRealizationRecord>,
    ) -> Result<(), GpuProgramBindingRealizationError> {
        self.validate_execution_record("bind group", record.affinity(), |registries| {
            registries.contains_bind_group(record)
        })
    }

    /// Validates and lexically lends exact authoritative bind-group records to private execution.
    /// The backend objects cannot escape this crate-private call boundary as public RunenGPU API.
    pub(crate) fn with_execution_bind_groups<R>(
        &self,
        groups: &[&GpuRealizedBindGroup],
        operation: impl FnOnce(&[&wgpu::BindGroup]) -> R,
    ) -> Result<R, GpuProgramBindingRealizationError> {
        for group in groups {
            self.validate_execution_bind_group(&group.record)?;
        }
        let objects = groups
            .iter()
            .map(|group| &group.record.object)
            .collect::<Vec<_>>();
        Ok(operation(&objects))
    }

    fn validate_execution_record(
        &self,
        request: &'static str,
        observed_affinity: GpuContextAffinity,
        contains: impl FnOnce(&ProgramBindingRegistries) -> bool,
    ) -> Result<(), GpuProgramBindingRealizationError> {
        super::health::validate_program_affinity(self.affinity, request, observed_affinity)?;
        self.ensure_available(request)?;
        let registries = self
            .registries
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if contains(&registries) {
            Ok(())
        } else {
            Err(GpuProgramBindingRealizationError::new(
                GpuProgramBindingRealizationErrorCategory::ExecutionAuthorityViolation,
                request,
                "the execution input is absent from authoritative G4C2 realization",
            ))
        }
    }
}

impl core::fmt::Debug for ProgramBindingRealizationState {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("ProgramBindingRealizationState")
            .field("affinity", &self.affinity)
            .field("stats", &self.stats())
            .finish_non_exhaustive()
    }
}

impl GpuContext {
    /// Returns the explicit shared G4C2 program/binding record policy selected at admission.
    pub fn program_binding_realization_policy(&self) -> GpuProgramBindingRealizationPolicy {
        self.backend.program_binding_realization.policy()
    }

    /// Observes ready-plus-in-flight G4C2 authority counts for this device generation.
    pub fn program_binding_realization_stats(&self) -> GpuProgramBindingRealizationStats {
        self.backend.program_binding_realization.stats()
    }

    /// Realizes one already-admitted canonical WGSL program for this device generation.
    pub async fn realize_program(
        &self,
        descriptor: &GpuProgramDescriptor,
    ) -> Result<GpuRealizedProgram, GpuProgramBindingRealizationError> {
        let request = program_request_name(descriptor);
        loop {
            self.backend
                .program_binding_realization
                .ensure_available(request.clone())?;
            let key = ProgramRequestKey::new(self.affinity(), descriptor.clone());
            match ProgramBindingRegistries::reserve_program(
                &self.backend.program_binding_realization.registries,
                self.backend.program_binding_realization.policy,
                key,
                request.clone(),
            )? {
                Reservation::Ready(record) => return Ok(GpuRealizedProgram::from_record(record)),
                Reservation::Waiter(attempt) => match attempt.wait().await {
                    InFlightOutcome::Complete(outcome) => {
                        return outcome.map(GpuRealizedProgram::from_record);
                    }
                    InFlightOutcome::Abandoned => continue,
                    InFlightOutcome::Pending => {
                        unreachable!("wait never returns a pending attempt")
                    }
                },
                Reservation::Owner(owner) => {
                    let outcome = self.realize_program_owner(descriptor).await;
                    return owner.finish(outcome).map(GpuRealizedProgram::from_record);
                }
            }
        }
    }

    /// Realizes one explicit typed bind-group layout without automatic backend layout inference.
    pub async fn realize_bind_group_layout(
        &self,
        descriptor: &GpuBindGroupLayoutDescriptor,
    ) -> Result<GpuRealizedBindGroupLayout, GpuProgramBindingRealizationError> {
        let request = layout_request_name(descriptor);
        loop {
            self.backend
                .program_binding_realization
                .ensure_available(request.clone())?;
            let key = BindGroupLayoutRequestKey::new(self.affinity(), descriptor.clone());
            match ProgramBindingRegistries::reserve_bind_group_layout(
                &self.backend.program_binding_realization.registries,
                self.backend.program_binding_realization.policy,
                key,
                request.clone(),
            )? {
                Reservation::Ready(record) => {
                    return Ok(GpuRealizedBindGroupLayout::from_record(record));
                }
                Reservation::Waiter(attempt) => match attempt.wait().await {
                    InFlightOutcome::Complete(outcome) => {
                        return outcome.map(GpuRealizedBindGroupLayout::from_record);
                    }
                    InFlightOutcome::Abandoned => continue,
                    InFlightOutcome::Pending => {
                        unreachable!("wait never returns a pending attempt")
                    }
                },
                Reservation::Owner(owner) => {
                    let outcome = self.realize_bind_group_layout_owner(descriptor).await;
                    return owner
                        .finish(outcome)
                        .map(GpuRealizedBindGroupLayout::from_record);
                }
            }
        }
    }

    /// Realizes a pipeline layout by internally realizing its descriptor-owned group layouts.
    pub async fn realize_pipeline_layout(
        &self,
        descriptor: &GpuPipelineLayoutDescriptor,
    ) -> Result<GpuRealizedPipelineLayout, GpuProgramBindingRealizationError> {
        let request = "pipeline layout".to_string();
        loop {
            self.backend
                .program_binding_realization
                .ensure_available(request.clone())?;
            let key = PipelineLayoutRequestKey::new(self.affinity(), descriptor.clone());
            match ProgramBindingRegistries::reserve_pipeline_layout(
                &self.backend.program_binding_realization.registries,
                self.backend.program_binding_realization.policy,
                key,
                request.clone(),
            )? {
                Reservation::Ready(record) => {
                    return Ok(GpuRealizedPipelineLayout::from_record(record));
                }
                Reservation::Waiter(attempt) => match attempt.wait().await {
                    InFlightOutcome::Complete(outcome) => {
                        return outcome.map(GpuRealizedPipelineLayout::from_record);
                    }
                    InFlightOutcome::Abandoned => continue,
                    InFlightOutcome::Pending => {
                        unreachable!("wait never returns a pending attempt")
                    }
                },
                Reservation::Owner(owner) => {
                    let outcome = self.realize_pipeline_layout_owner(descriptor).await;
                    return owner
                        .finish(outcome)
                        .map(GpuRealizedPipelineLayout::from_record);
                }
            }
        }
    }

    /// Validates typed runtime values against this admitted device, resolves their G4C1 resource
    /// records, and realizes one exact bind group against a realized G4C2 layout.
    pub async fn realize_bind_group(
        &self,
        layout: &GpuRealizedBindGroupLayout,
        values: impl IntoIterator<Item = GpuRuntimeBindingValue>,
    ) -> Result<GpuRealizedBindGroup, GpuProgramBindingRealizationError> {
        let request = format!("bind group group={}", layout.descriptor().group());
        super::health::validate_program_affinity(
            self.affinity(),
            request.clone(),
            layout.affinity(),
        )?;
        let validated = GpuValidatedBindGroupBindings::new(layout.descriptor().clone(), values)
            .and_then(|validated| {
                validated
                    .validate_device_facts(&self.runtime_binding_device_facts())
                    .map(|()| validated)
            })
            .map_err(|error| {
                GpuProgramBindingRealizationError::new(
                    GpuProgramBindingRealizationErrorCategory::RuntimeBindingIncompatible,
                    request,
                    error.to_string(),
                )
            })?;
        self.realize_validated_bind_group(layout, validated).await
    }

    /// Realizes a bind group whose logical and admitted-device facts were already validated by
    /// canonical contextual submission preparation. This is crate-private execution plumbing, not
    /// an unchecked public realization path.
    pub(super) async fn realize_validated_bind_group(
        &self,
        layout: &GpuRealizedBindGroupLayout,
        validated: GpuValidatedBindGroupBindings,
    ) -> Result<GpuRealizedBindGroup, GpuProgramBindingRealizationError> {
        let request = format!("bind group group={}", layout.descriptor().group());
        super::health::validate_program_affinity(
            self.affinity(),
            request.clone(),
            layout.affinity(),
        )?;
        if validated.layout() != layout.descriptor() {
            return Err(GpuProgramBindingRealizationError::new(
                GpuProgramBindingRealizationErrorCategory::RuntimeBindingIncompatible,
                request,
                "validated runtime binding layout does not match the realized bind-group layout",
            ));
        }
        let values = validated.values().cloned().collect::<Vec<_>>();
        loop {
            self.backend
                .program_binding_realization
                .ensure_available(request.clone())?;
            let key = BindGroupRequestKey::new(
                self.affinity(),
                layout.descriptor().clone(),
                values.clone(),
            );
            match ProgramBindingRegistries::reserve_bind_group(
                &self.backend.program_binding_realization.registries,
                self.backend.program_binding_realization.policy,
                key,
                request.clone(),
            )? {
                Reservation::Ready(record) => return Ok(GpuRealizedBindGroup::from_record(record)),
                Reservation::Waiter(attempt) => match attempt.wait().await {
                    InFlightOutcome::Complete(outcome) => {
                        return outcome.map(GpuRealizedBindGroup::from_record);
                    }
                    InFlightOutcome::Abandoned => continue,
                    InFlightOutcome::Pending => {
                        unreachable!("wait never returns a pending attempt")
                    }
                },
                Reservation::Owner(owner) => {
                    let outcome = self
                        .realize_bind_group_owner(layout, validated.clone(), values.clone())
                        .await;
                    return owner.finish(outcome).map(GpuRealizedBindGroup::from_record);
                }
            }
        }
    }

    async fn realize_program_owner(
        &self,
        descriptor: &GpuProgramDescriptor,
    ) -> Result<Arc<ProgramRecord>, GpuProgramBindingRealizationError> {
        let label = descriptor.source().identity().diagnostic_label();
        let object = scoped_create(
            &self.backend.device,
            &self.backend.program_binding_realization,
            program_request_name(descriptor),
            GpuProgramBindingRealizationErrorCategory::ShaderValidationPathMismatch,
            Some("canonical WGSL program admission accepted the source before private WGPU realization"),
            || {
                self.backend
                    .device
                    .create_shader_module(ShaderModuleDescriptor {
                        label: Some(label.as_str()),
                        source: ShaderSource::Wgsl(Cow::Borrowed(
                            descriptor.source().canonical_wgsl(),
                        )),
                    })
            },
        )
        .await?;
        Ok(Arc::new(ProgramRecord {
            affinity: self.affinity(),
            descriptor: descriptor.clone(),
            object,
        }))
    }

    async fn realize_bind_group_layout_owner(
        &self,
        descriptor: &GpuBindGroupLayoutDescriptor,
    ) -> Result<Arc<LayoutRecord>, GpuProgramBindingRealizationError> {
        let entries = lowering::layout_entries(self, descriptor)?;
        let object = scoped_create(
            &self.backend.device,
            &self.backend.program_binding_realization,
            layout_request_name(descriptor),
            GpuProgramBindingRealizationErrorCategory::UnexpectedBackendProgramOrBindingValidationRejection,
            None,
            || {
                self.backend
                    .device
                    .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                        label: Some("runengpu-bind-group-layout"),
                        entries: &entries,
                    })
            },
        )
        .await?;
        Ok(Arc::new(LayoutRecord {
            affinity: self.affinity(),
            descriptor: descriptor.clone(),
            object,
        }))
    }

    async fn realize_pipeline_layout_owner(
        &self,
        descriptor: &GpuPipelineLayoutDescriptor,
    ) -> Result<Arc<PipelineRecord>, GpuProgramBindingRealizationError> {
        let positional_descriptors = positional_pipeline_layout_groups(
            descriptor,
            self.backend.device.limits().max_bind_groups,
        )?;
        let mut groups = Vec::with_capacity(positional_descriptors.len());
        for group in &positional_descriptors {
            groups.push(self.realize_bind_group_layout(group).await?.record);
        }
        let layout_refs = groups
            .iter()
            .map(|group| Some(&group.object))
            .collect::<Vec<_>>();
        let object = scoped_create(
            &self.backend.device,
            &self.backend.program_binding_realization,
            "pipeline layout",
            GpuProgramBindingRealizationErrorCategory::UnexpectedBackendProgramOrBindingValidationRejection,
            None,
            || {
                self.backend.device.create_pipeline_layout(&PipelineLayoutDescriptor {
                    label: Some("runengpu-pipeline-layout"),
                    bind_group_layouts: &layout_refs,
                    immediate_size: 0,
                })
            },
        )
        .await?;
        Ok(Arc::new(PipelineRecord {
            affinity: self.affinity(),
            descriptor: descriptor.clone(),
            object,
            groups,
        }))
    }

    async fn realize_bind_group_owner(
        &self,
        layout: &GpuRealizedBindGroupLayout,
        validated: GpuValidatedBindGroupBindings,
        values: Vec<GpuRuntimeBindingValue>,
    ) -> Result<Arc<GroupRecord>, GpuProgramBindingRealizationError> {
        let resolved = resolve_binding_resources(self, &validated)?;
        let static_values = static_bind_group_values(values);
        let object = scoped_create(
            &self.backend.device,
            &self.backend.program_binding_realization,
            format!("bind group group={}", layout.descriptor().group()),
            GpuProgramBindingRealizationErrorCategory::UnexpectedBackendProgramOrBindingValidationRejection,
            None,
            || resolved.with_entries(|entries| {
                self.backend.device.create_bind_group(&BindGroupDescriptor {
                    label: Some("runengpu-bind-group"),
                    layout: &layout.record.object,
                    entries,
                })
            }),
        )
        .await?;
        Ok(Arc::new(GroupRecord {
            affinity: self.affinity(),
            layout: Arc::clone(&layout.record),
            static_values,
            object,
            resources: resolved.into_dependencies(),
        }))
    }
}

/// WGPU pipeline-layout arrays are indexed by bind-group number, while G4B deliberately permits
/// an ordered sparse set of group descriptors. Materialize only the deterministic empty slots
/// needed to preserve those accepted group indices at the backend boundary.
fn positional_pipeline_layout_groups(
    descriptor: &GpuPipelineLayoutDescriptor,
    max_bind_groups: u32,
) -> Result<Vec<GpuBindGroupLayoutDescriptor>, GpuProgramBindingRealizationError> {
    let Some(highest_group) = descriptor
        .groups()
        .map(GpuBindGroupLayoutDescriptor::group)
        .max()
    else {
        return Ok(Vec::new());
    };
    let required_slots = highest_group.checked_add(1).ok_or_else(|| {
        GpuProgramBindingRealizationError::new(
            GpuProgramBindingRealizationErrorCategory::LayoutDescriptorInvalid,
            "construct pipeline layout",
            "the highest bind-group index cannot be represented as a positional WGPU layout slot count",
        )
    })?;
    if required_slots > max_bind_groups {
        return Err(GpuProgramBindingRealizationError::new(
            GpuProgramBindingRealizationErrorCategory::LayoutDescriptorInvalid,
            "construct pipeline layout",
            format!(
                "highest bind-group index {highest_group} requires {required_slots} positional WGPU layout slots, but the admitted device exposes only {max_bind_groups}"
            ),
        ));
    }

    let mut positional = Vec::with_capacity(required_slots as usize);
    let mut next_group = 0;
    for group in descriptor.groups() {
        while next_group < group.group() {
            positional.push(
                GpuBindGroupLayoutDescriptor::new(
                    next_group,
                    std::iter::empty::<GpuBindingDeclaration>(),
                )
                .expect("an empty G4B bind-group layout is always structurally valid"),
            );
            next_group += 1;
        }
        positional.push(group.clone());
        next_group += 1;
    }
    Ok(positional)
}

/// Pushes all required scopes, synchronously creates one object, pops every scope, releases the
/// non-reentrant device gate, then awaits completion before publication.
async fn scoped_create<T>(
    device: &wgpu::Device,
    realization: &ProgramBindingRealizationState,
    request: impl Into<String>,
    validation_category: GpuProgramBindingRealizationErrorCategory,
    validation_path_evidence: Option<&str>,
    create: impl FnOnce() -> T,
) -> Result<T, GpuProgramBindingRealizationError> {
    let request = request.into();
    realization.health.ensure_program_binding(request.clone())?;
    let (candidate, validation, out_of_memory, internal) = {
        let _gate = realization.error_attribution_gate.acquire();
        let internal_scope = device.push_error_scope(ErrorFilter::Internal);
        let out_of_memory_scope = device.push_error_scope(ErrorFilter::OutOfMemory);
        let validation_scope = device.push_error_scope(ErrorFilter::Validation);
        let candidate = create();
        // Reverse stack order, all dispatched before releasing the shared attribution gate.
        let validation = validation_scope.pop();
        let out_of_memory = out_of_memory_scope.pop();
        let internal = internal_scope.pop();
        (candidate, validation, out_of_memory, internal)
    };
    let validation = validation.await;
    let out_of_memory = out_of_memory.await;
    let internal = internal.await;
    let validation_detail = validation.map(|error| format!("Validation scope: {error}"));
    let out_of_memory_detail = out_of_memory.map(|error| format!("OutOfMemory scope: {error}"));
    let internal_detail = internal.map(|error| format!("Internal scope: {error}"));
    let health_fault = realization.health.terminal_fault();

    // A concurrently observed loss is terminal even when an error scope also reported a lower
    // precedence failure. It must not publish or report a stale validation/OOM result instead.
    if let Some(fault) = health_fault
        .as_ref()
        .filter(|fault| fault.class == super::health::WgpuDeviceFaultClass::InternalOrDeviceLost)
    {
        return Err(scoped_failure(
            GpuProgramBindingRealizationErrorCategory::ContextOrDeviceUnavailableOrLost,
            request,
            format!("shared device health: {}", fault.detail),
            [
                fault.secondary_detail.clone(),
                internal_detail,
                out_of_memory_detail,
                validation_detail,
            ],
        ));
    }
    if let Some(detail) = internal_detail {
        realization
            .health
            .mark_scoped_internal(format!("scoped WGPU internal error: {detail}"));
        return Err(scoped_failure(
            GpuProgramBindingRealizationErrorCategory::ContextOrDeviceUnavailableOrLost,
            request,
            detail,
            [
                out_of_memory_detail,
                validation_detail,
                health_fault.as_ref().map(shared_health_evidence),
            ],
        ));
    }
    if let Some(fault) = health_fault
        .as_ref()
        .filter(|fault| fault.class == super::health::WgpuDeviceFaultClass::OutOfMemory)
    {
        return Err(scoped_failure(
            GpuProgramBindingRealizationErrorCategory::BackendResourceExhaustion,
            request,
            format!("shared device health: {}", fault.detail),
            [
                fault.secondary_detail.clone(),
                out_of_memory_detail,
                validation_detail,
            ],
        ));
    }
    if let Some(detail) = out_of_memory_detail {
        return Err(scoped_failure(
            GpuProgramBindingRealizationErrorCategory::BackendResourceExhaustion,
            request,
            detail,
            [
                validation_detail,
                health_fault.as_ref().map(shared_health_evidence),
            ],
        ));
    }
    if let Some(detail) = validation_detail {
        return Err(scoped_failure(
            validation_category,
            request,
            detail,
            [
                validation_path_evidence.map(str::to_owned),
                health_fault.as_ref().map(shared_health_evidence),
            ],
        ));
    }
    realization.health.ensure_program_binding(request)?;
    Ok(candidate)
}

fn shared_health_evidence(fault: &super::health::WgpuDeviceFaultEvidence) -> String {
    match fault.secondary_detail.as_deref() {
        Some(secondary) => format!("shared device health: {}; {secondary}", fault.detail),
        None => format!("shared device health: {}", fault.detail),
    }
}

fn scoped_failure(
    category: GpuProgramBindingRealizationErrorCategory,
    request: String,
    detail: impl Into<String>,
    secondary_details: impl IntoIterator<Item = Option<String>>,
) -> GpuProgramBindingRealizationError {
    let secondary_detail = secondary_details
        .into_iter()
        .flatten()
        .filter(|detail| !detail.trim().is_empty())
        .take(3)
        .collect::<Vec<_>>()
        .join("; ");
    let error = GpuProgramBindingRealizationError::new(category, request, detail);
    if secondary_detail.is_empty() {
        error
    } else {
        error.with_secondary_detail(secondary_detail)
    }
}

enum ResolvedBindingResources {
    Buffers {
        binding: u32,
        resources: Vec<Arc<super::resource_realization::BufferRealizationRecord>>,
        offsets: Vec<u64>,
        sizes: Vec<wgpu::BufferSize>,
    },
    TextureViews {
        binding: u32,
        resources: Vec<Arc<super::resource_realization::TextureViewRealizationRecord>>,
    },
    Samplers {
        binding: u32,
        resources: Vec<Arc<super::resource_realization::SamplerRealizationRecord>>,
    },
}

struct ResolvedBindGroup {
    values: Vec<ResolvedBindingResources>,
}

impl ResolvedBindGroup {
    fn with_entries<T>(&self, call: impl FnOnce(&[BindGroupEntry<'_>]) -> T) -> T {
        let mut buffer_arrays = Vec::<Vec<BufferBinding<'_>>>::new();
        let mut texture_arrays = Vec::<Vec<&wgpu::TextureView>>::new();
        let mut sampler_arrays = Vec::<Vec<&wgpu::Sampler>>::new();
        let mut references = Vec::with_capacity(self.values.len());
        for value in &self.values {
            match value {
                ResolvedBindingResources::Buffers {
                    resources,
                    offsets,
                    sizes,
                    binding,
                } => {
                    let index = buffer_arrays.len();
                    buffer_arrays.push(
                        resources
                            .iter()
                            .zip(offsets)
                            .zip(sizes)
                            .map(|((resource, offset), size)| BufferBinding {
                                buffer: &resource.object,
                                offset: *offset,
                                size: Some(*size),
                            })
                            .collect(),
                    );
                    references.push((*binding, 0_u8, index));
                }
                ResolvedBindingResources::TextureViews { binding, resources } => {
                    let index = texture_arrays.len();
                    texture_arrays
                        .push(resources.iter().map(|resource| &resource.object).collect());
                    references.push((*binding, 1_u8, index));
                }
                ResolvedBindingResources::Samplers { binding, resources } => {
                    let index = sampler_arrays.len();
                    sampler_arrays
                        .push(resources.iter().map(|resource| &resource.object).collect());
                    references.push((*binding, 2_u8, index));
                }
            }
        }
        let entries = references
            .into_iter()
            .map(|(binding, kind, index)| {
                let resource = match kind {
                    0 if buffer_arrays[index].len() == 1 => {
                        BindingResource::Buffer(buffer_arrays[index][0].clone())
                    }
                    0 => BindingResource::BufferArray(&buffer_arrays[index]),
                    1 if texture_arrays[index].len() == 1 => {
                        BindingResource::TextureView(texture_arrays[index][0])
                    }
                    1 => BindingResource::TextureViewArray(&texture_arrays[index]),
                    2 if sampler_arrays[index].len() == 1 => {
                        BindingResource::Sampler(sampler_arrays[index][0])
                    }
                    2 => BindingResource::SamplerArray(&sampler_arrays[index]),
                    _ => unreachable!("resolved binding kinds are exhaustive"),
                };
                BindGroupEntry { binding, resource }
            })
            .collect::<Vec<_>>();
        call(&entries)
    }

    fn into_dependencies(self) -> Vec<BindGroupResourceDependency> {
        self.values
            .into_iter()
            .flat_map(|value| match value {
                ResolvedBindingResources::Buffers { resources, .. } => resources
                    .into_iter()
                    .map(BindGroupResourceDependency::Buffer)
                    .collect::<Vec<_>>(),
                ResolvedBindingResources::TextureViews { resources, .. } => resources
                    .into_iter()
                    .map(BindGroupResourceDependency::TextureView)
                    .collect::<Vec<_>>(),
                ResolvedBindingResources::Samplers { resources, .. } => resources
                    .into_iter()
                    .map(BindGroupResourceDependency::Sampler)
                    .collect::<Vec<_>>(),
            })
            .collect()
    }
}

fn resolve_binding_resources(
    context: &GpuContext,
    validated: &GpuValidatedBindGroupBindings,
) -> Result<ResolvedBindGroup, GpuProgramBindingRealizationError> {
    let mut values = Vec::with_capacity(validated.values().len());
    for value in validated.values() {
        let binding = value.key().binding();
        let mut buffers = Vec::new();
        let mut offsets = Vec::new();
        let mut sizes = Vec::new();
        let mut views = Vec::new();
        let mut samplers = Vec::new();
        for resource in value.resources() {
            match resource {
                GpuRuntimeBindingResource::Buffer(binding) => {
                    let realized = context
                        .realize_buffer(binding.handle())
                        .map_err(resource_failure)?;
                    buffers.push(realized.record);
                    offsets.push(binding.offset());
                    sizes.push(
                        wgpu::BufferSize::new(binding.size().get())
                            .expect("runtime binding sizes are nonzero"),
                    );
                }
                GpuRuntimeBindingResource::TextureView(binding) => {
                    let parent = context
                        .realize_texture(binding.handle().descriptor().texture())
                        .map_err(resource_failure)?;
                    let realized = context
                        .realize_texture_view(binding.handle(), &parent)
                        .map_err(resource_failure)?;
                    views.push(realized.record);
                }
                GpuRuntimeBindingResource::Sampler(handle) => {
                    let realized = context.realize_sampler(handle).map_err(resource_failure)?;
                    samplers.push(realized.record);
                }
            }
        }
        if !buffers.is_empty() {
            values.push(ResolvedBindingResources::Buffers {
                binding,
                resources: buffers,
                offsets,
                sizes,
            });
        } else if !views.is_empty() {
            values.push(ResolvedBindingResources::TextureViews {
                binding,
                resources: views,
            });
        } else if !samplers.is_empty() {
            values.push(ResolvedBindingResources::Samplers {
                binding,
                resources: samplers,
            });
        } else {
            return Err(GpuProgramBindingRealizationError::new(
                GpuProgramBindingRealizationErrorCategory::BindingValueMismatch,
                format!("bind group binding={binding}"),
                "validated runtime binding retained no G4C1 resource",
            ));
        }
    }
    Ok(ResolvedBindGroup { values })
}

fn resource_failure(
    error: crate::GpuResourceRealizationError,
) -> GpuProgramBindingRealizationError {
    use crate::GpuResourceRealizationErrorCategory as ResourceCategory;

    let category = match error.category() {
        ResourceCategory::ForeignContext => GpuProgramBindingRealizationErrorCategory::ForeignContext,
        ResourceCategory::StaleDeviceGeneration => {
            GpuProgramBindingRealizationErrorCategory::StaleDeviceGeneration
        }
        ResourceCategory::UnknownLogicalResource
        | ResourceCategory::DescriptorChangedForIdentity
        | ResourceCategory::ResourceKindMismatch
        | ResourceCategory::RequirementNotAdmitted
        | ResourceCategory::FormatOrAlignmentNotAdmitted
        | ResourceCategory::ImportGenerationMismatch
        | ResourceCategory::ImportSourceUnavailable => {
            GpuProgramBindingRealizationErrorCategory::RuntimeBindingIncompatible
        }
        ResourceCategory::RegistryCapacityExceeded => {
            GpuProgramBindingRealizationErrorCategory::RegistryCapacityExceeded
        }
        ResourceCategory::CacheRejected => GpuProgramBindingRealizationErrorCategory::CacheRejected,
        ResourceCategory::UnexpectedBackendValidationRejection => {
            GpuProgramBindingRealizationErrorCategory::UnexpectedBackendProgramOrBindingValidationRejection
        }
        ResourceCategory::BackendResourceExhaustion => {
            GpuProgramBindingRealizationErrorCategory::BackendResourceExhaustion
        }
        ResourceCategory::ContextOrDeviceUnavailableOrLost => {
            GpuProgramBindingRealizationErrorCategory::ContextOrDeviceUnavailableOrLost
        }
        ResourceCategory::ExecutionAuthorityViolation => {
            GpuProgramBindingRealizationErrorCategory::ExecutionAuthorityViolation
        }
    };

    GpuProgramBindingRealizationError::new(
        category,
        "realize G4C1 runtime binding resource",
        error.to_string(),
    )
}

fn program_request_name(descriptor: &GpuProgramDescriptor) -> String {
    descriptor.source().identity().diagnostic_label()
}

fn layout_request_name(descriptor: &GpuBindGroupLayoutDescriptor) -> String {
    format!("bind-group layout group={}", descriptor.group())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        GpuResourceRealizationError, GpuResourceRealizationErrorCategory as ResourceCategory,
    };

    fn empty_layout(group: u32) -> GpuBindGroupLayoutDescriptor {
        GpuBindGroupLayoutDescriptor::new(group, std::iter::empty::<GpuBindingDeclaration>())
            .expect("empty bind-group layouts are structurally valid")
    }

    #[test]
    fn sparse_pipeline_layout_preserves_group_indices_with_empty_lower_slots() {
        let declared_group_one = empty_layout(1);
        let descriptor = GpuPipelineLayoutDescriptor::new([declared_group_one.clone()])
            .expect("a sparse ordered G4B layout is valid");

        let positional = positional_pipeline_layout_groups(&descriptor, 4)
            .expect("the admitted WGPU slot range should accept group one");

        assert_eq!(positional.len(), 2);
        assert_eq!(positional[0].group(), 0);
        assert_eq!(positional[0].bindings().len(), 0);
        assert_eq!(positional[1], declared_group_one);
    }

    #[test]
    fn sparse_pipeline_layout_rejects_group_indices_outside_admitted_wgpu_slots() {
        let descriptor = GpuPipelineLayoutDescriptor::new([empty_layout(4)])
            .expect("the G4B descriptor itself permits an ordered group four layout");

        let error = positional_pipeline_layout_groups(&descriptor, 4)
            .expect_err("group four requires a fifth positional WGPU slot");

        assert_eq!(
            error.category(),
            GpuProgramBindingRealizationErrorCategory::LayoutDescriptorInvalid
        );
    }

    #[test]
    fn g4c1_resource_failures_translate_without_losing_actionable_failure_class() {
        let cases = [
            (
                ResourceCategory::ForeignContext,
                GpuProgramBindingRealizationErrorCategory::ForeignContext,
            ),
            (
                ResourceCategory::StaleDeviceGeneration,
                GpuProgramBindingRealizationErrorCategory::StaleDeviceGeneration,
            ),
            (
                ResourceCategory::UnknownLogicalResource,
                GpuProgramBindingRealizationErrorCategory::RuntimeBindingIncompatible,
            ),
            (
                ResourceCategory::DescriptorChangedForIdentity,
                GpuProgramBindingRealizationErrorCategory::RuntimeBindingIncompatible,
            ),
            (
                ResourceCategory::ResourceKindMismatch,
                GpuProgramBindingRealizationErrorCategory::RuntimeBindingIncompatible,
            ),
            (
                ResourceCategory::RequirementNotAdmitted,
                GpuProgramBindingRealizationErrorCategory::RuntimeBindingIncompatible,
            ),
            (
                ResourceCategory::FormatOrAlignmentNotAdmitted,
                GpuProgramBindingRealizationErrorCategory::RuntimeBindingIncompatible,
            ),
            (
                ResourceCategory::ImportGenerationMismatch,
                GpuProgramBindingRealizationErrorCategory::RuntimeBindingIncompatible,
            ),
            (
                ResourceCategory::ImportSourceUnavailable,
                GpuProgramBindingRealizationErrorCategory::RuntimeBindingIncompatible,
            ),
            (
                ResourceCategory::RegistryCapacityExceeded,
                GpuProgramBindingRealizationErrorCategory::RegistryCapacityExceeded,
            ),
            (
                ResourceCategory::CacheRejected,
                GpuProgramBindingRealizationErrorCategory::CacheRejected,
            ),
            (
                ResourceCategory::UnexpectedBackendValidationRejection,
                GpuProgramBindingRealizationErrorCategory::UnexpectedBackendProgramOrBindingValidationRejection,
            ),
            (
                ResourceCategory::BackendResourceExhaustion,
                GpuProgramBindingRealizationErrorCategory::BackendResourceExhaustion,
            ),
            (
                ResourceCategory::ContextOrDeviceUnavailableOrLost,
                GpuProgramBindingRealizationErrorCategory::ContextOrDeviceUnavailableOrLost,
            ),
            (
                ResourceCategory::ExecutionAuthorityViolation,
                GpuProgramBindingRealizationErrorCategory::ExecutionAuthorityViolation,
            ),
        ];

        for (resource_category, expected) in cases {
            let source = GpuResourceRealizationError::new(
                resource_category,
                None,
                "representative G4C1 failure evidence",
            );
            let translated = resource_failure(source);

            assert_eq!(translated.category(), expected);
            assert_eq!(
                translated.request(),
                Some("realize G4C1 runtime binding resource")
            );
            assert!(
                translated
                    .detail()
                    .is_some_and(|detail| detail.contains("representative G4C1 failure evidence")),
                "translated G4C2 failure must retain bounded G4C1 evidence"
            );
        }
    }
}
