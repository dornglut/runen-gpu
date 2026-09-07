use super::{
    GpuAdmittedProgramSource, GpuBindingLayoutRefinement, GpuBlendMode, GpuBufferDescriptor,
    GpuBufferHandle, GpuColorTargetStateDescriptor, GpuColorWriteMask,
    GpuComputePipelineDescriptor, GpuContext, GpuEntryPointName, GpuFragmentOutputStateDescriptor,
    GpuMultisampleStateDescriptor, GpuPipelineConfiguration, GpuPrimitiveStateDescriptor,
    GpuProgramContractError, GpuProgramDescriptor, GpuProgramSourceCause, GpuProgramSourceError,
    GpuProgramSourceIdentity, GpuProgramSourceKey, GpuProgramSourceOwnerId,
    GpuProgramSourceProvenance, GpuProgramSourceRegistry, GpuProgramSourceRevision,
    GpuQuerySetDescriptor, GpuQuerySetHandle, GpuRenderEntryPoints, GpuRenderPipelineDescriptor,
    GpuRenderPipelineStateDescriptor, GpuResourceDescriptorError, GpuResourceLabel,
    GpuSamplerDescriptor, GpuSamplerHandle, GpuSubmission, GpuSubmissionPreparationError,
    GpuSubmissionRejectionReason, GpuTextureDescriptor, GpuTextureFormat, GpuTextureHandle,
    GpuTextureViewDescriptor, GpuTextureViewHandle, GpuVertexInputStateDescriptor,
    GpuWorkAuthoringError, GpuWorkFragment, GpuWorkFragmentBuilder, GpuWorkGraphError,
    GpuWorkNodeId, GpuWorkOperation, GpuWorkResourceIdAllocationError, GpuWorkResourceIdAllocator,
};
use core::fmt;

const STATIC_WGSL_PROVENANCE_PRODUCER: &str = "runengpu-static-wgsl";

/// Rejection from the ordinary build-and-submit path.
///
/// The variants preserve the existing label, graph, execution-preparation, and
/// submission-rejection authorities rather than flattening them into one
/// backend-specific diagnostic.
#[derive(Debug)]
pub enum GpuWorkSubmissionError {
    Label(GpuResourceDescriptorError),
    GraphPreparation(GpuWorkGraphError),
    SubmissionPreparation(GpuSubmissionPreparationError),
    SubmissionRejected(GpuSubmissionRejectionReason),
}

impl fmt::Display for GpuWorkSubmissionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Label(error) => error.fmt(formatter),
            Self::GraphPreparation(error) => error.fmt(formatter),
            Self::SubmissionPreparation(error) => error.fmt(formatter),
            Self::SubmissionRejected(reason) => write!(
                formatter,
                "GPU submission rejected ({:?}): {}",
                reason.kind(),
                reason.detail()
            ),
        }
    }
}

impl std::error::Error for GpuWorkSubmissionError {}

impl From<GpuResourceDescriptorError> for GpuWorkSubmissionError {
    fn from(error: GpuResourceDescriptorError) -> Self {
        Self::Label(error)
    }
}

impl From<GpuWorkGraphError> for GpuWorkSubmissionError {
    fn from(error: GpuWorkGraphError) -> Self {
        Self::GraphPreparation(error)
    }
}

impl From<GpuSubmissionPreparationError> for GpuWorkSubmissionError {
    fn from(error: GpuSubmissionPreparationError) -> Self {
        Self::SubmissionPreparation(error)
    }
}

/// Admits one complete static/embedded WGSL source set through the canonical
/// bounded source-consistency registry without making ordinary callers manage
/// owner IDs, registry capacities, identity assembly, or provenance records.
///
/// Source key, nonzero revision, and canonical WGSL remain explicit semantic
/// inputs. Each call is one complete source-owner set. Dynamic/reloadable source
/// owners that need later admissions or explicit provenance should use
/// [`GpuProgramSourceRegistry`] directly.
pub fn admit_static_wgsl_sources<const N: usize>(
    sources: [(&'static str, u64, &'static str); N],
) -> Result<[GpuAdmittedProgramSource; N], GpuProgramSourceError> {
    let mut total_source_bytes = 0usize;
    let mut checked = Vec::with_capacity(N);

    for (key, revision, canonical_wgsl) in sources {
        let key = GpuProgramSourceKey::new(key)?;
        let revision = GpuProgramSourceRevision::try_from_raw(revision)?;
        total_source_bytes = total_source_bytes
            .checked_add(canonical_wgsl.len())
            .ok_or_else(|| {
                GpuProgramSourceError::invalid(
                    "admit static GPU program sources",
                    "source byte total overflow",
                    GpuProgramSourceCause::SourceAdmissionCapacityExceeded,
                    "reduce the static source set",
                )
            })?;
        checked.push((key, revision, canonical_wgsl));
    }

    // Use the exact complete-set bounds. `max(1)` exists only so an all-empty
    // invalid set reaches canonical WGSL validation instead of being misclassified
    // as a zero-byte registry policy error.
    let mut registry = GpuProgramSourceRegistry::new(N, total_source_bytes.max(1))?;
    let owner = GpuProgramSourceOwnerId::allocate()?;
    let provenance = GpuProgramSourceProvenance::new(STATIC_WGSL_PROVENANCE_PRODUCER, None)?;
    let mut admitted = Vec::with_capacity(N);
    for (key, revision, canonical_wgsl) in checked {
        admitted.push(registry.admit_wgsl(
            GpuProgramSourceIdentity::new(owner, key, revision),
            canonical_wgsl,
            provenance.clone(),
        )?);
    }

    Ok(admitted
        .try_into()
        .unwrap_or_else(|_| unreachable!("static source admission preserves source count")))
}

impl GpuComputePipelineDescriptor {
    /// Constructs the ordinary unrefined compute-pipeline case from one admitted
    /// source and one compute entry point.
    ///
    /// Binding-layout refinements, specialization values, and additional capability
    /// requirements are intentionally absent here. Use [`GpuProgramDescriptor::new`]
    /// plus [`GpuComputePipelineDescriptor::new`] when those semantics are material.
    pub fn ordinary(
        source: GpuAdmittedProgramSource,
        entry_point: impl AsRef<str>,
    ) -> Result<Self, GpuProgramContractError> {
        let entry_point = GpuEntryPointName::new(entry_point.as_ref())?;
        let program = GpuProgramDescriptor::new(
            source,
            [entry_point.clone()],
            std::iter::empty::<GpuBindingLayoutRefinement>(),
        )?;
        Self::new(program, entry_point, GpuPipelineConfiguration::default())
    }
}

impl GpuRenderPipelineDescriptor {
    /// Constructs the ordinary single-color render-pipeline case from one admitted
    /// source, vertex/fragment entry points, and the target format.
    ///
    /// The constrained defaults are no host vertex buffers, replacement blending,
    /// full color writes, triangle-list primitive state with no culling, no depth,
    /// single-sample rendering, no binding-layout refinements, and default pipeline
    /// configuration. Canonical stage-IO validation still rejects shaders that do
    /// not match those choices. Use the explicit program/state/pipeline constructors
    /// when any of those semantics are material.
    pub fn ordinary_color(
        source: GpuAdmittedProgramSource,
        vertex_entry_point: impl AsRef<str>,
        fragment_entry_point: impl AsRef<str>,
        format: GpuTextureFormat,
    ) -> Result<Self, GpuProgramContractError> {
        let vertex_entry_point = GpuEntryPointName::new(vertex_entry_point.as_ref())?;
        let fragment_entry_point = GpuEntryPointName::new(fragment_entry_point.as_ref())?;
        let program = GpuProgramDescriptor::new(
            source,
            [vertex_entry_point.clone(), fragment_entry_point.clone()],
            std::iter::empty::<GpuBindingLayoutRefinement>(),
        )?;
        let state = GpuRenderPipelineStateDescriptor::new(
            GpuVertexInputStateDescriptor::new([])?,
            Some(GpuFragmentOutputStateDescriptor::new([
                GpuColorTargetStateDescriptor::new(
                    format,
                    GpuBlendMode::Replace,
                    GpuColorWriteMask::ALL,
                )?,
            ])),
            GpuPrimitiveStateDescriptor::default(),
            None,
            GpuMultisampleStateDescriptor::default(),
        )?;
        Self::new(
            program,
            GpuRenderEntryPoints::new(vertex_entry_point, Some(fragment_entry_point)),
            state,
            GpuPipelineConfiguration::default(),
        )
    }
}

impl GpuContext {
    /// Prepares, validates, and submits ordinary authored GPU work through the
    /// same canonical authorities as the explicit advanced path.
    ///
    /// Ordinary callers provide a diagnostic label as text; RunenGPU preserves the canonical
    /// checked label and graph-preparation authorities internally.
    pub async fn submit_work(
        &self,
        label: impl AsRef<str>,
        fragments: impl IntoIterator<Item = GpuWorkFragment>,
    ) -> Result<GpuSubmission, GpuWorkSubmissionError> {
        let label = GpuResourceLabel::new(label.as_ref())?;
        let graph = self.prepare_work_graph(label, fragments)?;
        let prepared = self.prepare_submission(graph).await?;
        self.submit_prepared(prepared).map_err(|rejected| {
            let (_, reason) = rejected.into_parts();
            GpuWorkSubmissionError::SubmissionRejected(reason)
        })
    }
}

impl From<super::GpuComputeOperation> for GpuWorkOperation {
    fn from(operation: super::GpuComputeOperation) -> Self {
        Self::Compute(operation)
    }
}

impl From<super::GpuRenderOperation> for GpuWorkOperation {
    fn from(operation: super::GpuRenderOperation) -> Self {
        Self::Render(operation)
    }
}

impl From<super::GpuCopyOperation> for GpuWorkOperation {
    fn from(operation: super::GpuCopyOperation) -> Self {
        Self::Copy(operation)
    }
}

impl From<super::GpuClearOperation> for GpuWorkOperation {
    fn from(operation: super::GpuClearOperation) -> Self {
        Self::Clear(operation)
    }
}

impl From<super::GpuQueryResolveOperation> for GpuWorkOperation {
    fn from(operation: super::GpuQueryResolveOperation) -> Self {
        Self::Resolve(operation)
    }
}

impl From<super::GpuPresentOperation> for GpuWorkOperation {
    fn from(operation: super::GpuPresentOperation) -> Self {
        Self::Present(operation)
    }
}

impl From<super::GpuUploadOperation> for GpuWorkOperation {
    fn from(operation: super::GpuUploadOperation) -> Self {
        Self::Upload(operation)
    }
}

impl From<super::GpuReadbackOperation> for GpuWorkOperation {
    fn from(operation: super::GpuReadbackOperation) -> Self {
        Self::Readback(operation)
    }
}

impl GpuWorkFragmentBuilder {
    /// Adds ordinary checked operation work through the existing node authority.
    ///
    /// Resources referenced by the typed operation are registered lexically from
    /// its derived accesses. Explicit resource declaration remains available for
    /// fragment inputs, imports, outputs, and advanced caller-declared accesses.
    /// Additional capability requirements, non-default execution preference, and
    /// explicit provenance remain available through [`GpuWorkFragmentBuilder::add_node`].
    pub fn operation<L>(
        &mut self,
        label: L,
        operation: impl Into<GpuWorkOperation>,
    ) -> Result<GpuWorkNodeId, GpuWorkAuthoringError>
    where
        L: AsRef<str>,
    {
        self.add_checked_lexical_operation(label, operation.into())
    }
}

/// Owns one bounded logical RunenGPU resource scope for ordinary authoring.
///
/// Related resources allocated through one scope retain the accepted owner-scoped
/// identity semantics without making callers administer `GpuWorkResourceId`
/// values directly. This is logical resource ownership only; it is not a GPU
/// context, backend-realization registry, lifecycle owner, or reconstruction
/// authority.
#[derive(Debug, Default)]
pub struct GpuResourceScope {
    identities: GpuWorkResourceIdAllocator,
}

impl GpuResourceScope {
    pub const fn new() -> Self {
        Self {
            identities: GpuWorkResourceIdAllocator::new(),
        }
    }

    pub fn buffer(
        &mut self,
        descriptor: GpuBufferDescriptor,
    ) -> Result<GpuBufferHandle, GpuWorkResourceIdAllocationError> {
        self.identities.allocate_buffer_handle(descriptor)
    }

    pub fn texture(
        &mut self,
        descriptor: GpuTextureDescriptor,
    ) -> Result<GpuTextureHandle, GpuWorkResourceIdAllocationError> {
        self.identities.allocate_texture_handle(descriptor)
    }

    pub fn texture_view(
        &mut self,
        descriptor: GpuTextureViewDescriptor,
    ) -> Result<GpuTextureViewHandle, GpuWorkResourceIdAllocationError> {
        self.identities.allocate_texture_view_handle(descriptor)
    }

    pub fn sampler(
        &mut self,
        descriptor: GpuSamplerDescriptor,
    ) -> Result<GpuSamplerHandle, GpuWorkResourceIdAllocationError> {
        self.identities.allocate_sampler_handle(descriptor)
    }

    pub fn query_set(
        &mut self,
        descriptor: GpuQuerySetDescriptor,
    ) -> Result<GpuQuerySetHandle, GpuWorkResourceIdAllocationError> {
        self.identities.allocate_query_set_handle(descriptor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GpuProgramContractCause;

    #[test]
    fn static_wgsl_sources_share_one_owner_and_preserve_semantic_inputs() {
        let [compute, render] = admit_static_wgsl_sources([
            (
                "proof.compute",
                3,
                "@compute @workgroup_size(1) fn main() {}",
            ),
            (
                "proof.render",
                7,
                "@vertex fn main() -> @builtin(position) vec4<f32> { return vec4<f32>(); }",
            ),
        ])
        .unwrap();

        assert_eq!(compute.identity().owner(), render.identity().owner());
        assert_eq!(compute.identity().key().as_str(), "proof.compute");
        assert_eq!(compute.identity().revision().get(), 3);
        assert_eq!(render.identity().key().as_str(), "proof.render");
        assert_eq!(render.identity().revision().get(), 7);
        assert_eq!(
            compute.provenance().producer(),
            STATIC_WGSL_PROVENANCE_PRODUCER
        );
        assert_eq!(
            render.provenance().producer(),
            STATIC_WGSL_PROVENANCE_PRODUCER
        );
    }

    #[test]
    fn static_wgsl_sources_retain_registry_conflict_authority() {
        let error = admit_static_wgsl_sources([
            ("proof.same", 1, "@compute @workgroup_size(1) fn first() {}"),
            (
                "proof.same",
                1,
                "@compute @workgroup_size(1) fn second() {}",
            ),
        ])
        .unwrap_err();

        assert_eq!(error.cause(), GpuProgramSourceCause::SourceRevisionConflict);
    }

    #[test]
    fn static_wgsl_sources_reuse_equal_revision_records() {
        let [first, second] = admit_static_wgsl_sources([
            ("proof.same", 2, "@compute @workgroup_size(1) fn main() {}"),
            ("proof.same", 2, "@compute @workgroup_size(1) fn main() {}"),
        ])
        .unwrap();

        assert!(first.is_same_record(&second));
    }

    #[test]
    fn static_wgsl_empty_source_uses_canonical_source_validation() {
        let error = admit_static_wgsl_sources([("proof.empty", 1, "")]).unwrap_err();
        assert_eq!(error.cause(), GpuProgramSourceCause::EmptyCanonicalWgsl);
    }

    #[test]
    fn ordinary_compute_pipeline_lowers_through_canonical_program_and_pipeline() {
        let [source] = admit_static_wgsl_sources([(
            "proof.pipeline.compute",
            1,
            "@compute @workgroup_size(1) fn main() {}",
        )])
        .unwrap();

        let pipeline = GpuComputePipelineDescriptor::ordinary(source.clone(), "main").unwrap();

        assert!(pipeline.program().source().is_same_record(&source));
        assert_eq!(pipeline.entry_point().as_str(), "main");
    }

    #[test]
    fn ordinary_color_pipeline_materializes_only_the_documented_defaults() {
        let [source] = admit_static_wgsl_sources([(
            "proof.pipeline.render",
            1,
            r#"
@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(f32(vertex_index), 0.0, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
"#,
        )])
        .unwrap();

        let pipeline = GpuRenderPipelineDescriptor::ordinary_color(
            source.clone(),
            "vs_main",
            "fs_main",
            GpuTextureFormat::Rgba8Unorm,
        )
        .unwrap();
        let state = pipeline.state();
        let target = state
            .fragment_output()
            .unwrap()
            .color_targets()
            .next()
            .unwrap();

        assert!(pipeline.program().source().is_same_record(&source));
        assert_eq!(pipeline.entry_points().vertex().as_str(), "vs_main");
        assert_eq!(
            pipeline.entry_points().fragment().unwrap().as_str(),
            "fs_main"
        );
        assert_eq!(state.vertex_input().layouts().len(), 0);
        assert_eq!(target.format(), GpuTextureFormat::Rgba8Unorm);
        assert_eq!(target.blend(), GpuBlendMode::Replace);
        assert_eq!(target.write_mask(), GpuColorWriteMask::ALL);
        assert_eq!(state.primitive(), GpuPrimitiveStateDescriptor::default());
        assert_eq!(state.depth_stencil(), None);
        assert_eq!(
            state.multisample(),
            GpuMultisampleStateDescriptor::default()
        );
    }

    #[test]
    fn ordinary_color_pipeline_retains_stage_io_validation() {
        let [source] = admit_static_wgsl_sources([(
            "proof.pipeline.vertex-input",
            1,
            r#"
@vertex
fn vs_main(@location(0) position: vec2<f32>) -> @builtin(position) vec4<f32> {
    return vec4<f32>(position, 0.0, 1.0);
}

@fragment
fn fs_main() -> @location(0) vec4<f32> {
    return vec4<f32>(1.0, 0.0, 0.0, 1.0);
}
"#,
        )])
        .unwrap();

        let error = GpuRenderPipelineDescriptor::ordinary_color(
            source,
            "vs_main",
            "fs_main",
            GpuTextureFormat::Rgba8Unorm,
        )
        .unwrap_err();

        assert_eq!(
            error.cause(),
            GpuProgramContractCause::PipelineStageIoMismatch
        );
    }

    #[test]
    fn ordinary_resource_scope_retains_one_owner_scope() {
        let mut scope = GpuResourceScope::new();
        let first = scope.identities.allocate().unwrap();
        let second = scope.identities.allocate().unwrap();
        let first_parts = first.diagnostic_parts();
        let second_parts = second.diagnostic_parts();

        assert_eq!(first_parts.0, second_parts.0);
        assert_eq!(first_parts.1, 1);
        assert_eq!(second_parts.1, 2);
    }

    #[test]
    fn ordinary_submission_delegates_to_the_existing_canonical_path() {
        let source = std::fs::read_to_string(file!()).expect("ordinary source must be readable");
        let method = source
            .split_once("    pub async fn submit_work(")
            .expect("ordinary submission method must remain present")
            .1
            .split_once("\n}\n\nimpl From<super::GpuComputeOperation> for GpuWorkOperation {")
            .expect("ordinary submission method must remain bounded")
            .0;

        let validate_label = method
            .find("GpuResourceLabel::new(label.as_ref())?")
            .expect("ordinary submission must validate its text label through the canonical label authority");
        let prepare_graph = method
            .find("self.prepare_work_graph(label, fragments)?")
            .expect("ordinary work must use context-aware canonical graph preparation");
        let prepare_submission = method
            .find("self.prepare_submission(graph).await?")
            .expect("ordinary work must use canonical submission preparation");
        let submit_prepared = method
            .find("self.submit_prepared(prepared)")
            .expect("ordinary work must use canonical prepared submission");

        assert!(validate_label < prepare_graph);
        assert!(prepare_graph < prepare_submission);
        assert!(prepare_submission < submit_prepared);
        for forbidden in [
            "GpuPreparedWorkGraph::prepare(",
            "prepare_execution_plan(",
            "encode_submit_and_register(",
            "create_command_encoder",
            "queue.submit",
        ] {
            assert!(
                !method.contains(forbidden),
                "ordinary submission must not duplicate execution authority through {forbidden:?}"
            );
        }
    }

    #[test]
    fn ordinary_operation_delegates_to_checked_lexical_canonical_authoring() {
        let source = std::fs::read_to_string(file!()).expect("ordinary source must be readable");
        let method = source
            .split_once("    pub fn operation<L>(")
            .expect("ordinary operation method must remain present")
            .1
            .split_once("\n}\n\n/// Owns one bounded logical RunenGPU resource scope")
            .expect("ordinary operation method must remain bounded")
            .0;

        assert!(method.contains("L: AsRef<str>"));
        assert!(method.contains("operation: impl Into<GpuWorkOperation>"));
        assert!(method.contains("self.add_checked_lexical_operation(label, operation.into())"));
        for forbidden in [
            "self.add_node(",
            "declare_resource(",
            "GpuResourceLabel::new(",
        ] {
            assert!(
                !method.contains(forbidden),
                "ordinary operation must not duplicate canonical checked authoring through {forbidden:?}"
            );
        }
    }
}
