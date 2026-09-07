use super::super::{
    GpuAttachmentStore, GpuBufferCoverage, GpuBufferInitialization, GpuBufferRange,
    GpuBufferStridedCoverage, GpuBufferTextureLayout, GpuClearOperation, GpuColorAttachmentLoad,
    GpuCopyOperation, GpuDepthAttachmentLoad, GpuQueryRange, GpuResourceAccess, GpuResourceRef,
    GpuRetainedInitializationSeed, GpuTextureAspect, GpuTextureCopyRegion, GpuTextureDimension,
    GpuTextureHandle, GpuTextureInitialization, GpuTextureSubresourceRange, GpuWorkGraphCause,
    GpuWorkGraphError, GpuWorkGraphErrorContext, GpuWorkGraphErrorSource, GpuWorkOperation,
    GpuWorkResourceId,
};
use super::{
    authoring::{GpuWorkFragment, GpuWorkNode},
    composition::ImportBindings,
    coverage::{
        GpuInitialCoverage, GpuInitialCoverageData, buffer_coverage_contains,
        canonical_storage_resource, canonical_texture_aspect, coverage_source_error,
        intersect_buffer_coverage, normalize_buffer_coverage, normalize_u32_intervals,
        storage_identity, texture_aspect,
    },
    diagnostics::{
        GpuPreparedWorkDiagnostic, GraphErrorOrigin, graph_error, graph_error_with_region,
    },
    identity::GpuPreparedWorkNodeId,
    initial_content::GpuPreparedInitialContent,
};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, PartialEq, Eq)]
enum InitializedCoverage {
    Buffer {
        buffer: super::super::GpuBufferHandle,
        values: Vec<GpuBufferCoverage>,
    },
    Texture(BTreeMap<(u32, GpuTextureAspect), Vec<(u32, u32)>>),
    Query(Vec<(u32, u32)>),
    Immutable,
}

impl InitializedCoverage {
    fn empty_for(resource: &GpuResourceRef) -> Self {
        match resource {
            GpuResourceRef::Buffer(buffer) => Self::Buffer {
                buffer: buffer.clone(),
                values: Vec::new(),
            },
            GpuResourceRef::Texture(_) | GpuResourceRef::TextureView(_) => {
                Self::Texture(BTreeMap::new())
            }
            GpuResourceRef::QuerySet(_) => Self::Query(Vec::new()),
            GpuResourceRef::Sampler(_) => Self::Immutable,
        }
    }

    fn union(&mut self, other: &Self) -> bool {
        match (self, other) {
            (
                Self::Buffer {
                    buffer: left_buffer,
                    values: left,
                },
                Self::Buffer {
                    buffer: right_buffer,
                    values: right,
                },
            ) if left_buffer == right_buffer => {
                left.extend(right.iter().cloned());
                normalize_buffer_coverage(left_buffer, left);
                true
            }
            (Self::Texture(left), Self::Texture(right)) => {
                for (key, intervals) in right {
                    let entry = left.entry(*key).or_default();
                    entry.extend(intervals.iter().copied());
                    *entry = normalize_u32_intervals(core::mem::take(entry));
                }
                true
            }
            (Self::Query(left), Self::Query(right)) => {
                left.extend(right.iter().copied());
                *left = normalize_u32_intervals(core::mem::take(left));
                true
            }
            (Self::Immutable, Self::Immutable) => true,
            _ => false,
        }
    }

    fn contains(&self, required: &Self) -> bool {
        match (self, required) {
            (
                Self::Buffer { values: have, .. },
                Self::Buffer {
                    values: required, ..
                },
            ) => buffer_coverage_contains(have, required),
            (Self::Texture(have), Self::Texture(required)) => {
                required.iter().all(|(key, intervals)| {
                    have.get(key).is_some_and(|have_intervals| {
                        intervals
                            .iter()
                            .all(|range| interval_set_contains_u32(have_intervals, *range))
                    })
                })
            }
            (Self::Query(have), Self::Query(required)) => required
                .iter()
                .all(|range| interval_set_contains_u32(have, *range)),
            (Self::Immutable, Self::Immutable) => true,
            _ => false,
        }
    }

    fn remove(&mut self, removed: &Self) -> bool {
        match (self, removed) {
            (Self::Buffer { .. }, Self::Buffer { .. }) => false,
            (Self::Texture(have), Self::Texture(removed)) => {
                for (key, intervals) in removed {
                    if let Some(have_intervals) = have.get_mut(key) {
                        for interval in intervals {
                            *have_intervals =
                                subtract_u32_intervals(core::mem::take(have_intervals), *interval);
                        }
                    }
                }
                have.retain(|_, intervals| !intervals.is_empty());
                true
            }
            (Self::Query(have), Self::Query(removed)) => {
                for interval in removed {
                    *have = subtract_u32_intervals(core::mem::take(have), *interval);
                }
                true
            }
            (Self::Immutable, Self::Immutable) => true,
            _ => false,
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            Self::Buffer { values, .. } => values.is_empty(),
            Self::Texture(ranges) => ranges.is_empty(),
            Self::Query(ranges) => ranges.is_empty(),
            Self::Immutable => false,
        }
    }
}

impl InitializedCoverage {
    fn semantically_equals(&self, other: &Self) -> bool {
        self.contains(other) && other.contains(self)
    }
}

fn interval_set_contains_u32(intervals: &[(u32, u32)], required: (u32, u32)) -> bool {
    intervals
        .iter()
        .any(|&(start, end)| start <= required.0 && end >= required.1)
}

fn subtract_u32_intervals(intervals: Vec<(u32, u32)>, removed: (u32, u32)) -> Vec<(u32, u32)> {
    let mut result = Vec::new();
    for (start, end) in intervals {
        if end <= removed.0 || start >= removed.1 {
            result.push((start, end));
            continue;
        }
        if start < removed.0 {
            result.push((start, removed.0));
        }
        if end > removed.1 {
            result.push((removed.1, end));
        }
    }
    result
}

fn coverage_for_access(access: &GpuResourceAccess) -> InitializedCoverage {
    match access {
        GpuResourceAccess::Buffer(access) => InitializedCoverage::Buffer {
            buffer: access.buffer().clone(),
            values: vec![GpuBufferCoverage::dense(access.range())],
        },
        GpuResourceAccess::Texture(access) => {
            let range = access.normalized_subresources();
            let aspect = canonical_texture_aspect(
                range.aspect(),
                texture_aspect(access.normalized_texture()),
            );
            let mut subresources = BTreeMap::new();
            for mip in range.base_mip_level()..range.mip_end() {
                subresources.insert(
                    (mip, aspect),
                    vec![(range.base_array_layer(), range.layer_end())],
                );
            }
            InitializedCoverage::Texture(subresources)
        }
        GpuResourceAccess::Query(access) => {
            InitializedCoverage::Query(vec![(access.range().first(), access.range().end())])
        }
        GpuResourceAccess::Sampler(_) => InitializedCoverage::Immutable,
    }
}

fn descriptor_coverage(resource: &GpuResourceRef) -> InitializedCoverage {
    match resource {
        GpuResourceRef::Buffer(buffer) => match buffer.descriptor().initialization() {
            GpuBufferInitialization::Uninitialized | GpuBufferInitialization::Prepared(_) => {
                InitializedCoverage::Buffer {
                    buffer: buffer.clone(),
                    values: Vec::new(),
                }
            }
            GpuBufferInitialization::Zeroed => InitializedCoverage::Buffer {
                buffer: buffer.clone(),
                values: vec![GpuBufferCoverage::dense(
                    GpuBufferRange::whole(buffer)
                        .expect("buffer descriptor has nonzero checked coverage"),
                )],
            },
        },
        GpuResourceRef::Texture(texture) => descriptor_texture_coverage(texture),
        GpuResourceRef::TextureView(view) => {
            let parent = view.descriptor().texture();
            let mut coverage = descriptor_texture_coverage(parent);
            let view_coverage = texture_range_coverage(parent, view.descriptor().subresources());
            intersect_coverage(&mut coverage, &view_coverage);
            coverage
        }
        GpuResourceRef::Sampler(_) => InitializedCoverage::Immutable,
        GpuResourceRef::QuerySet(_) => InitializedCoverage::Query(Vec::new()),
    }
}

fn descriptor_texture_coverage(texture: &GpuTextureHandle) -> InitializedCoverage {
    let descriptor = texture.descriptor();
    let mip_count = match descriptor.initialization() {
        GpuTextureInitialization::Uninitialized | GpuTextureInitialization::Prepared(_) => 0,
        GpuTextureInitialization::Zeroed => descriptor.mip_level_count(),
    };
    let layers = match descriptor.dimension() {
        GpuTextureDimension::D2 => descriptor.extent().depth_or_layers(),
        GpuTextureDimension::D1 | GpuTextureDimension::D3 => 1,
    };
    let mut ranges = BTreeMap::new();
    for mip in 0..mip_count {
        ranges.insert((mip, texture_aspect(texture)), vec![(0, layers)]);
    }
    InitializedCoverage::Texture(ranges)
}

fn prepared_initial_content_coverage(candidate: &GpuPreparedInitialContent) -> InitializedCoverage {
    match candidate {
        GpuPreparedInitialContent::Buffer(buffer) => InitializedCoverage::Buffer {
            buffer: buffer.clone(),
            values: vec![GpuBufferCoverage::dense(
                GpuBufferRange::whole(buffer)
                    .expect("prepared buffer descriptor has nonzero checked coverage"),
            )],
        },
        GpuPreparedInitialContent::Texture(texture) => {
            let descriptor = texture.descriptor();
            let layers = match descriptor.dimension() {
                GpuTextureDimension::D2 => descriptor.extent().depth_or_layers(),
                GpuTextureDimension::D1 | GpuTextureDimension::D3 => 1,
            };
            let mut ranges = BTreeMap::new();
            ranges.insert((0, texture_aspect(texture)), vec![(0, layers)]);
            InitializedCoverage::Texture(ranges)
        }
    }
}

fn apply_fragment_prepared_initial_content(
    graph_label: &str,
    fragment: &GpuWorkFragment,
    state: &mut BTreeMap<GpuWorkResourceId, InitializedCoverage>,
    initial_content: &[GpuPreparedInitialContent],
) -> Result<(), GpuWorkGraphError> {
    for candidate in initial_content {
        let identity = candidate.resource_identity();
        if !fragment
            .resources()
            .iter()
            .any(|resource| storage_identity(resource) == identity)
        {
            continue;
        }
        let Some(existing) = state.get_mut(&identity) else {
            return Err(graph_error(
                "apply prepared initial-content validation effect",
                graph_label,
                GraphErrorOrigin::new(Some(fragment), None),
                None,
                Some(identity),
                GpuWorkGraphCause::UnknownIdentity,
                "retain each used prepared resource in the fragment's normalized storage state",
            ));
        };
        if !existing.union(&prepared_initial_content_coverage(candidate)) {
            return Err(graph_error(
                "apply prepared initial-content validation effect",
                graph_label,
                GraphErrorOrigin::new(Some(fragment), None),
                None,
                Some(identity),
                GpuWorkGraphCause::OperationAccessContradiction,
                "apply prepared initial content only to its matching normalized storage kind",
            ));
        }
    }
    Ok(())
}

fn apply_global_prepared_initial_content(
    graph_label: &str,
    state: &mut BTreeMap<GpuWorkResourceId, InitializedCoverage>,
    initial_content: &[GpuPreparedInitialContent],
    explanations: &mut PendingInitializationExplanations,
) -> Result<(), GpuWorkGraphError> {
    for candidate in initial_content {
        let identity = candidate.resource_identity();
        let Some(existing) = state.get_mut(&identity) else {
            let provenance = match candidate {
                GpuPreparedInitialContent::Buffer(buffer) => {
                    buffer.descriptor().common().provenance().clone()
                }
                GpuPreparedInitialContent::Texture(texture) => {
                    texture.descriptor().common().provenance().clone()
                }
            };
            return Err(GpuWorkGraphError::invalid(
                "apply prepared initial-content simulation effect",
                GpuWorkGraphErrorContext::new(
                    graph_label,
                    None,
                    None,
                    None,
                    Some(identity),
                    None,
                    Some(provenance),
                ),
                GpuWorkGraphCause::UnknownIdentity,
                "retain each operation-used prepared resource in the normalized storage registry",
            ));
        };
        let materialized = prepared_initial_content_coverage(candidate);
        if !existing.union(&materialized) {
            return Err(GpuWorkGraphError::invalid(
                "apply prepared initial-content simulation effect",
                GpuWorkGraphErrorContext::new(
                    graph_label,
                    None,
                    None,
                    None,
                    Some(identity),
                    None,
                    None,
                ),
                GpuWorkGraphCause::OperationAccessContradiction,
                "apply prepared initial content only to its matching normalized storage kind",
            ));
        }
        record_pending_initialization_explanation(
            explanations,
            identity,
            GpuInitializationExplanationKind::PreparedInitialContent,
            materialized,
        );
    }
    Ok(())
}

fn texture_range_coverage(
    texture: &GpuTextureHandle,
    range: GpuTextureSubresourceRange,
) -> InitializedCoverage {
    let aspect = canonical_texture_aspect(range.aspect(), texture_aspect(texture));
    let mut ranges = BTreeMap::new();
    for mip in range.base_mip_level()..range.mip_end() {
        ranges.insert(
            (mip, aspect),
            vec![(range.base_array_layer(), range.layer_end())],
        );
    }
    InitializedCoverage::Texture(ranges)
}

fn intersect_coverage(left: &mut InitializedCoverage, right: &InitializedCoverage) -> bool {
    match (left, right) {
        (
            InitializedCoverage::Buffer {
                buffer: left_buffer,
                values: left_values,
            },
            InitializedCoverage::Buffer {
                buffer: right_buffer,
                values: right_values,
            },
        ) if left_buffer == right_buffer => {
            *left_values = intersect_buffer_coverage(left_buffer, left_values, right_values);
            true
        }
        (InitializedCoverage::Texture(left), InitializedCoverage::Texture(right)) => {
            left.retain(|key, left_intervals| {
                let Some(right_intervals) = right.get(key) else {
                    return false;
                };
                *left_intervals = intersect_u32_intervals(left_intervals, right_intervals);
                !left_intervals.is_empty()
            });
            true
        }
        (InitializedCoverage::Query(left), InitializedCoverage::Query(right)) => {
            *left = intersect_u32_intervals(left, right);
            true
        }
        (InitializedCoverage::Immutable, InitializedCoverage::Immutable) => true,
        _ => false,
    }
}

fn intersect_u32_intervals(left: &[(u32, u32)], right: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let mut result = Vec::new();
    for &(left_start, left_end) in left {
        for &(right_start, right_end) in right {
            let start = left_start.max(right_start);
            let end = left_end.min(right_end);
            if start < end {
                result.push((start, end));
            }
        }
    }
    normalize_u32_intervals(result)
}

fn initial_coverage_value(coverage: &GpuInitialCoverage) -> InitializedCoverage {
    match &coverage.data {
        GpuInitialCoverageData::DescriptorInitialization => {
            descriptor_coverage(coverage.resource())
        }
        GpuInitialCoverageData::Buffer(values) => match coverage.resource() {
            GpuResourceRef::Buffer(buffer) => InitializedCoverage::Buffer {
                buffer: buffer.clone(),
                values: values.clone(),
            },
            _ => InitializedCoverage::Immutable,
        },
        GpuInitialCoverageData::TextureSubresources(ranges) => {
            let texture = match coverage.resource() {
                GpuResourceRef::Texture(texture) => texture,
                GpuResourceRef::TextureView(view) => view.descriptor().texture(),
                _ => return InitializedCoverage::Texture(BTreeMap::new()),
            };
            let mut value = InitializedCoverage::Texture(BTreeMap::new());
            for range in ranges {
                let _ = value.union(&texture_range_coverage(texture, *range));
            }
            value
        }
        GpuInitialCoverageData::QueryRanges(ranges) => InitializedCoverage::Query(
            ranges
                .iter()
                .map(|range| (range.first(), range.end()))
                .collect(),
        ),
    }
}

pub(crate) fn initial_coverage_contains(
    have: &GpuInitialCoverage,
    required: &GpuInitialCoverage,
) -> bool {
    have.storage_resource == required.storage_resource
        && initial_coverage_value(have).contains(&initial_coverage_value(required))
}

pub(crate) fn initial_coverage_intersection(
    resource: &GpuResourceRef,
    left: &GpuInitialCoverage,
    right: &GpuInitialCoverage,
) -> Option<GpuInitialCoverage> {
    if left.storage_resource != right.storage_resource {
        return None;
    }
    let mut intersection = initial_coverage_value(left);
    if !intersect_coverage(&mut intersection, &initial_coverage_value(right)) {
        return None;
    }
    coverage_to_public("retained continuity", resource, &intersection)
        .ok()
        .flatten()
}

fn apply_retained_initial_coverage(
    graph_label: &str,
    state: &mut BTreeMap<GpuWorkResourceId, InitializedCoverage>,
    retained_storage: &BTreeSet<GpuWorkResourceId>,
    retained_state: &[GpuRetainedInitializationSeed],
) -> Result<(), GpuWorkGraphError> {
    for seed in retained_state {
        let seed_storage = canonical_storage_resource(seed.resource());
        let identity = seed.resource_identity();
        if !retained_storage.contains(&identity) || !seed_storage.common().lifetime().is_retained()
        {
            continue;
        }
        let Some(existing) = state.get_mut(&identity) else {
            continue;
        };
        *existing = InitializedCoverage::empty_for(&seed_storage);
        if let Some(coverage) = seed.initialized_coverage() {
            if !existing.union(&initial_coverage_value(coverage)) {
                return Err(GpuWorkGraphError::invalid(
                    "apply retained GPU initialization state",
                    GpuWorkGraphErrorContext::new(
                        graph_label,
                        None,
                        None,
                        None,
                        Some(identity),
                        None,
                        Some(seed_storage.common().provenance().clone()),
                    ),
                    GpuWorkGraphCause::ImportExportMismatch,
                    "retain lifecycle state only when current storage and retained coverage describe the same normalized resource",
                ));
            }
        }
    }
    Ok(())
}

/// Typed reason why canonical initialized coverage was established or removed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuInitializationExplanationKind {
    DescriptorGuaranteed,
    DeclaredInput,
    RetainedCoverage,
    PreparedInitialContent,
    OperationGuaranteed,
    AttachmentDiscard,
}

/// One ordered typed explanation fact emitted by the canonical initialization simulator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuInitializationExplanation {
    kind: GpuInitializationExplanationKind,
    coverage: GpuInitialCoverage,
}

impl GpuInitializationExplanation {
    pub const fn kind(&self) -> GpuInitializationExplanationKind {
        self.kind
    }

    pub fn coverage(&self) -> &GpuInitialCoverage {
        &self.coverage
    }
}

#[derive(Debug, Clone)]
struct PendingInitializationExplanation {
    kind: GpuInitializationExplanationKind,
    coverage: InitializedCoverage,
}

type PendingInitializationExplanations =
    BTreeMap<GpuWorkResourceId, Vec<PendingInitializationExplanation>>;

fn record_pending_initialization_explanation(
    explanations: &mut PendingInitializationExplanations,
    resource: GpuWorkResourceId,
    kind: GpuInitializationExplanationKind,
    coverage: InitializedCoverage,
) {
    if coverage.is_empty() {
        return;
    }
    explanations
        .entry(resource)
        .or_default()
        .push(PendingInitializationExplanation { kind, coverage });
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuPreparedResourceInitialization {
    resource: GpuResourceRef,
    initial: Option<GpuInitialCoverage>,
    final_coverage: Option<GpuInitialCoverage>,
    explanations: Vec<GpuInitializationExplanation>,
}

impl GpuPreparedResourceInitialization {
    pub fn resource(&self) -> &GpuResourceRef {
        &self.resource
    }

    pub fn initial(&self) -> Option<&GpuInitialCoverage> {
        self.initial.as_ref()
    }

    pub fn final_coverage(&self) -> Option<&GpuInitialCoverage> {
        self.final_coverage.as_ref()
    }

    pub fn explanations(&self) -> &[GpuInitializationExplanation] {
        &self.explanations
    }
}

type PreparedInitializationSimulation = (
    Vec<GpuPreparedResourceInitialization>,
    Vec<GpuPreparedWorkDiagnostic>,
    BTreeMap<GpuWorkResourceId, GpuInitialCoverage>,
);

pub(super) fn validate_fragment_initialization(
    graph_label: &str,
    fragments: &[GpuWorkFragment],
    fragment_order: &[usize],
    import_bindings: &ImportBindings,
    initial_content: &[GpuPreparedInitialContent],
    retained_coverage: &[GpuRetainedInitializationSeed],
) -> Result<(), GpuWorkGraphError> {
    let mut prepared_outputs = BTreeMap::<(usize, usize), InitializedCoverage>::new();
    for &fragment_index in fragment_order {
        let fragment = &fragments[fragment_index];
        let mut state = fragment_entry_state(graph_label, fragment, retained_coverage)?;
        apply_fragment_prepared_initial_content(
            graph_label,
            fragment,
            &mut state,
            initial_content,
        )?;
        for (import_index, import) in fragment.imports().iter().enumerate() {
            let Some(binding) = import_bindings.get(&(fragment_index, import_index)) else {
                return Err(graph_error(
                    "resolve GPU work import coverage",
                    graph_label,
                    GraphErrorOrigin::new(Some(fragment), None),
                    None,
                    Some(storage_identity(import.resource())),
                    GpuWorkGraphCause::ImportExportMismatch,
                    "bind every import to one validated producer output",
                ));
            };
            let Some(coverage) = prepared_outputs.get(binding) else {
                return Err(graph_error(
                    "resolve GPU work import coverage",
                    graph_label,
                    GraphErrorOrigin::new(Some(fragment), None),
                    None,
                    Some(storage_identity(import.resource())),
                    GpuWorkGraphCause::ImportExportMismatch,
                    "order producer fragments before consumers through typed imports",
                ));
            };
            union_state_coverage(
                graph_label,
                fragment,
                &mut state,
                storage_identity(import.resource()),
                coverage,
            )?;
        }
        for node in fragment.nodes() {
            let id = GpuPreparedWorkNodeId::new(
                u32::try_from(fragment_index).map_err(|_| {
                    graph_error(
                        "assign prepared fragment identity",
                        graph_label,
                        GraphErrorOrigin::new(Some(fragment), Some(node)),
                        None,
                        None,
                        GpuWorkGraphCause::UnknownIdentity,
                        "prepare fewer fragments in one bounded graph",
                    )
                })?,
                node.id().local,
            );
            apply_node_initialization(graph_label, fragment, node, id, &mut state, None)?;
        }
        for (output_index, output) in fragment.outputs().iter().enumerate() {
            let resource = storage_identity(output.relationship().resource());
            let expected = initial_coverage_value(output.final_initialized_coverage());
            let mut actual = state.get(&resource).cloned().unwrap_or_else(|| {
                InitializedCoverage::empty_for(output.relationship().resource())
            });
            let domain = resource_domain_coverage(output.relationship().resource());
            if !intersect_coverage(&mut actual, &domain) || !actual.semantically_equals(&expected) {
                return Err(graph_error_with_region(
                    "validate GPU work output coverage",
                    graph_label,
                    GraphErrorOrigin::new(Some(fragment), None),
                    None,
                    (resource, format!("{expected:?}")),
                    GpuWorkGraphCause::ImportExportMismatch,
                    "declare the producer fragment's exact final initialized coverage",
                ));
            }
            prepared_outputs.insert((fragment_index, output_index), expected);
        }
    }
    Ok(())
}

fn fragment_entry_state(
    graph_label: &str,
    fragment: &GpuWorkFragment,
    retained_coverage: &[GpuRetainedInitializationSeed],
) -> Result<BTreeMap<GpuWorkResourceId, InitializedCoverage>, GpuWorkGraphError> {
    let mut state = BTreeMap::new();
    let mut retained_storage = BTreeSet::new();
    for resource in fragment.resources() {
        let storage = canonical_storage_resource(resource);
        let identity = storage_identity(&storage);
        if storage.common().lifetime().is_retained() {
            retained_storage.insert(identity);
        }
        let coverage = descriptor_coverage(resource);
        state
            .entry(identity)
            .and_modify(|existing: &mut InitializedCoverage| {
                let _ = existing.union(&coverage);
            })
            .or_insert(coverage);
    }
    apply_retained_initial_coverage(
        graph_label,
        &mut state,
        &retained_storage,
        retained_coverage,
    )?;
    for input in fragment.inputs() {
        let resource = input.initialized_coverage().storage_resource;
        let coverage = initial_coverage_value(input.initialized_coverage());
        state
            .entry(resource)
            .and_modify(|existing| {
                let _ = existing.union(&coverage);
            })
            .or_insert(coverage);
    }
    Ok(state)
}

fn union_state_coverage(
    graph_label: &str,
    fragment: &GpuWorkFragment,
    state: &mut BTreeMap<GpuWorkResourceId, InitializedCoverage>,
    resource: GpuWorkResourceId,
    coverage: &InitializedCoverage,
) -> Result<(), GpuWorkGraphError> {
    let Some(existing) = state.get_mut(&resource) else {
        return Err(graph_error(
            "merge GPU initialization coverage",
            graph_label,
            GraphErrorOrigin::new(Some(fragment), None),
            None,
            Some(resource),
            GpuWorkGraphCause::UnknownIdentity,
            "declare the initialized resource in the fragment",
        ));
    };
    if !existing.union(coverage) {
        return Err(graph_error(
            "merge GPU initialization coverage",
            graph_label,
            GraphErrorOrigin::new(Some(fragment), None),
            None,
            Some(resource),
            GpuWorkGraphCause::ImportExportMismatch,
            "bind coverage with the same normalized resource kind",
        ));
    }
    Ok(())
}

fn resource_domain_coverage(resource: &GpuResourceRef) -> InitializedCoverage {
    match resource {
        GpuResourceRef::Buffer(buffer) => InitializedCoverage::Buffer {
            buffer: buffer.clone(),
            values: vec![GpuBufferCoverage::dense(
                GpuBufferRange::whole(buffer)
                    .expect("buffer descriptor has nonzero checked coverage"),
            )],
        },
        GpuResourceRef::Texture(texture) => whole_texture_coverage(texture, None),
        GpuResourceRef::TextureView(view) => whole_texture_coverage(
            view.descriptor().texture(),
            Some(view.descriptor().subresources()),
        ),
        GpuResourceRef::QuerySet(query_set) => {
            InitializedCoverage::Query(vec![(0, query_set.descriptor().count())])
        }
        GpuResourceRef::Sampler(_) => InitializedCoverage::Immutable,
    }
}

fn whole_texture_coverage(
    texture: &GpuTextureHandle,
    restriction: Option<GpuTextureSubresourceRange>,
) -> InitializedCoverage {
    if let Some(restriction) = restriction {
        return texture_range_coverage(texture, restriction);
    }
    let descriptor = texture.descriptor();
    let layers = match descriptor.dimension() {
        GpuTextureDimension::D2 => descriptor.extent().depth_or_layers(),
        GpuTextureDimension::D1 | GpuTextureDimension::D3 => 1,
    };
    let mut ranges = BTreeMap::new();
    for mip in 0..descriptor.mip_level_count() {
        ranges.insert((mip, texture_aspect(texture)), vec![(0, layers)]);
    }
    InitializedCoverage::Texture(ranges)
}

fn apply_node_initialization(
    graph_label: &str,
    fragment: &GpuWorkFragment,
    node: &GpuWorkNode,
    prepared_id: GpuPreparedWorkNodeId,
    state: &mut BTreeMap<GpuWorkResourceId, InitializedCoverage>,
    mut explanations: Option<&mut PendingInitializationExplanations>,
) -> Result<(), GpuWorkGraphError> {
    let mut requirements = node
        .caller_readable_accesses()
        .iter()
        .map(initialization_region_for_access)
        .collect::<Vec<_>>();
    let (operation_requirements, effects) =
        operation_initialization(node.operation(), graph_label, fragment, node, prepared_id)?;
    requirements.extend(operation_requirements);
    for required in requirements {
        let resource = required.resource;
        if !state
            .get(&resource)
            .is_some_and(|coverage| coverage.contains(&required.coverage))
        {
            let storage_resource = fragment
                .resources()
                .iter()
                .find(|candidate| storage_identity(candidate) == resource)
                .map(canonical_storage_resource)
                .expect("validated fragment retains every initialization requirement resource");
            let required_initialization =
                coverage_to_public(graph_label, &storage_resource, &required.coverage)?
                    .expect("nonempty initialization requirements publish typed coverage");
            return Err(GpuWorkGraphError::invalid(
                "prepare GPU work initialization",
                GpuWorkGraphErrorContext::new(
                    graph_label,
                    Some(fragment.label().as_str().to_string()),
                    Some(node.label().as_str().to_string()),
                    Some(prepared_id),
                    Some(resource),
                    Some(required.description),
                    Some(node.provenance().clone()),
                )
                .with_required_initialization(required_initialization),
                GpuWorkGraphCause::ReadBeforeInitialization,
                "provide descriptor/input/import/retained coverage or preceding work that initializes the exact region",
            ));
        }
    }
    for effect in effects {
        let contained = node.accesses().iter().any(|access| {
            access.resource_identity() == effect.resource
                && access_has_compatible_role(access, &effect.support)
                && coverage_for_access(access).contains(&effect.coverage)
        });
        if !contained {
            return Err(graph_error(
                "validate GPU operation initialization effect",
                graph_label,
                GraphErrorOrigin::new(Some(fragment), Some(node)),
                Some(prepared_id),
                Some(effect.resource),
                GpuWorkGraphCause::OperationAccessContradiction,
                "keep every operation-guaranteed initialization effect inside a checked compatible write access",
            ));
        }
        let Some(coverage) = state.get_mut(&effect.resource) else {
            return Err(graph_error(
                "apply GPU operation initialization effect",
                graph_label,
                GraphErrorOrigin::new(Some(fragment), Some(node)),
                Some(prepared_id),
                Some(effect.resource),
                GpuWorkGraphCause::UnknownIdentity,
                "declare each normalized storage resource before applying operation effects",
            ));
        };
        if !coverage.union(&effect.coverage) {
            return Err(graph_error(
                "apply GPU operation initialization effect",
                graph_label,
                GraphErrorOrigin::new(Some(fragment), Some(node)),
                Some(prepared_id),
                Some(effect.resource),
                GpuWorkGraphCause::OperationAccessContradiction,
                "apply initialization effects only to matching normalized storage kinds",
            ));
        }
        if let Some(explanations) = explanations.as_deref_mut() {
            record_pending_initialization_explanation(
                explanations,
                effect.resource,
                GpuInitializationExplanationKind::OperationGuaranteed,
                effect.coverage,
            );
        }
    }
    for discarded in operation_discard_regions(node.operation()) {
        let resource = discarded.resource;
        let affected = discarded.coverage;
        let Some(coverage) = state.get_mut(&resource) else {
            return Err(graph_error(
                "apply GPU work initialization",
                graph_label,
                GraphErrorOrigin::new(Some(fragment), Some(node)),
                Some(prepared_id),
                Some(resource),
                GpuWorkGraphCause::UnknownIdentity,
                "declare each normalized storage resource before use",
            ));
        };
        let mut removed = coverage.clone();
        if !intersect_coverage(&mut removed, &affected) {
            return Err(graph_error(
                "derive GPU attachment discard explanation",
                graph_label,
                GraphErrorOrigin::new(Some(fragment), Some(node)),
                Some(prepared_id),
                Some(resource),
                GpuWorkGraphCause::OperationAccessContradiction,
                "derive discard evidence only from matching normalized texture storage",
            ));
        }
        if !coverage.remove(&affected) {
            return Err(graph_error(
                "apply GPU attachment discard",
                graph_label,
                GraphErrorOrigin::new(Some(fragment), Some(node)),
                Some(prepared_id),
                Some(resource),
                GpuWorkGraphCause::OperationAccessContradiction,
                "discard coverage only from the matching texture storage",
            ));
        }
        if let Some(explanations) = explanations.as_deref_mut() {
            record_pending_initialization_explanation(
                explanations,
                resource,
                GpuInitializationExplanationKind::AttachmentDiscard,
                removed,
            );
        }
    }
    Ok(())
}

#[derive(Debug)]
struct InitializationRegion {
    resource: GpuWorkResourceId,
    coverage: InitializedCoverage,
    description: String,
}

#[derive(Debug)]
struct InitializationEffect {
    resource: GpuWorkResourceId,
    coverage: InitializedCoverage,
    support: GpuResourceAccess,
}

fn initialization_region_for_access(access: &GpuResourceAccess) -> InitializationRegion {
    InitializationRegion {
        resource: access.resource_identity(),
        coverage: coverage_for_access(access),
        description: access_region_description(access),
    }
}

fn operation_initialization(
    operation: &GpuWorkOperation,
    graph_label: &str,
    fragment: &GpuWorkFragment,
    node: &GpuWorkNode,
    prepared_id: GpuPreparedWorkNodeId,
) -> Result<(Vec<InitializationRegion>, Vec<InitializationEffect>), GpuWorkGraphError> {
    let mut requirements = Vec::new();
    let mut effects = Vec::new();
    let mut require =
        |access: &GpuResourceAccess| requirements.push(initialization_region_for_access(access));
    let mut effect =
        |access: &GpuResourceAccess| effects.push(initialization_region_for_access(access));
    match operation {
        GpuWorkOperation::Compute(compute) => {
            let shader_may_execute = compute
                .dispatch()
                .direct_size()
                .is_none_or(|size| size.as_array().into_iter().all(|dimension| dimension != 0));
            if shader_may_execute {
                for access in compute
                    .bindings()
                    .accesses()
                    .iter()
                    .filter(|access| access.reads())
                {
                    require(access);
                }
            }
            if let Some(arguments) = compute.dispatch().indirect_access() {
                require(&GpuResourceAccess::Buffer(arguments.clone()));
            }
            if let Some(timestamp_writes) = compute.timestamp_writes() {
                for timestamp in timestamp_writes.accesses() {
                    effect(&GpuResourceAccess::Query(timestamp.clone()));
                }
            }
        }
        GpuWorkOperation::Render(render) => {
            for attachment in render.color_attachments() {
                let access = GpuResourceAccess::Texture(attachment.source_access().clone());
                match attachment.load() {
                    GpuColorAttachmentLoad::Load => require(&access),
                    GpuColorAttachmentLoad::Clear(_) => effect(&access),
                }
                if let Some(resolve) = attachment.resolve_target() {
                    effect(&GpuResourceAccess::Texture(resolve.access().clone()));
                }
            }
            if let Some(attachment) = render.depth_stencil_attachment() {
                let access = GpuResourceAccess::Texture(attachment.source_access().clone());
                match attachment.load() {
                    GpuDepthAttachmentLoad::Load => require(&access),
                    GpuDepthAttachmentLoad::Clear(_) => effect(&access),
                }
            }
            for draw in render.draws() {
                for access in draw.accesses().iter().filter(|access| access.reads()) {
                    require(access);
                }
            }
            if let Some(timestamp_writes) = render.timestamp_writes() {
                for timestamp in timestamp_writes.accesses() {
                    effect(&GpuResourceAccess::Query(timestamp.clone()));
                }
            }
        }
        GpuWorkOperation::Copy(copy) => match copy {
            GpuCopyOperation::BufferToBuffer {
                source,
                destination,
            } => {
                requirements.push(buffer_region(
                    source.buffer(),
                    source.range(),
                    "buffer copy source",
                ));
                effects.push(buffer_region(
                    destination.buffer(),
                    destination.range(),
                    "buffer copy destination",
                ));
            }
            GpuCopyOperation::BufferToTexture {
                source,
                destination,
            } => {
                requirements.push(buffer_layout_region(
                    source,
                    destination,
                    "logical buffer copy source",
                    graph_label,
                    fragment,
                    node,
                    prepared_id,
                )?);
                if texture_copy_is_complete(destination) {
                    effects.push(texture_copy_region(destination, "texture copy destination"));
                }
            }
            GpuCopyOperation::TextureToBuffer {
                source,
                destination,
            } => {
                requirements.push(texture_copy_region(source, "texture copy source"));
                effects.push(buffer_layout_region(
                    destination,
                    source,
                    "logical buffer copy destination",
                    graph_label,
                    fragment,
                    node,
                    prepared_id,
                )?);
            }
            GpuCopyOperation::TextureToTexture {
                source,
                destination,
            } => {
                requirements.push(texture_copy_region(source, "texture copy source"));
                if texture_copy_is_complete(destination) {
                    effects.push(texture_copy_region(destination, "texture copy destination"));
                }
            }
        },
        GpuWorkOperation::Clear(GpuClearOperation::BufferZero(region)) => {
            effects.push(buffer_region(
                region.buffer(),
                region.range(),
                "buffer zero",
            ));
        }
        GpuWorkOperation::Resolve(resolve) => {
            require(&GpuResourceAccess::Query(resolve.source_access().clone()));
            effect(&GpuResourceAccess::Buffer(
                resolve.destination_access().clone(),
            ));
        }
        GpuWorkOperation::Present(present) => {
            require(&GpuResourceAccess::Texture(present.source_access().clone()));
        }
        GpuWorkOperation::Upload(upload) => {
            if upload.establishes_initialization_effect() {
                effect(upload.destination_access());
            }
        }
        GpuWorkOperation::Readback(readback) => {
            require(readback.source_access());
        }
    }
    let derived = operation.derived_accesses().map_err(|_| {
        graph_error(
            "derive GPU operation initialization effects",
            graph_label,
            GraphErrorOrigin::new(Some(fragment), Some(node)),
            Some(prepared_id),
            None,
            GpuWorkGraphCause::OperationAccessContradiction,
            "retain checked operation-derived accesses while preparing initialization",
        )
    })?;
    let effects = effects
        .into_iter()
        .map(|effect| {
            let support = derived
                .iter()
                .find(|access| {
                    access.resource_identity() == effect.resource
                        && access.writes()
                        && coverage_for_access(access).contains(&effect.coverage)
                })
                .cloned()
                .ok_or_else(|| {
                    graph_error(
                        "derive GPU operation initialization effects",
                        graph_label,
                        GraphErrorOrigin::new(Some(fragment), Some(node)),
                        Some(prepared_id),
                        Some(effect.resource),
                        GpuWorkGraphCause::OperationAccessContradiction,
                        "derive each initialization effect from a checked operation write access",
                    )
                })?;
            Ok(InitializationEffect {
                resource: effect.resource,
                coverage: effect.coverage,
                support,
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    Ok((requirements, effects))
}

fn buffer_region(
    buffer: &super::super::GpuBufferHandle,
    range: GpuBufferRange,
    description: &'static str,
) -> InitializationRegion {
    InitializationRegion {
        resource: buffer.diagnostic_identity(),
        coverage: InitializedCoverage::Buffer {
            buffer: buffer.clone(),
            values: vec![GpuBufferCoverage::dense(range)],
        },
        description: description.to_string(),
    }
}

fn texture_copy_region(
    region: &GpuTextureCopyRegion,
    description: &'static str,
) -> InitializationRegion {
    InitializationRegion {
        resource: region.texture().diagnostic_identity(),
        coverage: texture_range_coverage(region.texture(), region.subresources()),
        description: description.to_string(),
    }
}

fn buffer_layout_region(
    layout: &GpuBufferTextureLayout,
    texture: &GpuTextureCopyRegion,
    description: &'static str,
    graph_label: &str,
    fragment: &GpuWorkFragment,
    node: &GpuWorkNode,
    prepared_id: GpuPreparedWorkNodeId,
) -> Result<InitializationRegion, GpuWorkGraphError> {
    let extent = texture.extent();
    let segment_size = u64::from(extent.width())
        .checked_mul(u64::from(
            texture.texture().descriptor().format().bytes_per_texel(),
        ))
        .ok_or_else(|| {
            initialization_layout_error(graph_label, fragment, node, prepared_id, layout)
        })?;
    let group_stride = if extent.depth_or_layers() > 1 {
        u64::from(layout.bytes_per_row())
            .checked_mul(u64::from(layout.rows_per_image()))
            .ok_or_else(|| {
                initialization_layout_error(graph_label, fragment, node, prepared_id, layout)
            })?
    } else {
        0
    };
    let coverage = GpuBufferStridedCoverage::new(
        layout.buffer(),
        layout.byte_offset(),
        segment_size,
        u64::from(layout.bytes_per_row()),
        extent.height(),
        group_stride,
        extent.depth_or_layers(),
    )
    .map_err(|_| initialization_layout_error(graph_label, fragment, node, prepared_id, layout))?;
    Ok(InitializationRegion {
        resource: layout.buffer().diagnostic_identity(),
        coverage: InitializedCoverage::Buffer {
            buffer: layout.buffer().clone(),
            values: vec![GpuBufferCoverage::strided(coverage)],
        },
        description: description.to_string(),
    })
}

fn initialization_layout_error(
    graph_label: &str,
    fragment: &GpuWorkFragment,
    node: &GpuWorkNode,
    prepared_id: GpuPreparedWorkNodeId,
    layout: &GpuBufferTextureLayout,
) -> GpuWorkGraphError {
    graph_error(
        "derive GPU buffer-texture initialization coverage",
        graph_label,
        GraphErrorOrigin::new(Some(fragment), Some(node)),
        Some(prepared_id),
        Some(layout.buffer().diagnostic_identity()),
        GpuWorkGraphCause::OperationAccessContradiction,
        "retain the checked logical buffer-texture layout while preparing initialization",
    )
}

fn texture_copy_is_complete(region: &GpuTextureCopyRegion) -> bool {
    let descriptor = region.texture().descriptor();
    let mip = region.mip_level();
    let width = (descriptor.extent().width() >> mip).max(1);
    let height = (descriptor.extent().height() >> mip).max(1);
    let origin = region.origin();
    let extent = region.extent();
    if origin.x() != 0 || origin.y() != 0 || extent.width() != width || extent.height() != height {
        return false;
    }
    match descriptor.dimension() {
        GpuTextureDimension::D1 => origin.z() == 0 && extent.depth_or_layers() == 1,
        GpuTextureDimension::D2 => true,
        GpuTextureDimension::D3 => {
            origin.z() == 0
                && extent.depth_or_layers() == (descriptor.extent().depth_or_layers() >> mip).max(1)
        }
    }
}

fn access_has_compatible_role(
    access: &GpuResourceAccess,
    operation_access: &GpuResourceAccess,
) -> bool {
    match (access, operation_access) {
        (GpuResourceAccess::Buffer(access), GpuResourceAccess::Buffer(operation_access)) => {
            access.kind() == operation_access.kind()
        }
        (GpuResourceAccess::Texture(access), GpuResourceAccess::Texture(operation_access)) => {
            access.kind() == operation_access.kind()
        }
        (GpuResourceAccess::Query(access), GpuResourceAccess::Query(operation_access)) => {
            access.kind() == operation_access.kind()
        }
        (GpuResourceAccess::Sampler(_), GpuResourceAccess::Sampler(_)) => true,
        _ => false,
    }
}

fn operation_discard_regions(operation: &GpuWorkOperation) -> Vec<InitializationRegion> {
    let GpuWorkOperation::Render(render) = operation else {
        return Vec::new();
    };
    let mut discarded = Vec::new();
    for attachment in render.color_attachments() {
        if attachment.store() == GpuAttachmentStore::Discard {
            discarded.push(initialization_region_for_access(
                &GpuResourceAccess::Texture(attachment.source_access().clone()),
            ));
        }
    }
    if let Some(attachment) = render.depth_stencil_attachment() {
        if attachment.store() == GpuAttachmentStore::Discard {
            discarded.push(initialization_region_for_access(
                &GpuResourceAccess::Texture(attachment.source_access().clone()),
            ));
        }
    }
    discarded
}

pub(super) fn access_region_description(access: &GpuResourceAccess) -> String {
    match access {
        GpuResourceAccess::Buffer(access) => {
            format!(
                "bytes {}..{}",
                access.range().offset(),
                access.range().end()
            )
        }
        GpuResourceAccess::Texture(access) => {
            let range = access.normalized_subresources();
            format!(
                "mips {}..{}, layers {}..{}, {:?}",
                range.base_mip_level(),
                range.mip_end(),
                range.base_array_layer(),
                range.layer_end(),
                range.aspect()
            )
        }
        GpuResourceAccess::Query(access) => format!(
            "queries {}..{}",
            access.range().first(),
            access.range().end()
        ),
        GpuResourceAccess::Sampler(_) => "immutable sampler".to_string(),
    }
}

pub(super) fn simulate_prepared_initialization(
    graph_label: &str,
    fragments: &[GpuWorkFragment],
    storage_resources: &BTreeMap<GpuWorkResourceId, GpuResourceRef>,
    node_locations: &BTreeMap<GpuPreparedWorkNodeId, (usize, usize)>,
    topological_order: &[GpuPreparedWorkNodeId],
    initial_content: &[GpuPreparedInitialContent],
    retained_coverage: &[GpuRetainedInitializationSeed],
) -> Result<PreparedInitializationSimulation, GpuWorkGraphError> {
    let mut state = BTreeMap::<GpuWorkResourceId, InitializedCoverage>::new();
    let mut explanation_state = PendingInitializationExplanations::new();
    let retained_storage = storage_resources
        .iter()
        .filter_map(|(identity, resource)| {
            resource
                .common()
                .lifetime()
                .is_retained()
                .then_some(*identity)
        })
        .collect::<BTreeSet<_>>();
    let retained_seeded = retained_coverage
        .iter()
        .map(GpuRetainedInitializationSeed::resource_identity)
        .collect::<BTreeSet<_>>();
    for (identity, resource) in storage_resources {
        let descriptor = descriptor_coverage(resource);
        if !retained_seeded.contains(identity) {
            record_pending_initialization_explanation(
                &mut explanation_state,
                *identity,
                GpuInitializationExplanationKind::DescriptorGuaranteed,
                descriptor.clone(),
            );
        }
        state.insert(*identity, descriptor);
    }
    apply_retained_initial_coverage(
        graph_label,
        &mut state,
        &retained_storage,
        retained_coverage,
    )?;
    for seed in retained_coverage {
        if let Some(coverage) = seed.initialized_coverage() {
            record_pending_initialization_explanation(
                &mut explanation_state,
                seed.resource_identity(),
                GpuInitializationExplanationKind::RetainedCoverage,
                initial_coverage_value(coverage),
            );
        }
    }
    for fragment in fragments {
        for input in fragment.inputs() {
            let resource = input.initialized_coverage().storage_resource;
            let coverage = initial_coverage_value(input.initialized_coverage());
            union_state_coverage(graph_label, fragment, &mut state, resource, &coverage)?;
            record_pending_initialization_explanation(
                &mut explanation_state,
                resource,
                GpuInitializationExplanationKind::DeclaredInput,
                coverage,
            );
        }
    }
    let initial = state.clone();
    let mut failure_preserved = initial.clone();
    apply_global_prepared_initial_content(
        graph_label,
        &mut state,
        initial_content,
        &mut explanation_state,
    )?;
    for &prepared_id in topological_order {
        let Some(&(fragment_index, node_index)) = node_locations.get(&prepared_id) else {
            return Err(GpuWorkGraphError::invalid(
                "simulate prepared GPU work initialization",
                GpuWorkGraphErrorContext::new(
                    graph_label,
                    None,
                    None,
                    Some(prepared_id),
                    None,
                    None,
                    None,
                ),
                GpuWorkGraphCause::UnknownIdentity,
                "retain every topological identity in the prepared node table",
            ));
        };
        let fragment = &fragments[fragment_index];
        let node = &fragment.nodes()[node_index];
        apply_node_initialization(
            graph_label,
            fragment,
            node,
            prepared_id,
            &mut state,
            Some(&mut explanation_state),
        )?;
        for discarded in operation_discard_regions(node.operation()) {
            let Some(coverage) = failure_preserved.get_mut(&discarded.resource) else {
                continue;
            };
            if !coverage.remove(&discarded.coverage) {
                return Err(graph_error(
                    "derive GPU failure-preserved initialization coverage",
                    graph_label,
                    GraphErrorOrigin::new(Some(fragment), Some(node)),
                    Some(prepared_id),
                    Some(discarded.resource),
                    GpuWorkGraphCause::OperationAccessContradiction,
                    "apply possible discard invalidation only to matching normalized texture storage",
                ));
            }
        }
    }
    let mut summaries = Vec::new();
    let mut failure_preserved_coverage = BTreeMap::new();
    for (identity, resource) in storage_resources {
        let initial_value = initial
            .get(identity)
            .cloned()
            .unwrap_or_else(|| InitializedCoverage::empty_for(resource));
        let final_value = state
            .get(identity)
            .cloned()
            .unwrap_or_else(|| InitializedCoverage::empty_for(resource));
        let failure_preserved_value = failure_preserved
            .get(identity)
            .cloned()
            .unwrap_or_else(|| InitializedCoverage::empty_for(resource));
        if retained_storage.contains(identity) {
            if let Some(coverage) =
                coverage_to_public(graph_label, resource, &failure_preserved_value)?
            {
                failure_preserved_coverage.insert(*identity, coverage);
            }
        }
        let mut explanations = Vec::new();
        for pending in explanation_state.remove(identity).unwrap_or_default() {
            if let Some(coverage) = coverage_to_public(graph_label, resource, &pending.coverage)? {
                explanations.push(GpuInitializationExplanation {
                    kind: pending.kind,
                    coverage,
                });
            }
        }
        let summary = GpuPreparedResourceInitialization {
            resource: resource.clone(),
            initial: coverage_to_public(graph_label, resource, &initial_value)?,
            final_coverage: coverage_to_public(graph_label, resource, &final_value)?,
            explanations,
        };
        summaries.push(summary);
    }
    let diagnostics = summaries
        .iter()
        .cloned()
        .map(GpuPreparedWorkDiagnostic::ResourceInitialization)
        .collect();
    Ok((summaries, diagnostics, failure_preserved_coverage))
}

fn coverage_to_public(
    graph_label: &str,
    resource: &GpuResourceRef,
    coverage: &InitializedCoverage,
) -> Result<Option<GpuInitialCoverage>, GpuWorkGraphError> {
    if coverage.is_empty() {
        return Ok(None);
    }
    let value = match (resource, coverage) {
        (
            GpuResourceRef::Buffer(buffer),
            InitializedCoverage::Buffer {
                buffer: coverage_buffer,
                values,
            },
        ) if buffer == coverage_buffer => {
            let mut values = values.clone();
            normalize_buffer_coverage(buffer, &mut values);
            GpuInitialCoverage {
                resource: resource.clone(),
                storage_resource: buffer.diagnostic_identity(),
                data: GpuInitialCoverageData::Buffer(values),
            }
        }
        (GpuResourceRef::Texture(texture), InitializedCoverage::Texture(subresources)) => {
            let mut ranges = Vec::new();
            for (&(mip, aspect), intervals) in subresources {
                for &(layer_start, layer_end) in intervals {
                    ranges.push(
                        GpuTextureSubresourceRange::new(
                            texture.descriptor().common().label(),
                            mip,
                            1,
                            layer_start,
                            layer_end - layer_start,
                            aspect,
                        )
                        .map_err(|_| {
                            GpuWorkGraphError::invalid(
                                "publish prepared texture initialization",
                                GpuWorkGraphErrorContext::new(
                                    graph_label,
                                    None,
                                    None,
                                    None,
                                    Some(texture.diagnostic_identity()),
                                    Some(format!("{coverage:?}")),
                                    Some(texture.descriptor().common().provenance().clone()),
                                ),
                                GpuWorkGraphCause::OperationAccessContradiction,
                                "retain checked texture coverage during preparation",
                            )
                        })?,
                    );
                }
            }
            GpuInitialCoverage {
                resource: resource.clone(),
                storage_resource: texture.diagnostic_identity(),
                data: GpuInitialCoverageData::TextureSubresources(ranges),
            }
        }
        (GpuResourceRef::QuerySet(query_set), InitializedCoverage::Query(intervals)) => {
            let ranges = intervals
                .iter()
                .map(|&(start, end)| {
                    GpuQueryRange::new(query_set, start, end - start).map_err(|source| {
                        GpuWorkGraphError::with_source(
                            "publish prepared query initialization",
                            GpuWorkGraphErrorContext::new(
                                graph_label,
                                None,
                                None,
                                None,
                                Some(query_set.diagnostic_identity()),
                                Some(format!("queries {start}..{end}")),
                                Some(query_set.descriptor().common().provenance().clone()),
                            ),
                            GpuWorkGraphCause::OperationAccessContradiction,
                            "retain checked query coverage during preparation",
                            GpuWorkGraphErrorSource::Authoring(coverage_source_error(
                                "publish prepared query initialization",
                                query_set.diagnostic_identity(),
                                source,
                            )),
                        )
                    })
                })
                .collect::<Result<Vec<_>, _>>()?;
            GpuInitialCoverage {
                resource: resource.clone(),
                storage_resource: query_set.diagnostic_identity(),
                data: GpuInitialCoverageData::QueryRanges(ranges),
            }
        }
        (GpuResourceRef::Sampler(_), InitializedCoverage::Immutable) => {
            GpuInitialCoverage::descriptor_initialization(resource.clone()).map_err(|source| {
                GpuWorkGraphError::with_source(
                    "publish prepared sampler initialization",
                    GpuWorkGraphErrorContext::new(
                        graph_label,
                        None,
                        None,
                        None,
                        Some(resource.diagnostic_identity()),
                        None,
                        Some(resource.common().provenance().clone()),
                    ),
                    GpuWorkGraphCause::OperationAccessContradiction,
                    "retain immutable sampler evidence",
                    GpuWorkGraphErrorSource::Authoring(source),
                )
            })?
        }
        _ => {
            return Err(GpuWorkGraphError::invalid(
                "publish prepared resource initialization",
                GpuWorkGraphErrorContext::new(
                    graph_label,
                    None,
                    None,
                    None,
                    Some(storage_identity(resource)),
                    Some(format!("{coverage:?}")),
                    Some(resource.common().provenance().clone()),
                ),
                GpuWorkGraphCause::OperationAccessContradiction,
                "retain coverage with the matching normalized resource kind",
            ));
        }
    };
    Ok(Some(value))
}
