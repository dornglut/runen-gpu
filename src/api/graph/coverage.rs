use super::super::{
    GpuAccessError, GpuBufferHandle, GpuBufferRange, GpuQueryRange, GpuQuerySetHandle,
    GpuResourceProvenance, GpuResourceRef, GpuTextureAccessResource, GpuTextureAspect,
    GpuTextureHandle, GpuTextureSubresourceRange, GpuWorkAuthoringCause, GpuWorkAuthoringError,
    GpuWorkAuthoringErrorContext, GpuWorkAuthoringErrorSource, GpuWorkResourceId,
};
use std::collections::BTreeMap;
use std::{cmp::Reverse, collections::BinaryHeap};

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuInitialCoverageKind {
    DescriptorInitialization,
    Buffer,
    TextureSubresources,
    QueryRanges,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum GpuInitialCoverageData {
    DescriptorInitialization,
    Buffer(Vec<GpuBufferCoverage>),
    TextureSubresources(Vec<GpuTextureSubresourceRange>),
    QueryRanges(Vec<GpuQueryRange>),
}

/// Exact initialized buffer coverage without expanding padded rows or images.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuBufferCoverage {
    Dense(GpuBufferRange),
    Strided(GpuBufferStridedCoverage),
}

impl GpuBufferCoverage {
    pub const fn dense(range: GpuBufferRange) -> Self {
        Self::Dense(range)
    }

    pub const fn strided(coverage: GpuBufferStridedCoverage) -> Self {
        Self::Strided(coverage)
    }

    pub const fn first(&self) -> u64 {
        match self {
            Self::Dense(range) => range.offset(),
            Self::Strided(coverage) => coverage.first(),
        }
    }

    fn fast_contains(&self, required: &Self) -> bool {
        match (self, required) {
            (Self::Dense(have), Self::Dense(required)) => {
                have.offset() <= required.offset() && have.end() >= required.end()
            }
            (Self::Dense(have), Self::Strided(required)) => {
                have.offset() <= required.first && have.end() >= required.end
            }
            (Self::Strided(have), Self::Dense(required)) => {
                strided_coverage_fast_contains_dense(have, required)
            }
            (Self::Strided(have), Self::Strided(required)) => {
                strided_coverage_fast_contains(have, required)
            }
        }
    }
}

/// A repeated exact buffer segment layout, used for padded buffer-texture copies.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuBufferStridedCoverage {
    buffer: GpuBufferHandle,
    first: u64,
    segment_size: u64,
    segment_stride: u64,
    segment_count: u32,
    group_stride: u64,
    group_count: u32,
    end: u64,
}

impl GpuBufferStridedCoverage {
    pub fn new(
        buffer: &GpuBufferHandle,
        first: u64,
        segment_size: u64,
        segment_stride: u64,
        segment_count: u32,
        group_stride: u64,
        group_count: u32,
    ) -> Result<Self, GpuWorkAuthoringError> {
        if segment_size == 0 || segment_count == 0 || group_count == 0 {
            return Err(coverage_error(
                "construct strided buffer coverage",
                buffer.diagnostic_identity(),
                "provide nonempty segment and group counts",
            ));
        }
        if segment_stride < segment_size {
            return Err(coverage_error(
                "construct strided buffer coverage",
                buffer.diagnostic_identity(),
                "keep each segment stride at least as large as the segment size",
            ));
        }
        let group_payload = u64::from(segment_count - 1)
            .checked_mul(segment_stride)
            .and_then(|value| value.checked_add(segment_size))
            .ok_or_else(|| {
                coverage_error(
                    "construct strided buffer coverage",
                    buffer.diagnostic_identity(),
                    "reduce the strided segment layout to avoid arithmetic overflow",
                )
            })?;
        if group_count > 1 && group_stride < group_payload {
            return Err(coverage_error(
                "construct strided buffer coverage",
                buffer.diagnostic_identity(),
                "keep each group stride large enough for all of its segments",
            ));
        }
        let end = first
            .checked_add(
                u64::from(group_count - 1)
                    .checked_mul(group_stride)
                    .and_then(|value| value.checked_add(group_payload))
                    .ok_or_else(|| {
                        coverage_error(
                            "construct strided buffer coverage",
                            buffer.diagnostic_identity(),
                            "reduce the strided group layout to avoid arithmetic overflow",
                        )
                    })?,
            )
            .ok_or_else(|| {
                coverage_error(
                    "construct strided buffer coverage",
                    buffer.diagnostic_identity(),
                    "reduce the strided buffer offset to avoid arithmetic overflow",
                )
            })?;
        if first >= buffer.descriptor().size_bytes() || end > buffer.descriptor().size_bytes() {
            return Err(coverage_error(
                "construct strided buffer coverage",
                buffer.diagnostic_identity(),
                "keep every strided segment inside the buffer descriptor",
            ));
        }
        Ok(Self {
            buffer: buffer.clone(),
            first,
            segment_size,
            segment_stride,
            segment_count,
            group_stride,
            group_count,
            end,
        })
    }

    pub fn buffer(&self) -> &GpuBufferHandle {
        &self.buffer
    }
    pub const fn first(&self) -> u64 {
        self.first
    }
    pub const fn segment_size(&self) -> u64 {
        self.segment_size
    }
    pub const fn segment_stride(&self) -> u64 {
        self.segment_stride
    }
    pub const fn segment_count(&self) -> u32 {
        self.segment_count
    }
    pub const fn group_stride(&self) -> u64 {
        self.group_stride
    }
    pub const fn group_count(&self) -> u32 {
        self.group_count
    }
    pub const fn end(&self) -> u64 {
        self.end
    }
}

fn strided_coverage_fast_contains(
    have: &GpuBufferStridedCoverage,
    required: &GpuBufferStridedCoverage,
) -> bool {
    if have == required {
        return true;
    }
    if have.segment_size < required.segment_size
        || have.segment_stride != required.segment_stride
        || have.group_stride != required.group_stride
        || required.first < have.first
    {
        return false;
    }
    let offset = required.first - have.first;
    let (group_offset, within_group) = if have.group_count == 1 {
        (0, offset)
    } else {
        (offset / have.group_stride, offset % have.group_stride)
    };
    let segment_offset = within_group / have.segment_stride;
    within_group == segment_offset * have.segment_stride
        && group_offset
            .checked_add(u64::from(required.group_count))
            .is_some_and(|end| end <= u64::from(have.group_count))
        && segment_offset
            .checked_add(u64::from(required.segment_count))
            .is_some_and(|end| end <= u64::from(have.segment_count))
}

fn strided_coverage_fast_contains_dense(
    have: &GpuBufferStridedCoverage,
    required: &GpuBufferRange,
) -> bool {
    if required.offset() < have.first || required.end() > have.end {
        return false;
    }
    if have.segment_stride == have.segment_size
        && (have.group_count == 1
            || have.group_stride
                == u64::from(have.segment_count - 1) * have.segment_stride + have.segment_size)
    {
        return true;
    }

    let offset = required.offset() - have.first;
    let group = if have.group_count == 1 {
        0
    } else {
        offset / have.group_stride
    };
    if group >= u64::from(have.group_count) {
        return false;
    }
    let group_start = have.first + group * have.group_stride;
    let within_group = required.offset() - group_start;
    if have.segment_stride == have.segment_size {
        let group_end = group_start
            + u64::from(have.segment_count - 1) * have.segment_stride
            + have.segment_size;
        return required.end() <= group_end;
    }

    let segment = within_group / have.segment_stride;
    if segment >= u64::from(have.segment_count) {
        return false;
    }
    let segment_start = group_start + segment * have.segment_stride;
    required.end() <= segment_start + have.segment_size
}

pub(super) fn normalize_buffer_coverage(
    buffer: &GpuBufferHandle,
    values: &mut Vec<GpuBufferCoverage>,
) {
    let mut strided = Vec::new();
    for value in values.drain(..) {
        match value {
            GpuBufferCoverage::Dense(range) => strided.push(GpuBufferCoverage::Dense(range)),
            GpuBufferCoverage::Strided(coverage) => {
                strided.push(canonical_strided_coverage(coverage));
            }
        }
    }
    let mut dense = strided
        .iter()
        .filter_map(|value| match value {
            GpuBufferCoverage::Dense(range) => Some((range.offset(), range.end())),
            GpuBufferCoverage::Strided(_) => None,
        })
        .collect::<Vec<_>>();
    let mut normalized = normalize_u64_intervals(core::mem::take(&mut dense))
        .into_iter()
        .map(|(start, end)| {
            GpuBufferCoverage::Dense(
                GpuBufferRange::new(buffer, start, end - start)
                    .expect("normalized checked buffer coverage remains checked"),
            )
        })
        .collect::<Vec<_>>();
    normalized.extend(strided.into_iter().filter_map(|value| match value {
        GpuBufferCoverage::Dense(_) => None,
        GpuBufferCoverage::Strided(coverage) => Some(GpuBufferCoverage::Strided(coverage)),
    }));
    normalized.sort();
    normalized.dedup();
    *values = normalized
        .iter()
        .enumerate()
        .filter(|(index, value)| {
            !normalized.iter().enumerate().any(|(other_index, other)| {
                index != &other_index
                    && other.fast_contains(value)
                    && (!value.fast_contains(other) || other_index < *index)
            })
        })
        .map(|(_, value)| value.clone())
        .collect();
}

fn canonical_strided_coverage(coverage: GpuBufferStridedCoverage) -> GpuBufferCoverage {
    let segment_stride = if coverage.segment_count == 1 {
        coverage.segment_size
    } else {
        coverage.segment_stride
    };
    let group_stride = if coverage.group_count == 1 {
        0
    } else {
        coverage.group_stride
    };
    let group_payload =
        u64::from(coverage.segment_count - 1) * segment_stride + coverage.segment_size;
    if (coverage.segment_count == 1 && coverage.group_count == 1)
        || (segment_stride == coverage.segment_size
            && (coverage.group_count == 1 || coverage.group_stride == group_payload))
    {
        return GpuBufferCoverage::Dense(
            GpuBufferRange::new(
                &coverage.buffer,
                coverage.first,
                coverage.end - coverage.first,
            )
            .expect("canonical contiguous strided coverage remains checked"),
        );
    }
    GpuBufferCoverage::Strided(
        GpuBufferStridedCoverage::new(
            &coverage.buffer,
            coverage.first,
            coverage.segment_size,
            segment_stride,
            coverage.segment_count,
            group_stride,
            coverage.group_count,
        )
        .expect("canonical strided coverage remains checked"),
    )
}

pub(super) fn buffer_coverage_contains(
    have: &[GpuBufferCoverage],
    required: &[GpuBufferCoverage],
) -> bool {
    if required
        .iter()
        .all(|required| have.iter().any(|coverage| coverage.fast_contains(required)))
    {
        return true;
    }
    let mut available = CoverageIntervals::new(have);
    let mut available_interval = available.next();
    for (required_start, required_end) in CoverageIntervals::new(required) {
        let mut cursor = required_start;
        while cursor < required_end {
            while available_interval.is_some_and(|(_, end)| end <= cursor) {
                available_interval = available.next();
            }
            let Some((start, end)) = available_interval else {
                return false;
            };
            if start > cursor {
                return false;
            }
            cursor = end.min(required_end);
        }
    }
    true
}

pub(super) fn intersect_buffer_coverage(
    buffer: &GpuBufferHandle,
    left: &[GpuBufferCoverage],
    right: &[GpuBufferCoverage],
) -> Vec<GpuBufferCoverage> {
    if buffer_coverage_contains(left, right) {
        let mut intersection = right.to_vec();
        normalize_buffer_coverage(buffer, &mut intersection);
        return intersection;
    }
    if buffer_coverage_contains(right, left) {
        let mut intersection = left.to_vec();
        normalize_buffer_coverage(buffer, &mut intersection);
        return intersection;
    }

    let mut left_intervals = CoverageIntervals::new(left);
    let mut right_intervals = CoverageIntervals::new(right);
    let mut left_interval = left_intervals.next();
    let mut right_interval = right_intervals.next();
    let mut intersection = Vec::new();

    while let (Some((left_start, left_end)), Some((right_start, right_end))) =
        (left_interval, right_interval)
    {
        let start = left_start.max(right_start);
        let end = left_end.min(right_end);
        if start < end {
            intersection.push(GpuBufferCoverage::Dense(
                GpuBufferRange::new(buffer, start, end - start)
                    .expect("intersection of checked buffer coverage remains checked"),
            ));
        }
        if left_end <= right_end {
            left_interval = left_intervals.next();
        }
        if right_end <= left_end {
            right_interval = right_intervals.next();
        }
    }

    normalize_buffer_coverage(buffer, &mut intersection);
    intersection
}

struct CoverageIntervals<'a> {
    cursors: Vec<CoverageIntervalCursor<'a>>,
    next: BinaryHeap<Reverse<(u64, u64, usize)>>,
}

impl<'a> CoverageIntervals<'a> {
    fn new(values: &'a [GpuBufferCoverage]) -> Self {
        let mut cursors = values
            .iter()
            .map(CoverageIntervalCursor::new)
            .collect::<Vec<_>>();
        let mut next = BinaryHeap::new();
        for (index, cursor) in cursors.iter_mut().enumerate() {
            if let Some((start, end)) = cursor.next() {
                next.push(Reverse((start, end, index)));
            }
        }
        Self { cursors, next }
    }

    fn push_next(&mut self, index: usize) {
        if let Some((start, end)) = self.cursors[index].next() {
            self.next.push(Reverse((start, end, index)));
        }
    }
}

impl Iterator for CoverageIntervals<'_> {
    type Item = (u64, u64);

    fn next(&mut self) -> Option<Self::Item> {
        let Reverse((start, mut end, index)) = self.next.pop()?;
        self.push_next(index);
        while self.next.peek().is_some_and(|entry| entry.0.0 <= end) {
            let Reverse((_, candidate_end, index)) = self.next.pop().expect("peeked entry exists");
            end = end.max(candidate_end);
            self.push_next(index);
        }
        Some((start, end))
    }
}

struct CoverageIntervalCursor<'a> {
    coverage: &'a GpuBufferCoverage,
    next_segment: u64,
}

impl<'a> CoverageIntervalCursor<'a> {
    fn new(coverage: &'a GpuBufferCoverage) -> Self {
        Self {
            coverage,
            next_segment: 0,
        }
    }

    fn next(&mut self) -> Option<(u64, u64)> {
        match self.coverage {
            GpuBufferCoverage::Dense(range) if self.next_segment == 0 => {
                self.next_segment = 1;
                Some((range.offset(), range.end()))
            }
            GpuBufferCoverage::Dense(_) => None,
            GpuBufferCoverage::Strided(coverage) => {
                let count = u64::from(coverage.segment_count) * u64::from(coverage.group_count);
                if self.next_segment == count {
                    return None;
                }
                let group = self.next_segment / u64::from(coverage.segment_count);
                let segment = self.next_segment % u64::from(coverage.segment_count);
                self.next_segment += 1;
                let start = coverage.first
                    + group * coverage.group_stride
                    + segment * coverage.segment_stride;
                Some((start, start + coverage.segment_size))
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct GpuInitialCoverage {
    pub(super) resource: GpuResourceRef,
    pub(super) storage_resource: GpuWorkResourceId,
    pub(super) data: GpuInitialCoverageData,
}

impl PartialEq for GpuInitialCoverage {
    fn eq(&self, other: &Self) -> bool {
        self.resource == other.resource
            && self.storage_resource == other.storage_resource
            && match (&self.data, &other.data) {
                (GpuInitialCoverageData::Buffer(left), GpuInitialCoverageData::Buffer(right)) => {
                    buffer_coverage_contains(left, right) && buffer_coverage_contains(right, left)
                }
                _ => self.data == other.data,
            }
    }
}

impl Eq for GpuInitialCoverage {}

impl GpuInitialCoverage {
    pub fn descriptor_initialization(
        resource: GpuResourceRef,
    ) -> Result<Self, GpuWorkAuthoringError> {
        if matches!(resource, GpuResourceRef::QuerySet(_)) {
            return Err(coverage_error(
                "construct descriptor initialization coverage",
                resource.diagnostic_identity(),
                "use explicit checked query ranges because query descriptors contain no initialized indices",
            ));
        }
        let storage_resource = storage_identity(&resource);
        Ok(Self {
            resource,
            storage_resource,
            data: GpuInitialCoverageData::DescriptorInitialization,
        })
    }

    pub fn buffer(
        buffer: &GpuBufferHandle,
        values: impl IntoIterator<Item = GpuBufferCoverage>,
    ) -> Result<Self, GpuWorkAuthoringError> {
        let mut values = values.into_iter().collect::<Vec<_>>();
        if values.is_empty() {
            return Err(coverage_error(
                "construct initial buffer coverage",
                buffer.diagnostic_identity(),
                "provide at least one checked initialized buffer coverage value",
            ));
        }
        if values.iter().any(|value| match value {
            GpuBufferCoverage::Dense(range) => {
                GpuBufferRange::new(buffer, range.offset(), range.size()).is_err()
            }
            GpuBufferCoverage::Strided(coverage) => coverage.buffer() != buffer,
        }) {
            return Err(coverage_error(
                "construct initial buffer coverage",
                buffer.diagnostic_identity(),
                "provide coverage checked against the same buffer",
            ));
        }
        normalize_buffer_coverage(buffer, &mut values);
        Ok(Self {
            resource: GpuResourceRef::Buffer(buffer.clone()),
            storage_resource: buffer.diagnostic_identity(),
            data: GpuInitialCoverageData::Buffer(values),
        })
    }

    pub fn texture_subresources(
        resource: &GpuTextureAccessResource,
        ranges: impl IntoIterator<Item = GpuTextureSubresourceRange>,
    ) -> Result<Self, GpuWorkAuthoringError> {
        let parent = resource.parent_texture();
        let parent_aspect = texture_aspect(parent);
        let view_range = match resource {
            GpuTextureAccessResource::Texture(_) => None,
            GpuTextureAccessResource::TextureView(view) => Some(view.descriptor().subresources()),
        };
        let mut by_mip = BTreeMap::<(u32, GpuTextureAspect), Vec<(u32, u32)>>::new();
        for range in ranges {
            let checked =
                GpuTextureSubresourceRange::checked_for(parent, range).map_err(|source| {
                    coverage_source_error(
                        "construct initial texture coverage",
                        parent.diagnostic_identity(),
                        source,
                    )
                })?;
            if view_range.is_some_and(|view| !view.contains(checked, parent_aspect)) {
                return Err(coverage_error(
                    "construct initial texture-view coverage",
                    resource.diagnostic_identity(),
                    "keep initialized mip, layer, and aspect coverage inside the texture view",
                ));
            }
            let aspect = canonical_texture_aspect(checked.aspect(), parent_aspect);
            for mip in checked.base_mip_level()..checked.mip_end() {
                by_mip
                    .entry((mip, aspect))
                    .or_default()
                    .push((checked.base_array_layer(), checked.layer_end()));
            }
        }
        if by_mip.is_empty() {
            return Err(coverage_error(
                "construct initial texture coverage",
                resource.diagnostic_identity(),
                "provide at least one checked initialized texture subresource",
            ));
        }
        let mut normalized = Vec::new();
        for ((mip, aspect), intervals) in by_mip {
            for (layer_start, layer_end) in normalize_u32_intervals(intervals) {
                normalized.push(
                    GpuTextureSubresourceRange::new(
                        parent.descriptor().common().label(),
                        mip,
                        1,
                        layer_start,
                        layer_end - layer_start,
                        aspect,
                    )
                    .map_err(|_| {
                        coverage_error(
                            "normalize initial texture coverage",
                            parent.diagnostic_identity(),
                            "use checked texture subresource coverage",
                        )
                    })?,
                );
            }
        }
        let resource_ref = match resource {
            GpuTextureAccessResource::Texture(texture) => GpuResourceRef::Texture(texture.clone()),
            GpuTextureAccessResource::TextureView(view) => {
                GpuResourceRef::TextureView(view.clone())
            }
        };
        Ok(Self {
            resource: resource_ref,
            storage_resource: parent.diagnostic_identity(),
            data: GpuInitialCoverageData::TextureSubresources(normalized),
        })
    }

    pub fn query_ranges(
        query_set: &GpuQuerySetHandle,
        ranges: impl IntoIterator<Item = GpuQueryRange>,
    ) -> Result<Self, GpuWorkAuthoringError> {
        let mut intervals = Vec::new();
        for range in ranges {
            let checked =
                GpuQueryRange::new(query_set, range.first(), range.count()).map_err(|source| {
                    coverage_source_error(
                        "construct initial query coverage",
                        query_set.diagnostic_identity(),
                        source,
                    )
                })?;
            intervals.push((checked.first(), checked.end()));
        }
        if intervals.is_empty() {
            return Err(coverage_error(
                "construct initial query coverage",
                query_set.diagnostic_identity(),
                "provide at least one checked initialized query range",
            ));
        }
        let ranges = normalize_u32_intervals(intervals)
            .into_iter()
            .map(|(start, end)| {
                GpuQueryRange::new(query_set, start, end - start).map_err(|source| {
                    coverage_source_error(
                        "normalize initial query coverage",
                        query_set.diagnostic_identity(),
                        source,
                    )
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            resource: GpuResourceRef::QuerySet(query_set.clone()),
            storage_resource: query_set.diagnostic_identity(),
            data: GpuInitialCoverageData::QueryRanges(ranges),
        })
    }

    pub const fn kind(&self) -> GpuInitialCoverageKind {
        match self.data {
            GpuInitialCoverageData::DescriptorInitialization => {
                GpuInitialCoverageKind::DescriptorInitialization
            }
            GpuInitialCoverageData::Buffer(_) => GpuInitialCoverageKind::Buffer,
            GpuInitialCoverageData::TextureSubresources(_) => {
                GpuInitialCoverageKind::TextureSubresources
            }
            GpuInitialCoverageData::QueryRanges(_) => GpuInitialCoverageKind::QueryRanges,
        }
    }

    pub fn resource(&self) -> &GpuResourceRef {
        &self.resource
    }

    pub fn buffer_values(&self) -> Option<&[GpuBufferCoverage]> {
        match &self.data {
            GpuInitialCoverageData::Buffer(values) => Some(values),
            _ => None,
        }
    }

    pub fn texture_subresource_values(&self) -> Option<&[GpuTextureSubresourceRange]> {
        match &self.data {
            GpuInitialCoverageData::TextureSubresources(ranges) => Some(ranges),
            _ => None,
        }
    }

    pub fn query_range_values(&self) -> Option<&[GpuQueryRange]> {
        match &self.data {
            GpuInitialCoverageData::QueryRanges(ranges) => Some(ranges),
            _ => None,
        }
    }
}

fn coverage_error(
    operation: &'static str,
    resource: GpuWorkResourceId,
    correction: &'static str,
) -> GpuWorkAuthoringError {
    GpuWorkAuthoringError::invalid(
        operation,
        GpuWorkAuthoringErrorContext::new(None, None, None, Some(resource), None),
        GpuWorkAuthoringCause::InvalidCoverage,
        correction,
    )
}

pub(super) fn coverage_source_error(
    operation: &'static str,
    resource: GpuWorkResourceId,
    source: GpuAccessError,
) -> GpuWorkAuthoringError {
    GpuWorkAuthoringError::with_source(
        operation,
        GpuWorkAuthoringErrorContext::new(None, None, None, Some(resource), None),
        GpuWorkAuthoringCause::InvalidCoverage,
        "provide coverage checked against the same typed resource",
        GpuWorkAuthoringErrorSource::Access(source),
    )
}

pub(super) fn normalize_u64_intervals(mut intervals: Vec<(u64, u64)>) -> Vec<(u64, u64)> {
    intervals.sort_unstable();
    let mut normalized: Vec<(u64, u64)> = Vec::new();
    for (start, end) in intervals {
        if let Some(last) = normalized.last_mut() {
            if start <= last.1 {
                last.1 = last.1.max(end);
                continue;
            }
        }
        normalized.push((start, end));
    }
    normalized
}

pub(super) fn normalize_u32_intervals(mut intervals: Vec<(u32, u32)>) -> Vec<(u32, u32)> {
    intervals.sort_unstable();
    let mut normalized: Vec<(u32, u32)> = Vec::new();
    for (start, end) in intervals {
        if let Some(last) = normalized.last_mut() {
            if start <= last.1 {
                last.1 = last.1.max(end);
                continue;
            }
        }
        normalized.push((start, end));
    }
    normalized
}

pub(super) fn texture_aspect(texture: &GpuTextureHandle) -> GpuTextureAspect {
    if texture.descriptor().format().is_depth() {
        GpuTextureAspect::DepthOnly
    } else {
        GpuTextureAspect::Color
    }
}

pub(super) fn canonical_texture_aspect(
    aspect: GpuTextureAspect,
    parent: GpuTextureAspect,
) -> GpuTextureAspect {
    if aspect == GpuTextureAspect::All {
        parent
    } else {
        aspect
    }
}

pub(super) fn storage_identity(resource: &GpuResourceRef) -> GpuWorkResourceId {
    match resource {
        GpuResourceRef::TextureView(view) => view.descriptor().texture().diagnostic_identity(),
        _ => resource.diagnostic_identity(),
    }
}

pub(super) fn canonical_storage_resource(resource: &GpuResourceRef) -> GpuResourceRef {
    match resource {
        GpuResourceRef::TextureView(view) => {
            GpuResourceRef::Texture(view.descriptor().texture().clone())
        }
        _ => resource.clone(),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuWorkResourceInput {
    resource: GpuResourceRef,
    initialized_coverage: GpuInitialCoverage,
    provenance: GpuResourceProvenance,
}

impl GpuWorkResourceInput {
    pub fn new(
        resource: GpuResourceRef,
        initialized_coverage: GpuInitialCoverage,
        provenance: GpuResourceProvenance,
    ) -> Result<Self, GpuWorkAuthoringError> {
        if resource != *initialized_coverage.resource() {
            return Err(GpuWorkAuthoringError::invalid(
                "construct GPU work-resource input",
                GpuWorkAuthoringErrorContext::new(
                    None,
                    None,
                    None,
                    Some(resource.diagnostic_identity()),
                    Some(provenance),
                ),
                GpuWorkAuthoringCause::InvalidResourceKind,
                "bind initialized coverage checked against the same kind-preserving resource",
            ));
        }
        Ok(Self {
            resource,
            initialized_coverage,
            provenance,
        })
    }

    pub fn resource(&self) -> &GpuResourceRef {
        &self.resource
    }

    pub fn initialized_coverage(&self) -> &GpuInitialCoverage {
        &self.initialized_coverage
    }

    pub fn provenance(&self) -> &GpuResourceProvenance {
        &self.provenance
    }
}
