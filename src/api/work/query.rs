use super::super::{
    GpuAccessCause, GpuBufferAccess, GpuBufferAccessKind, GpuBufferHandle, GpuBufferRange,
    GpuQueryAccess, GpuQueryAccessKind, GpuQueryKind, GpuQueryRange, GpuQuerySetHandle,
    GpuWorkOperationCause, GpuWorkOperationError,
};

/// Explicit backend-neutral timestamp writes for one compute or render pass.
///
/// Timestamp query access alone is not enough execution meaning: a later executor must know whether
/// a query slot is written at the beginning or the end of the pass. This value owns that semantic
/// placement and derives the exact one-slot query accesses used by G3 hazard tracking.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GpuTimestampWrites {
    query_set: GpuQuerySetHandle,
    beginning_of_pass: Option<u32>,
    end_of_pass: Option<u32>,
    accesses: Vec<GpuQueryAccess>,
}

impl GpuTimestampWrites {
    pub fn new(
        query_set: &GpuQuerySetHandle,
        beginning_of_pass: Option<u32>,
        end_of_pass: Option<u32>,
    ) -> Result<Self, GpuWorkOperationError> {
        if query_set.descriptor().kind() != GpuQueryKind::Timestamp {
            return Err(GpuWorkOperationError::invalid(
                "construct GPU timestamp writes",
                query_set.descriptor().common().label().as_str(),
                Some(query_set.diagnostic_identity()),
                GpuWorkOperationCause::InvalidQueryRange,
                "use a timestamp query set for pass timestamp writes",
            ));
        }
        if beginning_of_pass.is_none() && end_of_pass.is_none() {
            return Err(GpuWorkOperationError::invalid(
                "construct GPU timestamp writes",
                query_set.descriptor().common().label().as_str(),
                Some(query_set.diagnostic_identity()),
                GpuWorkOperationCause::ZeroWork,
                "provide a beginning-of-pass query, an end-of-pass query, or both",
            ));
        }
        if beginning_of_pass.is_some() && beginning_of_pass == end_of_pass {
            return Err(GpuWorkOperationError::invalid(
                "construct GPU timestamp writes",
                query_set.descriptor().common().label().as_str(),
                Some(query_set.diagnostic_identity()),
                GpuWorkOperationCause::OperationAccessContradiction,
                "use distinct query slots when both beginning-of-pass and end-of-pass timestamps are written",
            ));
        }

        let mut accesses = Vec::with_capacity(
            usize::from(beginning_of_pass.is_some()) + usize::from(end_of_pass.is_some()),
        );
        for index in [beginning_of_pass, end_of_pass].into_iter().flatten() {
            let range = GpuQueryRange::new(query_set, index, 1).map_err(|source| {
                GpuWorkOperationError::from_access(
                    "construct GPU timestamp write range",
                    query_set.descriptor().common().label().as_str(),
                    GpuWorkOperationCause::InvalidQueryRange,
                    "keep every timestamp write index inside the timestamp query set",
                    source,
                )
            })?;
            accesses.push(
                GpuQueryAccess::new(query_set, range, GpuQueryAccessKind::WriteTimestamp).map_err(
                    |source| {
                        GpuWorkOperationError::from_access(
                            "construct GPU timestamp write access",
                            query_set.descriptor().common().label().as_str(),
                            GpuWorkOperationCause::InvalidQueryRange,
                            "retain a checked one-slot timestamp write access",
                            source,
                        )
                    },
                )?,
            );
        }

        Ok(Self {
            query_set: query_set.clone(),
            beginning_of_pass,
            end_of_pass,
            accesses,
        })
    }

    pub fn query_set(&self) -> &GpuQuerySetHandle {
        &self.query_set
    }

    pub const fn beginning_of_pass(&self) -> Option<u32> {
        self.beginning_of_pass
    }

    pub const fn end_of_pass(&self) -> Option<u32> {
        self.end_of_pass
    }

    pub fn accesses(&self) -> &[GpuQueryAccess] {
        &self.accesses
    }
}


#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GpuTimestampMarkerOperation {
    query_set: GpuQuerySetHandle,
    query_index: u32,
    access: GpuQueryAccess,
}

impl GpuTimestampMarkerOperation {
    pub fn new(
        query_set: &GpuQuerySetHandle,
        query_index: u32,
    ) -> Result<Self, GpuWorkOperationError> {
        if query_set.descriptor().kind() != GpuQueryKind::Timestamp {
            return Err(GpuWorkOperationError::invalid(
                "construct GPU timestamp marker operation",
                query_set.descriptor().common().label().as_str(),
                Some(query_set.diagnostic_identity()),
                GpuWorkOperationCause::InvalidQueryRange,
                "use a timestamp query set for an ordered timestamp marker",
            ));
        }
        let range = GpuQueryRange::new(query_set, query_index, 1).map_err(|source| {
            GpuWorkOperationError::from_access(
                "construct GPU timestamp marker range",
                query_set.descriptor().common().label().as_str(),
                GpuWorkOperationCause::InvalidQueryRange,
                "keep the timestamp marker index inside the timestamp query set",
                source,
            )
        })?;
        let access =
            GpuQueryAccess::new(query_set, range, GpuQueryAccessKind::WriteTimestamp).map_err(
                |source| {
                    GpuWorkOperationError::from_access(
                        "construct GPU timestamp marker access",
                        query_set.descriptor().common().label().as_str(),
                        GpuWorkOperationCause::InvalidQueryRange,
                        "retain a checked one-slot timestamp marker write access",
                        source,
                    )
                },
            )?;
        Ok(Self {
            query_set: query_set.clone(),
            query_index,
            access,
        })
    }

    pub fn query_set(&self) -> &GpuQuerySetHandle {
        &self.query_set
    }

    pub const fn query_index(&self) -> u32 {
        self.query_index
    }

    pub fn access(&self) -> &GpuQueryAccess {
        &self.access
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuQueryResolveOperation {
    source: GpuQuerySetHandle,
    source_range: GpuQueryRange,
    destination: GpuBufferHandle,
    destination_offset: u64,
    destination_range: GpuBufferRange,
    source_access: GpuQueryAccess,
    destination_access: GpuBufferAccess,
}

impl GpuQueryResolveOperation {
    pub fn new(
        source: &GpuQuerySetHandle,
        source_range: GpuQueryRange,
        destination: &GpuBufferHandle,
        destination_offset: u64,
    ) -> Result<Self, GpuWorkOperationError> {
        if source.descriptor().kind() != GpuQueryKind::Timestamp {
            return Err(GpuWorkOperationError::invalid(
                "construct GPU query resolve operation",
                source.descriptor().common().label().as_str(),
                Some(source.diagnostic_identity()),
                GpuWorkOperationCause::InvalidQueryResolution,
                "use a timestamp query set for the current G3 resolve operation",
            ));
        }
        let byte_len = u64::from(source_range.count())
            .checked_mul(8)
            .ok_or_else(|| {
                GpuWorkOperationError::invalid(
                    "construct GPU query resolve operation",
                    source.descriptor().common().label().as_str(),
                    Some(source.diagnostic_identity()),
                    GpuWorkOperationCause::QueryDestinationOverflow,
                    "reduce the query count",
                )
            })?;
        let destination_range = GpuBufferRange::new(destination, destination_offset, byte_len)
            .map_err(|source| {
                let cause = match source.cause() {
                    GpuAccessCause::ArithmeticOverflow => {
                        GpuWorkOperationCause::QueryDestinationOverflow
                    }
                    _ => GpuWorkOperationCause::QueryDestinationOutOfBounds,
                };
                GpuWorkOperationError::from_access(
                    "construct GPU query resolve destination",
                    destination.descriptor().common().label().as_str(),
                    cause,
                    "keep count-times-eight bytes at the destination offset inside the buffer",
                    source,
                )
            })?;
        let source_access =
            GpuQueryAccess::new(source, source_range, GpuQueryAccessKind::ResolveSource).map_err(
                |source| {
                    GpuWorkOperationError::from_access(
                        "construct GPU query resolve source",
                        "query resolve",
                        GpuWorkOperationCause::InvalidQueryRange,
                        "provide a checked query range",
                        source,
                    )
                },
            )?;
        let destination_access = GpuBufferAccess::new(
            destination,
            destination_range,
            GpuBufferAccessKind::QueryResolveDestination,
        )
        .map_err(|source| {
            GpuWorkOperationError::from_access(
                "construct GPU query resolve destination",
                destination.descriptor().common().label().as_str(),
                GpuWorkOperationCause::InvalidQueryResolution,
                "declare QueryResolve usage on the destination buffer",
                source,
            )
        })?;
        Ok(Self {
            source: source.clone(),
            source_range,
            destination: destination.clone(),
            destination_offset,
            destination_range,
            source_access,
            destination_access,
        })
    }

    pub fn source(&self) -> &GpuQuerySetHandle {
        &self.source
    }
    pub const fn source_range(&self) -> GpuQueryRange {
        self.source_range
    }
    pub fn destination(&self) -> &GpuBufferHandle {
        &self.destination
    }
    pub const fn destination_offset(&self) -> u64 {
        self.destination_offset
    }
    pub const fn destination_range(&self) -> GpuBufferRange {
        self.destination_range
    }
    pub fn source_access(&self) -> &GpuQueryAccess {
        &self.source_access
    }
    pub fn destination_access(&self) -> &GpuBufferAccess {
        &self.destination_access
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        GpuCapabilityFeature, GpuCapabilityRequirement, GpuMemoryIntent, GpuQuerySetDescriptor,
        GpuReconstruction, GpuResourceCommon, GpuResourceLabel, GpuResourceLifetime,
        GpuResourceProvenance, GpuWorkOperation, GpuWorkResourceIdAllocator,
    };

    fn query_set(count: u32) -> GpuQuerySetHandle {
        let mut allocator = GpuWorkResourceIdAllocator::new();
        let label = GpuResourceLabel::new("timestamp marker queries").unwrap();
        let provenance = GpuResourceProvenance::new(label.clone(), None, None);
        let common = GpuResourceCommon::owned(
            label,
            GpuResourceLifetime::Transient,
            GpuMemoryIntent::Device,
            GpuReconstruction::SourceBacked,
            provenance,
        )
        .unwrap();
        allocator
            .allocate_query_set_handle(
                GpuQuerySetDescriptor::new(common, GpuQueryKind::Timestamp, count).unwrap(),
            )
            .unwrap()
    }

    #[test]
    fn timestamp_marker_is_one_checked_timestamp_write() {
        let queries = query_set(2);
        let marker = GpuTimestampMarkerOperation::new(&queries, 1).unwrap();
        assert_eq!(marker.query_set(), &queries);
        assert_eq!(marker.query_index(), 1);
        assert_eq!(marker.access().kind(), GpuQueryAccessKind::WriteTimestamp);
        assert_eq!(
            marker.access().range(),
            GpuQueryRange::new(&queries, 1, 1).unwrap()
        );

        let operation = GpuWorkOperation::TimestampMarker(marker);
        let accesses = operation.derived_accesses().unwrap();
        assert_eq!(accesses.len(), 1);
        assert_eq!(
            accesses[0].resource_identity(),
            queries.diagnostic_identity()
        );
        assert!(
            operation
                .derived_requirements()
                .unwrap()
                .iter()
                .any(|requirement| matches!(
                    requirement,
                    GpuCapabilityRequirement::Required(GpuCapabilityFeature::TimestampQuery)
                ))
        );
    }

    #[test]
    fn timestamp_marker_rejects_out_of_range_index() {
        let queries = query_set(2);
        let error = GpuTimestampMarkerOperation::new(&queries, 2).unwrap_err();
        assert_eq!(error.cause(), GpuWorkOperationCause::InvalidQueryRange);
    }
}
