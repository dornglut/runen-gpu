use super::work::{
    GpuBufferTextureLayout, GpuClearOperation, GpuColorAttachmentLoad, GpuComputeOperation,
    GpuCopyOperation, GpuDepthAttachmentLoad, GpuDrawIntent, GpuPresentOperation,
    GpuQueryResolveOperation, GpuRenderColorAttachment, GpuRenderDepthStencilAttachment,
    GpuTextureCopyRegion, GpuTimestampWrites,
};
use super::{
    GpuBufferAccess, GpuBufferAccessKind, GpuBufferRange, GpuCapabilityFeature,
    GpuCapabilityRequirement, GpuCapabilityRequirementError, GpuCapabilityRequirements,
    GpuDepthStencilAccess, GpuReadbackOperation, GpuRenderDraw, GpuRenderPassSignature,
    GpuResourceAccess, GpuTextureAccess, GpuTextureAccessKind, GpuTextureAccessResource,
    GpuUploadOperation, GpuWorkOperationCause, GpuWorkOperationError,
    render_pass_usage::validate_render_pass_usage_scope,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuWorkNodeKind {
    Compute,
    Render,
    Copy,
    Clear,
    Resolve,
    Present,
    Upload,
    Readback,
}

/// One logical render pass with ordered execution-complete draws.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GpuRenderOperation {
    color_attachments: Vec<GpuRenderColorAttachment>,
    depth_stencil_attachment: Option<GpuRenderDepthStencilAttachment>,
    draws: Vec<GpuRenderDraw>,
    timestamp_writes: Option<GpuTimestampWrites>,
    signature: GpuRenderPassSignature,
    accesses: Vec<GpuResourceAccess>,
}

impl GpuRenderOperation {
    pub fn new(
        color_attachments: impl IntoIterator<Item = GpuRenderColorAttachment>,
        depth_stencil_attachment: Option<GpuRenderDepthStencilAttachment>,
        draws: impl IntoIterator<Item = GpuRenderDraw>,
        timestamp_writes: Option<GpuTimestampWrites>,
    ) -> Result<Self, GpuWorkOperationError> {
        let color_attachments = color_attachments.into_iter().collect::<Vec<_>>();
        let draws = draws.into_iter().collect::<Vec<_>>();

        if color_attachments.is_empty() && depth_stencil_attachment.is_none() {
            return Err(GpuWorkOperationError::invalid(
                "construct GPU render operation",
                "attachments=0",
                None,
                GpuWorkOperationCause::InvalidAttachment,
                "provide at least one explicit color or depth/stencil attachment view",
            ));
        }

        let clears_color = color_attachments
            .iter()
            .any(|attachment| matches!(attachment.load(), GpuColorAttachmentLoad::Clear(_)));
        let clears_depth = depth_stencil_attachment.as_ref().is_some_and(|attachment| {
            matches!(attachment.load(), GpuDepthAttachmentLoad::Clear(_))
        });
        if draws.is_empty() && !clears_color && !clears_depth && timestamp_writes.is_none() {
            return Err(GpuWorkOperationError::invalid(
                "construct GPU render operation",
                "render",
                None,
                GpuWorkOperationCause::ZeroWork,
                "add a draw, attachment clear, or timestamp write",
            ));
        }

        let signature = GpuRenderPassSignature::from_attachments(
            &color_attachments,
            depth_stencil_attachment.as_ref(),
        )?;

        for draw in &draws {
            signature.validate_draw(draw)?;
            validate_depth_access_for_draw(depth_stencil_attachment.as_ref(), draw)?;
        }

        let mut accesses = Vec::new();
        for attachment in &color_attachments {
            accesses.push(GpuResourceAccess::Texture(
                attachment.source_access().clone(),
            ));
            if let Some(resolve) = attachment.resolve_target() {
                accesses.push(GpuResourceAccess::Texture(resolve.access().clone()));
            }
        }
        if let Some(attachment) = &depth_stencil_attachment {
            accesses.push(GpuResourceAccess::Texture(
                attachment.source_access().clone(),
            ));
        }
        for draw in &draws {
            accesses.extend(draw.accesses().iter().cloned());
        }
        if let Some(timestamp_writes) = &timestamp_writes {
            accesses.extend(
                timestamp_writes
                    .accesses()
                    .iter()
                    .cloned()
                    .map(GpuResourceAccess::Query),
            );
        }

        validate_render_pass_usage_scope(&accesses)?;

        Ok(Self {
            color_attachments,
            depth_stencil_attachment,
            draws,
            timestamp_writes,
            signature,
            accesses,
        })
    }

    pub fn color_attachments(&self) -> &[GpuRenderColorAttachment] {
        &self.color_attachments
    }

    pub fn depth_stencil_attachment(&self) -> Option<&GpuRenderDepthStencilAttachment> {
        self.depth_stencil_attachment.as_ref()
    }

    pub fn draws(&self) -> &[GpuRenderDraw] {
        &self.draws
    }

    pub fn timestamp_writes(&self) -> Option<&GpuTimestampWrites> {
        self.timestamp_writes.as_ref()
    }

    pub fn signature(&self) -> &GpuRenderPassSignature {
        &self.signature
    }

    pub fn accesses(&self) -> &[GpuResourceAccess] {
        &self.accesses
    }
}

fn validate_depth_access_for_draw(
    attachment: Option<&GpuRenderDepthStencilAttachment>,
    draw: &GpuRenderDraw,
) -> Result<(), GpuWorkOperationError> {
    let Some(attachment) = attachment else {
        return Ok(());
    };
    if attachment.access() == GpuDepthStencilAccess::ReadOnly
        && draw
            .pipeline()
            .state()
            .depth_stencil()
            .is_some_and(|depth| depth.depth_write_enabled())
    {
        return Err(GpuWorkOperationError::invalid(
            "validate GPU render draw depth access",
            "read-only depth attachment with depth-writing pipeline",
            Some(
                attachment
                    .source()
                    .descriptor()
                    .texture()
                    .diagnostic_identity(),
            ),
            GpuWorkOperationCause::InvalidAttachment,
            "disable pipeline depth writes when the render pass uses a read-only depth attachment",
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GpuWorkOperation {
    Compute(GpuComputeOperation),
    Render(GpuRenderOperation),
    Copy(GpuCopyOperation),
    Clear(GpuClearOperation),
    Resolve(GpuQueryResolveOperation),
    Present(GpuPresentOperation),
    Upload(GpuUploadOperation),
    Readback(GpuReadbackOperation),
}

impl GpuWorkOperation {
    pub const fn kind(&self) -> GpuWorkNodeKind {
        match self {
            Self::Compute(_) => GpuWorkNodeKind::Compute,
            Self::Render(_) => GpuWorkNodeKind::Render,
            Self::Copy(_) => GpuWorkNodeKind::Copy,
            Self::Clear(_) => GpuWorkNodeKind::Clear,
            Self::Resolve(_) => GpuWorkNodeKind::Resolve,
            Self::Present(_) => GpuWorkNodeKind::Present,
            Self::Upload(_) => GpuWorkNodeKind::Upload,
            Self::Readback(_) => GpuWorkNodeKind::Readback,
        }
    }

    pub fn derived_accesses(&self) -> Result<Vec<GpuResourceAccess>, GpuWorkOperationError> {
        match self {
            Self::Compute(operation) => {
                let mut accesses = operation.bindings().accesses().to_vec();
                if let Some(access) = operation.dispatch().indirect_access() {
                    accesses.push(GpuResourceAccess::Buffer(access.clone()));
                }
                if let Some(timestamp_writes) = operation.timestamp_writes() {
                    accesses.extend(
                        timestamp_writes
                            .accesses()
                            .iter()
                            .cloned()
                            .map(GpuResourceAccess::Query),
                    );
                }
                Ok(accesses)
            }
            Self::Render(operation) => Ok(operation.accesses().to_vec()),
            Self::Copy(operation) => copy_derived_accesses(operation),
            Self::Clear(operation) => clear_derived_accesses(operation),
            Self::Resolve(operation) => Ok(vec![
                GpuResourceAccess::Query(operation.source_access().clone()),
                GpuResourceAccess::Buffer(operation.destination_access().clone()),
            ]),
            Self::Present(operation) => Ok(vec![GpuResourceAccess::Texture(
                operation.source_access().clone(),
            )]),
            Self::Upload(operation) => Ok(vec![operation.destination_access().clone()]),
            Self::Readback(operation) => Ok(vec![operation.source_access().clone()]),
        }
    }

    pub(crate) fn validate_shape(&self) -> Result<(), GpuWorkOperationError> {
        match self {
            Self::Compute(_)
            | Self::Render(_)
            | Self::Resolve(_)
            | Self::Present(_)
            | Self::Upload(_)
            | Self::Readback(_) => self.derived_accesses().map(|_| ()),
            Self::Copy(operation) => match operation {
                GpuCopyOperation::BufferToBuffer {
                    source,
                    destination,
                } => GpuCopyOperation::buffer_to_buffer(source.clone(), destination.clone())
                    .map(|_| ()),
                GpuCopyOperation::BufferToTexture {
                    source,
                    destination,
                } => GpuCopyOperation::buffer_to_texture(source.clone(), destination.clone())
                    .map(|_| ()),
                GpuCopyOperation::TextureToBuffer {
                    source,
                    destination,
                } => GpuCopyOperation::texture_to_buffer(source.clone(), destination.clone())
                    .map(|_| ()),
                GpuCopyOperation::TextureToTexture {
                    source,
                    destination,
                } => GpuCopyOperation::texture_to_texture(source.clone(), destination.clone())
                    .map(|_| ()),
            },
            Self::Clear(GpuClearOperation::BufferZero(region)) => {
                GpuClearOperation::buffer_zero(region.clone()).map(|_| ())
            }
        }
    }

    pub fn derived_requirements(
        &self,
    ) -> Result<GpuCapabilityRequirements, GpuCapabilityRequirementError> {
        let mut requirements = GpuCapabilityRequirements::new();
        let primary = match self {
            Self::Compute(_) => GpuCapabilityFeature::Compute,
            Self::Render(_) => GpuCapabilityFeature::RenderPipeline,
            Self::Copy(_) | Self::Clear(_) | Self::Upload(_) | Self::Readback(_) => {
                GpuCapabilityFeature::Copy
            }
            Self::Resolve(_) => GpuCapabilityFeature::TimestampQuery,
            Self::Present(_) => GpuCapabilityFeature::Presentation,
        };
        requirements.insert(GpuCapabilityRequirement::Required(primary))?;

        match self {
            Self::Compute(operation) => {
                requirements = requirements.merge(operation.pipeline().requirements())?;
                if operation.dispatch().is_indirect() {
                    requirements.insert(GpuCapabilityRequirement::Required(
                        GpuCapabilityFeature::IndirectExecution,
                    ))?;
                }
                if operation.timestamp_writes().is_some() {
                    requirements.insert(GpuCapabilityRequirement::Required(
                        GpuCapabilityFeature::TimestampQuery,
                    ))?;
                }
            }
            Self::Render(operation) => {
                for draw in operation.draws() {
                    requirements = requirements.merge(draw.requirements())?;
                    if matches!(draw.draw(), GpuDrawIntent::Indirect { .. }) {
                        requirements.insert(GpuCapabilityRequirement::Required(
                            GpuCapabilityFeature::IndirectExecution,
                        ))?;
                    }
                }
                if operation.depth_stencil_attachment().is_some() {
                    requirements.insert(GpuCapabilityRequirement::Required(
                        GpuCapabilityFeature::DepthAttachment,
                    ))?;
                }
                if operation.timestamp_writes().is_some() {
                    requirements.insert(GpuCapabilityRequirement::Required(
                        GpuCapabilityFeature::TimestampQuery,
                    ))?;
                }
            }
            Self::Copy(_)
            | Self::Clear(_)
            | Self::Resolve(_)
            | Self::Present(_)
            | Self::Upload(_)
            | Self::Readback(_) => {}
        }

        for access in self.derived_accesses().map_err(|_| {
            GpuCapabilityRequirementError::Invalid {
                operation: "derive GPU operation capability requirements",
                label: format!("{:?}", self.kind()),
                cause: super::GpuCapabilityRequirementCause::ConflictingStrength,
                correction: "retain checked operation accesses before deriving capability requirements",
            }
        })? {
            requirements = requirements.merge(&access.derived_requirements()?)?;
        }
        Ok(requirements)
    }
}

fn copy_derived_accesses(
    operation: &GpuCopyOperation,
) -> Result<Vec<GpuResourceAccess>, GpuWorkOperationError> {
    match operation {
        GpuCopyOperation::BufferToBuffer {
            source,
            destination,
        } => Ok(vec![
            GpuResourceAccess::Buffer(copy_buffer_access(
                source.buffer(),
                source.range(),
                GpuBufferAccessKind::CopySource,
                "derive GPU buffer copy source access",
            )?),
            GpuResourceAccess::Buffer(copy_buffer_access(
                destination.buffer(),
                destination.range(),
                GpuBufferAccessKind::CopyDestination,
                "derive GPU buffer copy destination access",
            )?),
        ]),
        GpuCopyOperation::BufferToTexture {
            source,
            destination,
        } => Ok(vec![
            GpuResourceAccess::Buffer(copy_buffer_layout_access(
                source,
                destination,
                GpuBufferAccessKind::CopySource,
            )?),
            GpuResourceAccess::Texture(copy_texture_access(
                destination,
                GpuTextureAccessKind::CopyDestination,
            )?),
        ]),
        GpuCopyOperation::TextureToBuffer {
            source,
            destination,
        } => Ok(vec![
            GpuResourceAccess::Texture(copy_texture_access(
                source,
                GpuTextureAccessKind::CopySource,
            )?),
            GpuResourceAccess::Buffer(copy_buffer_layout_access(
                destination,
                source,
                GpuBufferAccessKind::CopyDestination,
            )?),
        ]),
        GpuCopyOperation::TextureToTexture {
            source,
            destination,
        } => Ok(vec![
            GpuResourceAccess::Texture(copy_texture_access(
                source,
                GpuTextureAccessKind::CopySource,
            )?),
            GpuResourceAccess::Texture(copy_texture_access(
                destination,
                GpuTextureAccessKind::CopyDestination,
            )?),
        ]),
    }
}

fn clear_derived_accesses(
    operation: &GpuClearOperation,
) -> Result<Vec<GpuResourceAccess>, GpuWorkOperationError> {
    match operation {
        GpuClearOperation::BufferZero(region) => Ok(vec![GpuResourceAccess::Buffer(
            GpuBufferAccess::new(
                region.buffer(),
                region.range(),
                GpuBufferAccessKind::CopyDestination,
            )
            .map_err(|source| {
                GpuWorkOperationError::from_access(
                    "derive GPU buffer-zero access",
                    region.buffer().descriptor().common().label().as_str(),
                    GpuWorkOperationCause::OperationAccessContradiction,
                    "construct buffer-zero work through the checked constructor",
                    source,
                )
            })?,
        )]),
    }
}

fn copy_buffer_access(
    buffer: &super::GpuBufferHandle,
    range: GpuBufferRange,
    kind: GpuBufferAccessKind,
    operation: &'static str,
) -> Result<GpuBufferAccess, GpuWorkOperationError> {
    GpuBufferAccess::new(buffer, range, kind).map_err(|source| {
        GpuWorkOperationError::from_access(
            operation,
            buffer.descriptor().common().label().as_str(),
            GpuWorkOperationCause::InvalidCopyRegion,
            "declare matching copy usage and checked coverage",
            source,
        )
    })
}

fn copy_texture_access(
    region: &GpuTextureCopyRegion,
    kind: GpuTextureAccessKind,
) -> Result<GpuTextureAccess, GpuWorkOperationError> {
    GpuTextureAccess::new(
        GpuTextureAccessResource::Texture(region.texture().clone()),
        region.subresources(),
        kind,
    )
    .map_err(|source| {
        GpuWorkOperationError::from_access(
            "derive GPU texture copy access",
            region.texture().descriptor().common().label().as_str(),
            GpuWorkOperationCause::InvalidCopyRegion,
            "declare matching texture copy usage and checked coverage",
            source,
        )
    })
}

fn copy_buffer_layout_access(
    layout: &GpuBufferTextureLayout,
    texture: &GpuTextureCopyRegion,
    kind: GpuBufferAccessKind,
) -> Result<GpuBufferAccess, GpuWorkOperationError> {
    let extent = texture.extent();
    let logical_row = extent
        .width()
        .checked_mul(texture.texture().descriptor().format().bytes_per_texel())
        .ok_or_else(|| copy_layout_error(layout, "reduce the copy width"))?;
    if layout.bytes_per_row() < logical_row
        || (extent.depth_or_layers() > 1 && layout.rows_per_image() < extent.height())
        || (extent.depth_or_layers() == 1
            && layout.rows_per_image() != 0
            && layout.rows_per_image() < extent.height())
    {
        return Err(copy_layout_error(
            layout,
            "provide bytes-per-row and rows-per-image covering the complete logical copy",
        ));
    }
    let image_rows = if extent.depth_or_layers() > 1 {
        layout.rows_per_image()
    } else {
        0
    };
    let image_stride = u64::from(layout.bytes_per_row())
        .checked_mul(u64::from(image_rows))
        .ok_or_else(|| copy_layout_error(layout, "reduce the logical image stride"))?;
    let preceding_images = u64::from(extent.depth_or_layers() - 1)
        .checked_mul(image_stride)
        .ok_or_else(|| copy_layout_error(layout, "reduce the copy depth or layer count"))?;
    let preceding_rows = u64::from(extent.height() - 1)
        .checked_mul(u64::from(layout.bytes_per_row()))
        .ok_or_else(|| copy_layout_error(layout, "reduce the copy height"))?;
    let size = preceding_images
        .checked_add(preceding_rows)
        .and_then(|value| value.checked_add(u64::from(logical_row)))
        .ok_or_else(|| copy_layout_error(layout, "reduce the logical copy byte coverage"))?;
    let range =
        GpuBufferRange::new(layout.buffer(), layout.byte_offset(), size).map_err(|source| {
            GpuWorkOperationError::from_access(
                "derive GPU buffer-texture layout access",
                layout.buffer().descriptor().common().label().as_str(),
                GpuWorkOperationCause::InvalidCopyLayout,
                "keep the complete logical row and image coverage inside the buffer",
                source,
            )
        })?;
    copy_buffer_access(
        layout.buffer(),
        range,
        kind,
        "derive GPU buffer-texture copy access",
    )
}

fn copy_layout_error(
    layout: &GpuBufferTextureLayout,
    correction: &'static str,
) -> GpuWorkOperationError {
    GpuWorkOperationError::invalid(
        "derive GPU buffer-texture layout access",
        layout.buffer().descriptor().common().label().as_str(),
        Some(layout.buffer().diagnostic_identity()),
        GpuWorkOperationCause::InvalidCopyLayout,
        correction,
    )
}
