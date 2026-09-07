use super::super::WgpuContextState;
use super::super::surface::execution::WgpuSurfaceLeaseGuard;
use super::observability::PreparedExecutionObservability;
use super::surface_resources::{
    PreparedSurfaceUse, PreparedTexture, PreparedTextureView, prepare_texture_view,
};
use super::{
    PreparedBindGroup, PreparedTimestampWrites, checked_dynamic_offsets,
    preparation_pipeline_failure, preparation_program_binding_failure, realized_buffer,
    realized_query_set, submission_pipeline_failure, submission_program_binding_failure,
};
use crate::{
    GpuAttachmentStore, GpuColorAttachmentLoad, GpuContext, GpuDepthAttachmentLoad,
    GpuDepthStencilAccess, GpuDrawIntent, GpuIndexFormat, GpuRealizedBuffer,
    GpuRealizedRenderPipeline, GpuRealizedTextureView, GpuRenderDraw, GpuRenderOperation,
    GpuSubmissionFailure, GpuSubmissionPreparationError, GpuSubmissionPreparationErrorKind,
    GpuWorkResourceId,
};
use std::collections::BTreeMap;
use wgpu::{
    Color, CommandEncoder, IndexFormat, LoadOp, Operations, RenderPassColorAttachment,
    RenderPassDepthStencilAttachment, RenderPassDescriptor, RenderPassTimestampWrites, StoreOp,
};

#[derive(Debug, Clone)]
pub(super) struct PreparedRenderOperation {
    color_attachments: Vec<PreparedRenderColorAttachment>,
    depth_stencil_attachment: Option<PreparedRenderDepthStencilAttachment>,
    draws: Vec<PreparedRenderDraw>,
    timestamp_writes: Option<PreparedTimestampWrites>,
}

impl PreparedRenderOperation {
    pub(super) fn append_surface_uses(&self, uses: &mut Vec<PreparedSurfaceUse>) {
        for attachment in &self.color_attachments {
            if let Some(surface) = attachment.source.surface_use() {
                uses.push(surface.clone());
            }
            if let Some(surface) = attachment
                .resolve_target
                .as_ref()
                .and_then(PreparedTextureView::surface_use)
            {
                uses.push(surface.clone());
            }
        }
    }
}

#[derive(Debug, Clone)]
struct PreparedRenderColorAttachment {
    source: PreparedTextureView,
    load: GpuColorAttachmentLoad,
    store: GpuAttachmentStore,
    resolve_target: Option<PreparedTextureView>,
}

#[derive(Debug, Clone)]
struct PreparedRenderDepthStencilAttachment {
    source: GpuRealizedTextureView,
    access: GpuDepthStencilAccess,
    load: GpuDepthAttachmentLoad,
    store: GpuAttachmentStore,
}

#[derive(Debug, Clone)]
struct PreparedRenderVertexBuffer {
    slot: u32,
    buffer: GpuRealizedBuffer,
    offset: u64,
    end: u64,
}

#[derive(Debug, Clone)]
struct PreparedRenderIndexBuffer {
    buffer: GpuRealizedBuffer,
    offset: u64,
    end: u64,
    format: GpuIndexFormat,
}

#[derive(Debug, Clone)]
enum PreparedRenderDrawIntent {
    Direct {
        vertices: std::ops::Range<u32>,
        instances: std::ops::Range<u32>,
    },
    Indexed {
        indices: std::ops::Range<u32>,
        base_vertex: i32,
        instances: std::ops::Range<u32>,
    },
    Indirect {
        arguments: GpuRealizedBuffer,
        offset: u64,
        indexed: bool,
    },
}

#[derive(Debug, Clone)]
struct PreparedRenderDraw {
    pipeline: GpuRealizedRenderPipeline,
    bind_groups: Vec<PreparedBindGroup>,
    vertex_buffers: Vec<PreparedRenderVertexBuffer>,
    index_buffer: Option<PreparedRenderIndexBuffer>,
    draw: PreparedRenderDrawIntent,
    viewport: [f32; 6],
    scissor: [u32; 4],
    blend_constant: [f64; 4],
    stencil_reference: u32,
}

pub(super) async fn prepare_render_operation(
    context: &GpuContext,
    texture_cache: &mut BTreeMap<GpuWorkResourceId, PreparedTexture>,
    texture_view_cache: &mut BTreeMap<GpuWorkResourceId, PreparedTextureView>,
    render: &GpuRenderOperation,
) -> Result<PreparedRenderOperation, GpuSubmissionPreparationError> {
    let mut buffer_cache = BTreeMap::<GpuWorkResourceId, GpuRealizedBuffer>::new();
    let mut query_set_cache = BTreeMap::new();

    let mut color_attachments = Vec::with_capacity(render.color_attachments().len());
    for attachment in render.color_attachments() {
        let source = prepare_texture_view(
            context,
            texture_cache,
            texture_view_cache,
            attachment.source(),
        )?;
        let resolve_target = attachment
            .resolve_target()
            .map(|resolve| {
                prepare_texture_view(
                    context,
                    texture_cache,
                    texture_view_cache,
                    resolve.destination(),
                )
            })
            .transpose()?;
        color_attachments.push(PreparedRenderColorAttachment {
            source,
            load: attachment.load(),
            store: attachment.store(),
            resolve_target,
        });
    }

    let depth_stencil_attachment = render
        .depth_stencil_attachment()
        .map(|attachment| {
            let prepared = prepare_texture_view(
                context,
                texture_cache,
                texture_view_cache,
                attachment.source(),
            )?;
            let PreparedTextureView::Realized(source) = prepared else {
                return Err(GpuSubmissionPreparationError::new(
                    GpuSubmissionPreparationErrorKind::UnsupportedOperation,
                    "SurfaceAcquired textures are presentation color resources and cannot be used as depth/stencil attachments",
                ));
            };
            Ok(PreparedRenderDepthStencilAttachment {
                source,
                access: attachment.access(),
                load: attachment.load(),
                store: attachment.store(),
            })
        })
        .transpose()?;

    let timestamp_writes = render
        .timestamp_writes()
        .map(|writes| {
            Ok(PreparedTimestampWrites {
                query_set: realized_query_set(context, &mut query_set_cache, writes.query_set())?,
                beginning_of_pass: writes.beginning_of_pass(),
                end_of_pass: writes.end_of_pass(),
            })
        })
        .transpose()?;

    let mut draws = Vec::with_capacity(render.draws().len());
    for draw in render.draws() {
        draws.push(prepare_render_draw(context, &mut buffer_cache, draw).await?);
    }

    Ok(PreparedRenderOperation {
        color_attachments,
        depth_stencil_attachment,
        draws,
        timestamp_writes,
    })
}

async fn prepare_render_draw(
    context: &GpuContext,
    buffer_cache: &mut BTreeMap<GpuWorkResourceId, GpuRealizedBuffer>,
    draw: &GpuRenderDraw,
) -> Result<PreparedRenderDraw, GpuSubmissionPreparationError> {
    let descriptor = draw.pipeline();
    let program = context
        .realize_program(descriptor.program())
        .await
        .map_err(preparation_program_binding_failure)?;
    let pipeline_layout = context
        .realize_pipeline_layout(descriptor.layout())
        .await
        .map_err(preparation_program_binding_failure)?;
    let pipeline = context
        .realize_render_pipeline(descriptor, &program, &pipeline_layout)
        .await
        .map_err(preparation_pipeline_failure)?;

    let mut bind_groups = Vec::with_capacity(draw.bindings().groups().len());
    for group in draw.bindings().groups() {
        let layout = context
            .realize_bind_group_layout(group.layout())
            .await
            .map_err(preparation_program_binding_failure)?;
        let realization = context
            .realize_validated_bind_group(&layout, group.clone())
            .await
            .map_err(preparation_program_binding_failure)?;
        bind_groups.push(PreparedBindGroup {
            index: group.layout().group(),
            realization,
            dynamic_offsets: checked_dynamic_offsets(group)?,
        });
    }

    let mut vertex_buffers = Vec::with_capacity(draw.vertex_buffers().len());
    for binding in draw.vertex_buffers() {
        vertex_buffers.push(PreparedRenderVertexBuffer {
            slot: binding.slot(),
            buffer: realized_buffer(context, buffer_cache, binding.buffer())?,
            offset: binding.range().offset(),
            end: binding.range().end(),
        });
    }

    let index_buffer = draw
        .index_buffer()
        .map(|binding| {
            Ok(PreparedRenderIndexBuffer {
                buffer: realized_buffer(context, buffer_cache, binding.buffer())?,
                offset: binding.range().offset(),
                end: binding.range().end(),
                format: binding.format(),
            })
        })
        .transpose()?;

    let draw_intent = match draw.draw() {
        GpuDrawIntent::Direct {
            vertices,
            instances,
        } => PreparedRenderDrawIntent::Direct {
            vertices: vertices.first()..vertices.end(),
            instances: instances.first()..instances.end(),
        },
        GpuDrawIntent::Indexed {
            indices,
            base_vertex,
            instances,
        } => PreparedRenderDrawIntent::Indexed {
            indices: indices.first()..indices.end(),
            base_vertex: *base_vertex,
            instances: instances.first()..instances.end(),
        },
        GpuDrawIntent::Indirect {
            arguments,
            range,
            indexed,
        } => PreparedRenderDrawIntent::Indirect {
            arguments: realized_buffer(context, buffer_cache, arguments)?,
            offset: range.offset(),
            indexed: *indexed,
        },
    };

    let scissor = draw.scissor();
    Ok(PreparedRenderDraw {
        pipeline,
        bind_groups,
        vertex_buffers,
        index_buffer,
        draw: draw_intent,
        viewport: draw.viewport().values(),
        scissor: [scissor.x(), scissor.y(), scissor.width(), scissor.height()],
        blend_constant: draw.blend_constant().components(),
        stencil_reference: draw.stencil_reference(),
    })
}

pub(super) fn encode_render_operation<'a>(
    backend: &WgpuContextState,
    encoder: &mut CommandEncoder,
    observability: &PreparedExecutionObservability,
    render: &'a PreparedRenderOperation,
    surface_guard: Option<&'a WgpuSurfaceLeaseGuard<'_>>,
) -> Result<(), GpuSubmissionFailure> {
    let resolved_sources = render
        .color_attachments
        .iter()
        .map(|attachment| attachment.source.resolve(surface_guard))
        .collect::<Result<Vec<_>, _>>()?;
    let resolved_targets = render
        .color_attachments
        .iter()
        .map(|attachment| {
            attachment
                .resolve_target
                .as_ref()
                .map(|target| target.resolve(surface_guard))
                .transpose()
        })
        .collect::<Result<Vec<_>, _>>()?;
    let color_attachments = render
        .color_attachments
        .iter()
        .zip(&resolved_sources)
        .zip(&resolved_targets)
        .map(|((attachment, source), resolve_target)| {
            Some(RenderPassColorAttachment {
                view: source.as_ref(),
                depth_slice: None,
                resolve_target: resolve_target.as_ref().map(|target| target.as_ref()),
                ops: Operations {
                    load: color_load(attachment.load),
                    store: attachment_store(attachment.store),
                },
            })
        })
        .collect::<Vec<_>>();

    let depth_stencil_attachment = render.depth_stencil_attachment.as_ref().map(|attachment| {
        RenderPassDepthStencilAttachment {
            view: &attachment.source.record.object,
            depth_ops: match attachment.access {
                GpuDepthStencilAccess::ReadOnly => None,
                GpuDepthStencilAccess::ReadWrite => Some(Operations {
                    load: depth_load(attachment.load),
                    store: attachment_store(attachment.store),
                }),
            },
            stencil_ops: None,
        }
    });
    let timestamp_writes =
        render
            .timestamp_writes
            .as_ref()
            .map(|writes| RenderPassTimestampWrites {
                query_set: &writes.query_set.record.object,
                beginning_of_pass_write_index: writes.beginning_of_pass,
                end_of_pass_write_index: writes.end_of_pass,
            });
    let realized_pipelines = render
        .draws
        .iter()
        .map(|draw| &draw.pipeline)
        .collect::<Vec<_>>();
    let debug_label = observability.debug_label();

    backend
        .pipeline_realization
        .with_execution_render_pipelines(
            &realized_pipelines,
            &backend.program_binding_realization,
            |pipeline_objects| {
                let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: Some(debug_label.as_str()),
                    color_attachments: &color_attachments,
                    depth_stencil_attachment,
                    timestamp_writes,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });

                for (draw, pipeline_object) in render.draws.iter().zip(pipeline_objects) {
                    pass.set_pipeline(pipeline_object);
                    let realized_groups = draw
                        .bind_groups
                        .iter()
                        .map(|group| &group.realization)
                        .collect::<Vec<_>>();
                    backend
                        .program_binding_realization
                        .with_execution_bind_groups(&realized_groups, |group_objects| {
                            for (prepared, object) in
                                draw.bind_groups.iter().zip(group_objects.iter())
                            {
                                pass.set_bind_group(
                                    prepared.index,
                                    *object,
                                    &prepared.dynamic_offsets,
                                );
                            }
                        })
                        .map_err(submission_program_binding_failure)?;

                    for vertex in &draw.vertex_buffers {
                        pass.set_vertex_buffer(
                            vertex.slot,
                            vertex.buffer.record.object.slice(vertex.offset..vertex.end),
                        );
                    }
                    if let Some(index) = &draw.index_buffer {
                        pass.set_index_buffer(
                            index.buffer.record.object.slice(index.offset..index.end),
                            match index.format {
                                GpuIndexFormat::Uint16 => IndexFormat::Uint16,
                                GpuIndexFormat::Uint32 => IndexFormat::Uint32,
                            },
                        );
                    }

                    let [x, y, width, height, min_depth, max_depth] = draw.viewport;
                    pass.set_viewport(x, y, width, height, min_depth, max_depth);
                    let [x, y, width, height] = draw.scissor;
                    pass.set_scissor_rect(x, y, width, height);
                    let [r, g, b, a] = draw.blend_constant;
                    pass.set_blend_constant(Color { r, g, b, a });
                    pass.set_stencil_reference(draw.stencil_reference);

                    match &draw.draw {
                        PreparedRenderDrawIntent::Direct {
                            vertices,
                            instances,
                        } => pass.draw(vertices.clone(), instances.clone()),
                        PreparedRenderDrawIntent::Indexed {
                            indices,
                            base_vertex,
                            instances,
                        } => pass.draw_indexed(indices.clone(), *base_vertex, instances.clone()),
                        PreparedRenderDrawIntent::Indirect {
                            arguments,
                            offset,
                            indexed,
                        } => {
                            if *indexed {
                                pass.draw_indexed_indirect(&arguments.record.object, *offset);
                            } else {
                                pass.draw_indirect(&arguments.record.object, *offset);
                            }
                        }
                    }
                }
                Ok(())
            },
        )
        .map_err(submission_pipeline_failure)?
}

fn color_load(load: GpuColorAttachmentLoad) -> LoadOp<Color> {
    match load {
        GpuColorAttachmentLoad::Load => LoadOp::Load,
        GpuColorAttachmentLoad::Clear(value) => {
            let [r, g, b, a] = value.components();
            LoadOp::Clear(Color { r, g, b, a })
        }
    }
}

fn depth_load(load: GpuDepthAttachmentLoad) -> LoadOp<f32> {
    match load {
        GpuDepthAttachmentLoad::Load => LoadOp::Load,
        GpuDepthAttachmentLoad::Clear(value) => LoadOp::Clear(value.value()),
    }
}

const fn attachment_store(store: GpuAttachmentStore) -> StoreOp {
    match store {
        GpuAttachmentStore::Store => StoreOp::Store,
        GpuAttachmentStore::Discard => StoreOp::Discard,
    }
}
