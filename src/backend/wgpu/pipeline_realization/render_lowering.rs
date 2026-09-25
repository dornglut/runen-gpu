use super::super::texture_format_mapping::texture_format;
use super::render_mapping;
use crate::{
    GpuBlendComponent, GpuBlendState, GpuColorTargetStateDescriptor, GpuContext,
    GpuPipelineRealizationError, GpuPipelineRealizationErrorCategory, GpuPrimitiveStateDescriptor,
    GpuRenderPipelineDescriptor, GpuTextureFormat,
};
use wgpu::{
    BlendState, ColorTargetState, DepthBiasState, DepthStencilState, DownlevelFlags, Features,
    MultisampleState, PolygonMode, PrimitiveState, StencilFaceState, StencilState, TextureFormat,
    TextureFormatFeatureFlags, TextureFormatFeatures, TextureUsages, VertexAttribute,
    VertexBufferLayout, VertexStepMode,
};

use super::super::WgpuContextState;

pub(super) struct LoweredRenderPipeline {
    vertex_buffers: Vec<LoweredVertexBuffer>,
    pub(super) color_targets: Vec<Option<ColorTargetState>>,
    pub(super) primitive: PrimitiveState,
    pub(super) depth_stencil: Option<DepthStencilState>,
    pub(super) multisample: MultisampleState,
}

struct LoweredVertexBuffer {
    array_stride: u64,
    step_mode: VertexStepMode,
    attributes: Vec<VertexAttribute>,
}

impl LoweredRenderPipeline {
    pub(super) fn vertex_buffer_layouts(&self) -> Vec<Option<VertexBufferLayout<'_>>> {
        self.vertex_buffers
            .iter()
            .map(|buffer| {
                Some(VertexBufferLayout {
                    array_stride: buffer.array_stride,
                    step_mode: buffer.step_mode,
                    attributes: buffer.attributes.as_slice(),
                })
            })
            .collect()
    }
}

pub(super) fn lower_render_pipeline(
    context: &GpuContext,
    descriptor: &GpuRenderPipelineDescriptor,
    request: &str,
) -> Result<LoweredRenderPipeline, GpuPipelineRealizationError> {
    let render_state = descriptor.state();
    let vertex_buffers = lower_vertex_buffers(context, descriptor, request)?;
    let sample_count = render_state.multisample().sample_count();
    let color_targets = render_state
        .fragment_output()
        .map(|output| {
            output
                .color_targets()
                .map(|target| {
                    let features = validate_attachment_support(
                        context,
                        target.format(),
                        sample_count,
                        request,
                    )?;
                    validate_color_blend_support(
                        target.blend(),
                        features
                            .flags
                            .contains(TextureFormatFeatureFlags::BLENDABLE),
                        request,
                    )?;
                    Ok(Some(lower_color_target(target)))
                })
                .collect::<Result<Vec<_>, GpuPipelineRealizationError>>()
        })
        .transpose()?
        .unwrap_or_default();

    validate_color_target_count(context, color_targets.len(), request)?;

    let depth_stencil = render_state
        .depth_stencil()
        .map(|depth| {
            validate_attachment_support(context, depth.format(), sample_count, request)?;
            Ok(lower_depth_stencil(depth))
        })
        .transpose()?;

    let multisample = render_state.multisample();
    Ok(LoweredRenderPipeline {
        vertex_buffers,
        color_targets,
        primitive: lower_primitive(render_state.primitive()),
        depth_stencil,
        multisample: MultisampleState {
            count: multisample.sample_count(),
            mask: multisample.sample_mask(),
            alpha_to_coverage_enabled: multisample.alpha_to_coverage_enabled(),
        },
    })
}

fn lower_vertex_buffers(
    context: &GpuContext,
    descriptor: &GpuRenderPipelineDescriptor,
    request: &str,
) -> Result<Vec<LoweredVertexBuffer>, GpuPipelineRealizationError> {
    let layouts = descriptor
        .state()
        .vertex_input()
        .layouts()
        .collect::<Vec<_>>();
    let positional_count_u32 = layouts
        .last()
        .map(|layout| {
            layout.slot().checked_add(1).ok_or_else(|| {
                incompatible(
                    request,
                    "the highest vertex-buffer slot exceeds the normalized u32 slot domain",
                )
            })
        })
        .transpose()?
        .unwrap_or(0);
    let positional_count = positional_count_u32 as usize;

    let device_limits = context.device_facts().device_limits().values();
    let workload_limits = context.device_facts().workload_budget().limits();

    // The accepted G4B sparse-slot contract remains positional. Empty slots continue to be
    // materialized as present, empty native layouts rather than adopting WGPU 30's new gap form.
    if positional_count_u32 > device_limits.max_vertex_buffers()
        || positional_count_u32 > workload_limits.max_vertex_buffers()
    {
        return Err(incompatible(
            request,
            "the highest vertex-buffer slot exceeds an admitted or created-device limit",
        ));
    }

    let total_attributes = layouts
        .iter()
        .map(|layout| layout.attributes().len())
        .sum::<usize>();
    let total_attributes = u32::try_from(total_attributes).map_err(|_| {
        incompatible(
            request,
            "the vertex attribute count exceeds the normalized u32 limit domain",
        )
    })?;
    if total_attributes > device_limits.max_vertex_attributes()
        || total_attributes > workload_limits.max_vertex_attributes()
    {
        return Err(incompatible(
            request,
            "the vertex attribute count exceeds an admitted or created-device limit",
        ));
    }

    let mut lowered = (0..positional_count)
        .map(|_| LoweredVertexBuffer {
            array_stride: 0,
            step_mode: VertexStepMode::Vertex,
            attributes: Vec::new(),
        })
        .collect::<Vec<_>>();

    for layout in layouts {
        if layout.array_stride() > u64::from(device_limits.max_vertex_buffer_array_stride())
            || layout.array_stride() > u64::from(workload_limits.max_vertex_buffer_array_stride())
        {
            return Err(incompatible(
                request,
                "a vertex-buffer stride exceeds an admitted or created-device limit",
            ));
        }
        let slot = layout.slot() as usize;
        lowered[slot] = LoweredVertexBuffer {
            array_stride: layout.array_stride(),
            step_mode: render_mapping::vertex_step_mode(layout.step_mode()),
            attributes: layout
                .attributes()
                .map(|attribute| VertexAttribute {
                    format: render_mapping::vertex_format(attribute.format()),
                    offset: attribute.offset(),
                    shader_location: attribute.shader_location(),
                })
                .collect(),
        };
    }
    Ok(lowered)
}

fn validate_color_target_count(
    context: &GpuContext,
    color_target_count: usize,
    request: &str,
) -> Result<(), GpuPipelineRealizationError> {
    let count = u32::try_from(color_target_count).map_err(|_| {
        incompatible(
            request,
            "the color-target count exceeds the normalized u32 limit domain",
        )
    })?;
    let device_limit = context
        .device_facts()
        .device_limits()
        .values()
        .max_color_attachments();
    let workload_limit = context
        .device_facts()
        .workload_budget()
        .limits()
        .max_color_attachments();
    if count > device_limit || count > workload_limit {
        return Err(incompatible(
            request,
            "the color-target count exceeds an admitted or created-device limit",
        ));
    }
    Ok(())
}

fn validate_attachment_support(
    context: &GpuContext,
    format: GpuTextureFormat,
    sample_count: u32,
    request: &str,
) -> Result<TextureFormatFeatures, GpuPipelineRealizationError> {
    let native_format = texture_format(format);
    let features = device_format_features(&context.backend, native_format);
    if !features
        .allowed_usages
        .contains(TextureUsages::RENDER_ATTACHMENT)
    {
        return Err(incompatible(
            request,
            "the created device cannot use the selected format as a render attachment",
        ));
    }
    if !features.flags.sample_count_supported(sample_count) {
        return Err(incompatible(
            request,
            "the selected attachment format does not support the requested sample count",
        ));
    }
    Ok(features)
}

fn validate_color_blend_support(
    blend: Option<GpuBlendState>,
    blendable: bool,
    request: &str,
) -> Result<(), GpuPipelineRealizationError> {
    if blend.is_some() && !blendable {
        return Err(incompatible(
            request,
            "the selected color attachment format does not support blending required by the pipeline state",
        ));
    }
    Ok(())
}

fn lower_color_target(target: GpuColorTargetStateDescriptor) -> ColorTargetState {
    ColorTargetState {
        format: texture_format(target.format()),
        blend: target.blend().map(lower_blend_state),
        write_mask: render_mapping::color_write_mask(target.write_mask()),
    }
}

fn lower_blend_state(state: GpuBlendState) -> BlendState {
    BlendState {
        color: lower_blend_component(state.color()),
        alpha: lower_blend_component(state.alpha()),
    }
}

fn lower_blend_component(component: GpuBlendComponent) -> wgpu::BlendComponent {
    wgpu::BlendComponent {
        src_factor: render_mapping::blend_factor(component.src_factor()),
        dst_factor: render_mapping::blend_factor(component.dst_factor()),
        operation: render_mapping::blend_operation(component.operation()),
    }
}

fn lower_depth_stencil(state: crate::GpuDepthStencilStateDescriptor) -> DepthStencilState {
    let depth = state.depth();
    let stencil = state.stencil();
    DepthStencilState {
        format: texture_format(state.format()),
        depth_write_enabled: depth.map(crate::GpuDepthStateDescriptor::write_enabled),
        depth_compare: depth
            .map(crate::GpuDepthStateDescriptor::compare)
            .map(render_mapping::compare_function),
        stencil: stencil.map(lower_stencil_state).unwrap_or_default(),
        bias: DepthBiasState::default(),
    }
}

fn lower_stencil_state(state: crate::GpuStencilStateDescriptor) -> StencilState {
    StencilState {
        front: lower_stencil_face(state.front()),
        back: lower_stencil_face(state.back()),
        read_mask: state.read_mask(),
        write_mask: state.write_mask(),
    }
}

fn lower_stencil_face(face: crate::GpuStencilFaceStateDescriptor) -> StencilFaceState {
    StencilFaceState {
        compare: render_mapping::compare_function(face.compare()),
        fail_op: render_mapping::stencil_operation(face.fail_op()),
        depth_fail_op: render_mapping::stencil_operation(face.depth_fail_op()),
        pass_op: render_mapping::stencil_operation(face.pass_op()),
    }
}

fn lower_primitive(primitive: GpuPrimitiveStateDescriptor) -> PrimitiveState {
    PrimitiveState {
        topology: render_mapping::primitive_topology(primitive.topology()),
        strip_index_format: primitive
            .strip_index_format()
            .map(render_mapping::index_format),
        front_face: render_mapping::front_face(primitive.front_face()),
        cull_mode: render_mapping::cull_mode(primitive.cull_mode()),
        unclipped_depth: false,
        polygon_mode: PolygonMode::Fill,
        conservative: false,
    }
}

fn device_format_features(
    state: &WgpuContextState,
    format: TextureFormat,
) -> TextureFormatFeatures {
    let device_features = state.device.features();
    let downlevel = state.adapter.get_downlevel_capabilities();
    if device_features.contains(Features::TEXTURE_ADAPTER_SPECIFIC_FORMAT_FEATURES)
        || !downlevel
            .flags
            .contains(DownlevelFlags::WEBGPU_TEXTURE_FORMAT_SUPPORT)
    {
        state.adapter.get_texture_format_features(format)
    } else {
        format.guaranteed_format_features(device_features)
    }
}

fn incompatible(request: &str, detail: &'static str) -> GpuPipelineRealizationError {
    GpuPipelineRealizationError::new(
        GpuPipelineRealizationErrorCategory::FormatOrAlignmentNotAdmitted,
        request,
        detail,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alpha_blending_requires_a_blendable_color_format() {
        let component = GpuBlendComponent::new(
            crate::GpuBlendFactor::One,
            crate::GpuBlendFactor::Zero,
            crate::GpuBlendOperation::Add,
        )
        .unwrap();
        let blend = GpuBlendState::new(component, component);
        assert!(validate_color_blend_support(None, false, "test").is_ok());
        assert!(validate_color_blend_support(Some(blend), true, "test").is_ok());

        let error = validate_color_blend_support(Some(blend), false, "test")
            .expect_err("alpha blending must reject a non-blendable color format");
        assert_eq!(
            error.category(),
            GpuPipelineRealizationErrorCategory::FormatOrAlignmentNotAdmitted
        );
        assert!(
            error
                .detail()
                .is_some_and(|detail| detail.contains("does not support blending"))
        );
    }
}
