use super::super::super::contract_diagnostics::{GpuProgramContractCause, GpuProgramContractError};
use super::primitive::{GpuMultisampleStateDescriptor, GpuPrimitiveStateDescriptor};
use super::target::{
    GpuColorTargetStateDescriptor, GpuDepthStencilStateDescriptor, GpuFragmentOutputStateDescriptor,
};
use super::vertex::GpuVertexInputStateDescriptor;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct GpuRenderPipelineStateDescriptor {
    vertex_input: GpuVertexInputStateDescriptor,
    fragment_output: Option<GpuFragmentOutputStateDescriptor>,
    primitive: GpuPrimitiveStateDescriptor,
    depth_stencil: Option<GpuDepthStencilStateDescriptor>,
    multisample: GpuMultisampleStateDescriptor,
}

impl GpuRenderPipelineStateDescriptor {
    pub fn new(
        vertex_input: GpuVertexInputStateDescriptor,
        fragment_output: Option<GpuFragmentOutputStateDescriptor>,
        primitive: GpuPrimitiveStateDescriptor,
        depth_stencil: Option<GpuDepthStencilStateDescriptor>,
        multisample: GpuMultisampleStateDescriptor,
    ) -> Result<Self, GpuProgramContractError> {
        let first_color_target = fragment_output
            .as_ref()
            .and_then(|output| output.color_targets().next());

        if first_color_target.is_none() && depth_stencil.is_none() {
            return Err(invalid_render_pipeline_state(
                "color_targets=0, depth_stencil=none",
                "declare at least one color target or a depth-stencil attachment",
            ));
        }

        if multisample.alpha_to_coverage_enabled()
            && !first_color_target
                .is_some_and(GpuColorTargetStateDescriptor::has_blendable_alpha_channel)
        {
            return Err(invalid_render_pipeline_state(
                format!(
                    "alpha_to_coverage_enabled=true, first_color_target={:?}",
                    first_color_target.map(GpuColorTargetStateDescriptor::format)
                ),
                "provide a fragment stage whose first color target is blendable and has an alpha channel",
            ));
        }

        Ok(Self {
            vertex_input,
            fragment_output,
            primitive,
            depth_stencil,
            multisample,
        })
    }

    pub fn vertex_input(&self) -> &GpuVertexInputStateDescriptor {
        &self.vertex_input
    }

    pub fn fragment_output(&self) -> Option<&GpuFragmentOutputStateDescriptor> {
        self.fragment_output.as_ref()
    }

    pub const fn primitive(&self) -> GpuPrimitiveStateDescriptor {
        self.primitive
    }

    pub const fn depth_stencil(&self) -> Option<GpuDepthStencilStateDescriptor> {
        self.depth_stencil
    }

    pub const fn multisample(&self) -> GpuMultisampleStateDescriptor {
        self.multisample
    }

    pub const fn has_fragment_stage(&self) -> bool {
        self.fragment_output.is_some()
    }

    pub fn has_color_targets(&self) -> bool {
        self.fragment_output
            .as_ref()
            .is_some_and(|output| output.color_targets().next().is_some())
    }
}

fn invalid_render_pipeline_state(
    label: impl Into<String>,
    correction: &'static str,
) -> GpuProgramContractError {
    GpuProgramContractError::invalid(
        "construct GPU render-pipeline state",
        label,
        GpuProgramContractCause::RenderPipelineStateInvalid,
        correction,
    )
}
