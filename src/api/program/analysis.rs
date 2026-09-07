use super::contract_diagnostics::{GpuProgramContractCause, GpuProgramContractError};
use super::entry_point::{GpuEntryPointDescriptor, GpuEntryPointName};
use super::interface::{
    GpuBindingDeclaration, GpuBindingKey, GpuBindingKind, GpuBindingLayoutRefinement,
    GpuBindingProvenance, GpuProgramInterfaceDescriptor, GpuSamplerClass, GpuShaderStage,
    GpuShaderStages, GpuStorageBufferAccess, GpuStorageTextureAccess, GpuTextureSampleClass,
    GpuTextureViewDimension,
};
use super::source::GpuAdmittedProgramSource;
use super::stage_io::{
    GpuFragmentOutputBuiltin, GpuObservedFragmentOutputSignature, GpuObservedVertexInputSignature,
    GpuShaderIoLocation, GpuShaderIoScalarClass, GpuShaderIoValueType, GpuVertexInputBuiltin,
};
use crate::GpuTextureFormat;
use core::num::{NonZeroU32, NonZeroU64};
use naga::{
    AddressSpace, ArraySize, Binding, BuiltIn, ImageClass, ImageDimension, ScalarKind, ShaderStage,
    StorageAccess, StorageFormat, TypeInner, VectorSize,
};

pub(crate) struct ProgramAnalysis {
    pub(crate) interface: GpuProgramInterfaceDescriptor,
    pub(crate) entry_points: Vec<GpuEntryPointDescriptor>,
    pub(crate) vertex_inputs: Vec<GpuObservedVertexInputSignature>,
    pub(crate) fragment_outputs: Vec<GpuObservedFragmentOutputSignature>,
}

#[derive(Debug, Clone, Copy)]
enum CompilerTextureSampleClass {
    Float,
    Depth,
    Sint,
    Uint,
}

#[derive(Debug, Clone, Copy)]
enum CompilerSamplerClass {
    NonComparison,
    Comparison,
}

#[derive(Debug, Clone, Copy)]
enum CompilerBindingKind {
    UniformBuffer {
        compiler_minimum_size: Option<NonZeroU64>,
    },
    StorageBuffer {
        access: GpuStorageBufferAccess,
        compiler_minimum_size: Option<NonZeroU64>,
    },
    SampledTexture {
        sample_class: CompilerTextureSampleClass,
        view_dimension: GpuTextureViewDimension,
        multisampled: bool,
    },
    StorageTexture {
        access: GpuStorageTextureAccess,
        format: GpuTextureFormat,
        view_dimension: GpuTextureViewDimension,
    },
    Sampler {
        class: CompilerSamplerClass,
    },
}

pub(crate) fn analyze_program(
    source: &GpuAdmittedProgramSource,
    selected_entry_points: impl IntoIterator<Item = GpuEntryPointName>,
    refinements: impl IntoIterator<Item = GpuBindingLayoutRefinement>,
) -> Result<ProgramAnalysis, GpuProgramContractError> {
    let operation = "admit canonical WGSL program";
    let source_label = source.identity().diagnostic_label();
    let module = naga::front::wgsl::parse_str(source.canonical_wgsl()).map_err(|error| {
        invalid(
            operation,
            &source_label,
            GpuProgramContractCause::CanonicalWgslInvalid,
            format!("canonical WGSL parse failed: {error}"),
        )
    })?;
    // Permit only fixed binding-array forms that are normalized into explicit RunenGPU
    // capability requirements; context/device admission remains responsible for backend support.
    let analysis_capabilities = naga::valid::Capabilities::default()
        | naga::valid::Capabilities::TEXTURE_AND_SAMPLER_BINDING_ARRAY
        | naga::valid::Capabilities::BUFFER_BINDING_ARRAY
        | naga::valid::Capabilities::STORAGE_TEXTURE_BINDING_ARRAY
        | naga::valid::Capabilities::STORAGE_BUFFER_BINDING_ARRAY;
    let module_info =
        naga::valid::Validator::new(naga::valid::ValidationFlags::all(), analysis_capabilities)
            .validate(&module)
            .map_err(|error| {
                invalid(
                    operation,
                    &source_label,
                    GpuProgramContractCause::CanonicalWgslInvalid,
                    format!("canonical WGSL validation failed: {error}"),
                )
            })?;

    let mut selected_names = selected_entry_points.into_iter().collect::<Vec<_>>();
    if selected_names.is_empty() {
        return Err(GpuProgramContractError::invalid(
            operation,
            source_label,
            GpuProgramContractCause::EntryPointMissing,
            "select at least one canonical WGSL entry-point name",
        ));
    }
    selected_names.sort();
    if let Some(duplicate) = selected_names
        .windows(2)
        .find(|pair| pair[0] == pair[1])
        .map(|pair| pair[0].clone())
    {
        return Err(GpuProgramContractError::invalid(
            operation,
            duplicate.to_string(),
            GpuProgramContractCause::DuplicateEntryPoint,
            "select each entry-point name exactly once",
        ));
    }

    let mut selected_indices = Vec::with_capacity(selected_names.len());
    let mut entry_points = Vec::with_capacity(selected_names.len());
    for name in selected_names {
        let mut matches = module
            .entry_points
            .iter()
            .enumerate()
            .filter(|(_, entry)| entry.name == name.as_str());
        let Some((index, entry)) = matches.next() else {
            return Err(GpuProgramContractError::invalid(
                operation,
                name.to_string(),
                GpuProgramContractCause::EntryPointMissing,
                "select an entry point present in canonical WGSL",
            ));
        };
        if matches.next().is_some() {
            return Err(GpuProgramContractError::invalid(
                operation,
                name.to_string(),
                GpuProgramContractCause::DuplicateEntryPoint,
                "canonical WGSL must resolve a selected entry-point name unambiguously",
            ));
        }
        let stage = runen_stage(entry.stage);
        selected_indices.push((stage, index, name.clone()));
        entry_points.push(GpuEntryPointDescriptor::derived(name, stage));
    }
    entry_points.sort();

    let selected_stages =
        GpuShaderStages::new(entry_points.iter().map(GpuEntryPointDescriptor::stage))?;
    let mut refinements = refinements.into_iter().collect::<Vec<_>>();
    refinements.sort_by_key(GpuBindingLayoutRefinement::key);
    if let Some(duplicate) = refinements
        .windows(2)
        .find(|pair| pair[0].key() == pair[1].key())
        .map(|pair| pair[0].key())
    {
        return Err(GpuProgramContractError::invalid(
            operation,
            format!("binding {duplicate}"),
            GpuProgramContractCause::BindingRefinementInvalid,
            "provide at most one host/layout refinement for each compiler-derived binding",
        ));
    }
    let mut consumed_refinements = vec![false; refinements.len()];

    let mut bindings = Vec::new();
    for (global_handle, global) in module.global_variables.iter() {
        let Some(binding) = global.binding else {
            continue;
        };
        let used_stages = selected_indices
            .iter()
            .filter_map(|(stage, index, _)| {
                (!module_info.get_entry_point(*index)[global_handle].is_empty()).then_some(*stage)
            })
            .collect::<Vec<_>>();
        if used_stages.is_empty() {
            continue;
        }

        let key = GpuBindingKey::try_new(binding.group as u64, binding.binding as u64)?;
        let (base_type, array_count) =
            binding_array_type(&module, global.ty).map_err(|detail| {
                invalid(
                    operation,
                    &format!("binding {key}"),
                    GpuProgramContractCause::ProgramInterfaceMismatch,
                    detail,
                )
            })?;
        let compiler_kind = compiler_binding_kind(&module, &module_info, global.space, base_type)
            .map_err(|detail| {
            invalid(
                operation,
                &format!("binding {key}"),
                GpuProgramContractCause::ProgramInterfaceMismatch,
                detail,
            )
        })?;
        let observed_visibility = GpuShaderStages::new(used_stages)?;
        let refinement_index = refinements
            .binary_search_by_key(&key, GpuBindingLayoutRefinement::key)
            .ok();
        let refinement = refinement_index.map(|index| {
            consumed_refinements[index] = true;
            &refinements[index]
        });
        bindings.push(effective_binding(
            key,
            observed_visibility,
            selected_stages,
            compiler_kind,
            array_count,
            refinement,
        )?);
    }

    if let Some((refinement, _)) = refinements
        .iter()
        .zip(consumed_refinements.iter())
        .find(|(_, consumed)| !**consumed)
    {
        return Err(GpuProgramContractError::invalid(
            operation,
            format!("binding {}", refinement.key()),
            GpuProgramContractCause::BindingRefinementInvalid,
            "refine only a binding statically used by at least one selected entry point",
        ));
    }

    let interface = GpuProgramInterfaceDescriptor::new(bindings)?;
    let mut vertex_inputs = Vec::new();
    let mut fragment_outputs = Vec::new();
    for (stage, index, name) in selected_indices {
        let entry = &module.entry_points[index];
        match stage {
            GpuShaderStage::Vertex => vertex_inputs.push(normalize_vertex_input(
                &module,
                &entry.function,
                name,
                &source_label,
            )?),
            GpuShaderStage::Fragment => fragment_outputs.push(normalize_fragment_output(
                &module,
                &entry.function,
                name,
                &source_label,
            )?),
            GpuShaderStage::Compute => {}
        }
    }
    vertex_inputs.sort_by(|left, right| left.entry_point().cmp(right.entry_point()));
    fragment_outputs.sort_by(|left, right| left.entry_point().cmp(right.entry_point()));

    Ok(ProgramAnalysis {
        interface,
        entry_points,
        vertex_inputs,
        fragment_outputs,
    })
}

fn effective_binding(
    key: GpuBindingKey,
    observed_visibility: GpuShaderStages,
    selected_stages: GpuShaderStages,
    compiler_kind: CompilerBindingKind,
    array_count: Option<NonZeroU32>,
    refinement: Option<&GpuBindingLayoutRefinement>,
) -> Result<GpuBindingDeclaration, GpuProgramContractError> {
    let operation = "admit canonical WGSL program";
    let visibility = refinement
        .and_then(GpuBindingLayoutRefinement::visibility)
        .unwrap_or(observed_visibility);
    if observed_visibility
        .iter()
        .any(|stage| !visibility.contains(stage))
    {
        return Err(GpuProgramContractError::invalid(
            operation,
            format!("binding {key}"),
            GpuProgramContractCause::BindingRefinementInvalid,
            "visibility refinement must include every compiler-observed stage use",
        ));
    }
    if visibility
        .iter()
        .any(|stage| !selected_stages.contains(stage))
    {
        return Err(GpuProgramContractError::invalid(
            operation,
            format!("binding {key}"),
            GpuProgramContractCause::BindingRefinementInvalid,
            "visibility refinement may include only shader stages selected by this program",
        ));
    }

    let dynamic_offset = refinement.is_some_and(GpuBindingLayoutRefinement::dynamic_offset);
    let host_minimum_size = refinement.and_then(GpuBindingLayoutRefinement::host_minimum_size);
    let texture_sample_class =
        refinement.and_then(GpuBindingLayoutRefinement::texture_sample_class);
    let sampler_class = refinement.and_then(GpuBindingLayoutRefinement::sampler_class);

    let (kind, compiler_required_minimum_size) = match compiler_kind {
        CompilerBindingKind::UniformBuffer {
            compiler_minimum_size,
        } => {
            reject_non_buffer_refinement(key, texture_sample_class, sampler_class)?;
            (
                GpuBindingKind::uniform_buffer(dynamic_offset, host_minimum_size),
                compiler_minimum_size,
            )
        }
        CompilerBindingKind::StorageBuffer {
            access,
            compiler_minimum_size,
        } => {
            reject_non_buffer_refinement(key, texture_sample_class, sampler_class)?;
            (
                GpuBindingKind::storage_buffer(access, dynamic_offset, host_minimum_size),
                compiler_minimum_size,
            )
        }
        CompilerBindingKind::SampledTexture {
            sample_class,
            view_dimension,
            multisampled,
        } => {
            reject_buffer_refinement(key, dynamic_offset, host_minimum_size)?;
            if sampler_class.is_some() {
                return Err(invalid_refinement(
                    key,
                    "sampler policy cannot refine a sampled-texture binding",
                ));
            }
            let class = match sample_class {
                CompilerTextureSampleClass::Float => match texture_sample_class {
                    Some(GpuTextureSampleClass::FloatFilterable) => {
                        GpuTextureSampleClass::FloatFilterable
                    }
                    Some(GpuTextureSampleClass::FloatUnfilterable) => {
                        GpuTextureSampleClass::FloatUnfilterable
                    }
                    Some(_) => {
                        return Err(invalid_refinement(
                            key,
                            "float sampled textures require a filterable or unfilterable float layout choice",
                        ));
                    }
                    None => {
                        return Err(invalid_refinement(
                            key,
                            "WGSL float sampled textures require explicit filterable versus unfilterable host layout policy",
                        ));
                    }
                },
                CompilerTextureSampleClass::Depth => {
                    reject_texture_policy(key, texture_sample_class)?;
                    GpuTextureSampleClass::Depth
                }
                CompilerTextureSampleClass::Sint => {
                    reject_texture_policy(key, texture_sample_class)?;
                    GpuTextureSampleClass::Sint
                }
                CompilerTextureSampleClass::Uint => {
                    reject_texture_policy(key, texture_sample_class)?;
                    GpuTextureSampleClass::Uint
                }
            };
            (
                GpuBindingKind::sampled_texture(class, view_dimension, multisampled)?,
                None,
            )
        }
        CompilerBindingKind::StorageTexture {
            access,
            format,
            view_dimension,
        } => {
            reject_buffer_refinement(key, dynamic_offset, host_minimum_size)?;
            reject_non_buffer_refinement(key, texture_sample_class, sampler_class)?;
            (
                GpuBindingKind::storage_texture(access, format, view_dimension)?,
                None,
            )
        }
        CompilerBindingKind::Sampler { class } => {
            reject_buffer_refinement(key, dynamic_offset, host_minimum_size)?;
            if texture_sample_class.is_some() {
                return Err(invalid_refinement(
                    key,
                    "texture policy cannot refine a sampler binding",
                ));
            }
            let class = match class {
                CompilerSamplerClass::Comparison => {
                    if sampler_class.is_some() {
                        return Err(invalid_refinement(
                            key,
                            "comparison sampler semantics are compiler-known and cannot be overridden",
                        ));
                    }
                    GpuSamplerClass::Comparison
                }
                CompilerSamplerClass::NonComparison => match sampler_class {
                    Some(GpuSamplerClass::Filtering) => GpuSamplerClass::Filtering,
                    Some(GpuSamplerClass::NonFiltering) => GpuSamplerClass::NonFiltering,
                    Some(GpuSamplerClass::Comparison) => {
                        return Err(invalid_refinement(
                            key,
                            "a non-comparison WGSL sampler cannot be refined into a comparison sampler",
                        ));
                    }
                    None => {
                        return Err(invalid_refinement(
                            key,
                            "WGSL non-comparison samplers require explicit filtering versus non-filtering host layout policy",
                        ));
                    }
                },
            };
            (GpuBindingKind::sampler(class), None)
        }
    };

    if let (Some(host), Some(compiler)) = (host_minimum_size, compiler_required_minimum_size) {
        if host < compiler {
            return Err(invalid_refinement(
                key,
                "host minimum binding size cannot be weaker than the compiler-required shader minimum",
            ));
        }
    }

    GpuBindingDeclaration::from_program_analysis(
        key,
        visibility,
        kind,
        array_count,
        compiler_required_minimum_size,
        format!("binding {key}"),
        GpuBindingProvenance::new("canonical-wgsl-program-admission", None)?,
    )
}

fn reject_buffer_refinement(
    key: GpuBindingKey,
    dynamic_offset: bool,
    host_minimum_size: Option<NonZeroU64>,
) -> Result<(), GpuProgramContractError> {
    if dynamic_offset || host_minimum_size.is_some() {
        return Err(invalid_refinement(
            key,
            "dynamic-offset and host minimum-size policy apply only to buffer bindings",
        ));
    }
    Ok(())
}

fn reject_non_buffer_refinement(
    key: GpuBindingKey,
    texture_sample_class: Option<GpuTextureSampleClass>,
    sampler_class: Option<GpuSamplerClass>,
) -> Result<(), GpuProgramContractError> {
    if texture_sample_class.is_some() || sampler_class.is_some() {
        return Err(invalid_refinement(
            key,
            "texture/sampler layout policy does not apply to this compiler-derived binding class",
        ));
    }
    Ok(())
}

fn reject_texture_policy(
    key: GpuBindingKey,
    texture_sample_class: Option<GpuTextureSampleClass>,
) -> Result<(), GpuProgramContractError> {
    if texture_sample_class.is_some() {
        return Err(invalid_refinement(
            key,
            "this sampled-texture class is compiler-known and cannot be overridden",
        ));
    }
    Ok(())
}

fn invalid_refinement(key: GpuBindingKey, correction: &'static str) -> GpuProgramContractError {
    GpuProgramContractError::invalid(
        "admit canonical WGSL program",
        format!("binding {key}"),
        GpuProgramContractCause::BindingRefinementInvalid,
        correction,
    )
}

fn binding_array_type(
    module: &naga::Module,
    ty: naga::Handle<naga::Type>,
) -> Result<(naga::Handle<naga::Type>, Option<NonZeroU32>), &'static str> {
    match module.types[ty].inner {
        TypeInner::BindingArray { base, size } => match size {
            ArraySize::Constant(count) => Ok((base, Some(count))),
            ArraySize::Pending(_) | ArraySize::Dynamic => {
                Err("binding arrays require one fixed nonzero cardinality")
            }
        },
        _ => Ok((ty, None)),
    }
}

fn compiler_binding_kind(
    module: &naga::Module,
    module_info: &naga::valid::ModuleInfo,
    space: AddressSpace,
    ty: naga::Handle<naga::Type>,
) -> Result<CompilerBindingKind, &'static str> {
    match space {
        AddressSpace::Uniform => Ok(CompilerBindingKind::UniformBuffer {
            compiler_minimum_size: compiler_minimum_size(module, module_info, ty),
        }),
        AddressSpace::Storage { access } => Ok(CompilerBindingKind::StorageBuffer {
            access: if access.intersects(StorageAccess::STORE | StorageAccess::ATOMIC) {
                GpuStorageBufferAccess::ReadWrite
            } else {
                GpuStorageBufferAccess::ReadOnly
            },
            compiler_minimum_size: compiler_minimum_size(module, module_info, ty),
        }),
        AddressSpace::Handle => match module.types[ty].inner {
            TypeInner::Sampler { comparison } => Ok(CompilerBindingKind::Sampler {
                class: if comparison {
                    CompilerSamplerClass::Comparison
                } else {
                    CompilerSamplerClass::NonComparison
                },
            }),
            TypeInner::Image {
                dim,
                arrayed,
                class,
            } => match class {
                ImageClass::Sampled { kind, multi } => Ok(CompilerBindingKind::SampledTexture {
                    sample_class: compiler_sample_class(kind)?,
                    view_dimension: texture_view_dimension(dim, arrayed)?,
                    multisampled: multi,
                }),
                ImageClass::Depth { multi } => Ok(CompilerBindingKind::SampledTexture {
                    sample_class: CompilerTextureSampleClass::Depth,
                    view_dimension: texture_view_dimension(dim, arrayed)?,
                    multisampled: multi,
                }),
                ImageClass::Storage { format, access } => Ok(CompilerBindingKind::StorageTexture {
                    access: storage_texture_access(access)?,
                    format: storage_texture_format(format)?,
                    view_dimension: texture_view_dimension(dim, arrayed)?,
                }),
                ImageClass::External => {
                    Err("external textures are outside the admitted RunenGPU binding vocabulary")
                }
            },
            _ => Err("handle-space binding has an unsupported WGSL resource type"),
        },
        _ => Err("only uniform, storage, sampler, and texture globals are admitted resources"),
    }
}

fn compiler_minimum_size(
    module: &naga::Module,
    module_info: &naga::valid::ModuleInfo,
    ty: naga::Handle<naga::Type>,
) -> Option<NonZeroU64> {
    module_info[ty]
        .contains(naga::valid::TypeFlags::SIZED)
        .then(|| module.types[ty].inner.size(module.to_ctx()))
        .and_then(|size| NonZeroU64::new(u64::from(size)))
}

fn compiler_sample_class(kind: ScalarKind) -> Result<CompilerTextureSampleClass, &'static str> {
    match kind {
        ScalarKind::Float => Ok(CompilerTextureSampleClass::Float),
        ScalarKind::Sint => Ok(CompilerTextureSampleClass::Sint),
        ScalarKind::Uint => Ok(CompilerTextureSampleClass::Uint),
        ScalarKind::Bool | ScalarKind::AbstractInt | ScalarKind::AbstractFloat => {
            Err("sampled texture has an unsupported scalar class")
        }
    }
}

fn texture_view_dimension(
    dimension: ImageDimension,
    arrayed: bool,
) -> Result<GpuTextureViewDimension, &'static str> {
    match (dimension, arrayed) {
        (ImageDimension::D1, false) => Ok(GpuTextureViewDimension::D1),
        (ImageDimension::D2, false) => Ok(GpuTextureViewDimension::D2),
        (ImageDimension::D2, true) => Ok(GpuTextureViewDimension::D2Array),
        (ImageDimension::D3, false) => Ok(GpuTextureViewDimension::D3),
        (ImageDimension::Cube, false) => Ok(GpuTextureViewDimension::Cube),
        (ImageDimension::Cube, true) => Ok(GpuTextureViewDimension::CubeArray),
        _ => Err(
            "WGSL image dimension/array combination is outside the admitted RunenGPU vocabulary",
        ),
    }
}

fn storage_texture_access(access: StorageAccess) -> Result<GpuStorageTextureAccess, &'static str> {
    match (
        access.contains(StorageAccess::LOAD),
        access.intersects(StorageAccess::STORE | StorageAccess::ATOMIC),
    ) {
        (true, false) => Ok(GpuStorageTextureAccess::ReadOnly),
        (false, true) => Ok(GpuStorageTextureAccess::WriteOnly),
        (true, true) => Ok(GpuStorageTextureAccess::ReadWrite),
        (false, false) => Err("storage texture declares no load/store access"),
    }
}

fn storage_texture_format(format: StorageFormat) -> Result<GpuTextureFormat, &'static str> {
    match format {
        StorageFormat::R8Unorm => Ok(GpuTextureFormat::R8Unorm),
        StorageFormat::Rgba8Unorm => Ok(GpuTextureFormat::Rgba8Unorm),
        StorageFormat::Bgra8Unorm => Ok(GpuTextureFormat::Bgra8Unorm),
        StorageFormat::R32Uint => Ok(GpuTextureFormat::R32Uint),
        _ => Err("storage texture format is outside the accepted RunenGPU format vocabulary"),
    }
}

fn normalize_vertex_input(
    module: &naga::Module,
    function: &naga::Function,
    entry_point: GpuEntryPointName,
    source_label: &str,
) -> Result<GpuObservedVertexInputSignature, GpuProgramContractError> {
    let mut locations = Vec::new();
    let mut builtins = Vec::new();
    for argument in &function.arguments {
        collect_io(
            module,
            argument.ty,
            argument.binding.as_ref(),
            &mut locations,
            &mut |builtin| match builtin {
                BuiltIn::VertexIndex => Ok(GpuVertexInputBuiltin::VertexIndex),
                BuiltIn::InstanceIndex => Ok(GpuVertexInputBuiltin::InstanceIndex),
                _ => Err("vertex input uses an unsupported builtin"),
            },
            &mut builtins,
        )
        .map_err(|detail| stage_io_failure(source_label, detail))?;
    }
    GpuObservedVertexInputSignature::new(entry_point, locations, builtins)
}

fn normalize_fragment_output(
    module: &naga::Module,
    function: &naga::Function,
    entry_point: GpuEntryPointName,
    source_label: &str,
) -> Result<GpuObservedFragmentOutputSignature, GpuProgramContractError> {
    let mut locations = Vec::new();
    let mut builtins = Vec::new();
    if let Some(result) = &function.result {
        collect_io(
            module,
            result.ty,
            result.binding.as_ref(),
            &mut locations,
            &mut |builtin| match builtin {
                BuiltIn::FragDepth => Ok(GpuFragmentOutputBuiltin::FragDepth),
                BuiltIn::SampleMask => Ok(GpuFragmentOutputBuiltin::SampleMask),
                _ => Err("fragment output uses an unsupported builtin"),
            },
            &mut builtins,
        )
        .map_err(|detail| stage_io_failure(source_label, detail))?;
    }
    GpuObservedFragmentOutputSignature::new(entry_point, locations, builtins)
}

fn collect_io<B>(
    module: &naga::Module,
    ty: naga::Handle<naga::Type>,
    binding: Option<&Binding>,
    locations: &mut Vec<GpuShaderIoLocation>,
    map_builtin: &mut impl FnMut(BuiltIn) -> Result<B, &'static str>,
    builtins: &mut Vec<B>,
) -> Result<(), String> {
    if binding.is_none() {
        if let TypeInner::Struct { members, .. } = &module.types[ty].inner {
            for member in members {
                collect_io(
                    module,
                    member.ty,
                    member.binding.as_ref(),
                    locations,
                    map_builtin,
                    builtins,
                )?;
            }
        }
        return Ok(());
    }
    match binding {
        Some(Binding::Location {
            location,
            blend_src: None,
            ..
        }) => {
            locations.push(GpuShaderIoLocation::new(
                *location,
                io_value_type(module, ty)?,
            ));
            Ok(())
        }
        Some(Binding::Location {
            blend_src: Some(_), ..
        }) => Err(
            "dual-source blend output is outside the accepted RunenGPU stage-IO vocabulary"
                .to_string(),
        ),
        Some(Binding::BuiltIn(builtin)) => {
            builtins.push(map_builtin(*builtin).map_err(str::to_string)?);
            Ok(())
        }
        None => Err(
            "entry-point IO lacks an explicit location/builtin or a typed struct member"
                .to_string(),
        ),
    }
}

fn io_value_type(
    module: &naga::Module,
    ty: naga::Handle<naga::Type>,
) -> Result<GpuShaderIoValueType, String> {
    let (scalar, width) = match module.types[ty].inner {
        TypeInner::Scalar(scalar) => (scalar, 1),
        TypeInner::Vector { size, scalar } => (scalar, vector_width(size)),
        _ => return Err("entry-point IO uses an unsupported non-scalar/vector type".to_string()),
    };
    let scalar_class = match scalar.kind {
        ScalarKind::Float => GpuShaderIoScalarClass::Float,
        ScalarKind::Sint => GpuShaderIoScalarClass::Sint,
        ScalarKind::Uint => GpuShaderIoScalarClass::Uint,
        _ => return Err("entry-point IO uses an unsupported scalar class".to_string()),
    };
    GpuShaderIoValueType::try_new(scalar_class, width).map_err(|error| error.to_string())
}

const fn vector_width(size: VectorSize) -> u8 {
    size as u8
}

fn runen_stage(stage: ShaderStage) -> GpuShaderStage {
    match stage {
        ShaderStage::Compute => GpuShaderStage::Compute,
        ShaderStage::Vertex => GpuShaderStage::Vertex,
        ShaderStage::Fragment => GpuShaderStage::Fragment,
        _ => unreachable!("WGSL parser admitted a non-WGSL shader stage"),
    }
}

fn invalid(
    operation: &'static str,
    label: &str,
    cause: GpuProgramContractCause,
    detail: impl Into<String>,
) -> GpuProgramContractError {
    GpuProgramContractError::invalid_with_detail(
        operation,
        label,
        cause,
        detail,
        "correct canonical WGSL or the selected program contract before admission",
    )
}

fn stage_io_failure(source_label: &str, detail: impl Into<String>) -> GpuProgramContractError {
    GpuProgramContractError::invalid_with_detail(
        "admit canonical WGSL program",
        source_label,
        GpuProgramContractCause::StageIoSignatureInvalid,
        detail,
        "use only supported backend-neutral scalar/vector shader-stage IO signatures",
    )
}
