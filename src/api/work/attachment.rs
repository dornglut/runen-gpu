use super::super::{
    GpuAttachmentLoadKind, GpuAttachmentStore, GpuDepthStencilAccess, GpuTextureAccess,
    GpuTextureAccessKind, GpuTextureAccessResource, GpuTextureAspect, GpuTextureFormat,
    GpuTextureSubresourceRange, GpuTextureUsage, GpuTextureViewDimension, GpuTextureViewHandle,
    GpuWorkOperationCause, GpuWorkOperationError,
};
use super::mip_extent;
use core::fmt;
use core::hash::{Hash, Hasher};

#[derive(Clone, Copy)]
pub struct GpuColorClearValue {
    bits: [u64; 4],
}

impl GpuColorClearValue {
    pub fn new(red: f64, green: f64, blue: f64, alpha: f64) -> Result<Self, GpuWorkOperationError> {
        Self::from_array([red, green, blue, alpha])
    }

    pub fn from_array(components: [f64; 4]) -> Result<Self, GpuWorkOperationError> {
        let mut bits = [0; 4];
        for (index, component) in components.into_iter().enumerate() {
            if !component.is_finite() {
                return Err(GpuWorkOperationError::invalid(
                    "construct GPU color clear value",
                    format!("component {index}"),
                    None,
                    GpuWorkOperationCause::NonFiniteClearValue,
                    "provide four finite components",
                ));
            }
            bits[index] = canonical_f64_bits(component);
        }
        Ok(Self { bits })
    }

    pub fn components(self) -> [f64; 4] {
        self.bits.map(f64::from_bits)
    }
}

impl fmt::Debug for GpuColorClearValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("GpuColorClearValue")
            .field(&self.components())
            .finish()
    }
}

impl PartialEq for GpuColorClearValue {
    fn eq(&self, other: &Self) -> bool {
        self.bits == other.bits
    }
}

impl Eq for GpuColorClearValue {}

impl PartialOrd for GpuColorClearValue {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for GpuColorClearValue {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.bits.cmp(&other.bits)
    }
}

impl Hash for GpuColorClearValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.bits.hash(state);
    }
}

#[derive(Clone, Copy)]
pub struct GpuDepthClearValue {
    bits: u32,
}

impl GpuDepthClearValue {
    pub fn new(value: f32) -> Result<Self, GpuWorkOperationError> {
        if !value.is_finite() {
            return Err(GpuWorkOperationError::invalid(
                "construct GPU depth clear value",
                "depth",
                None,
                GpuWorkOperationCause::NonFiniteClearValue,
                "provide a finite normalized depth value",
            ));
        }
        if !(0.0..=1.0).contains(&value) {
            return Err(GpuWorkOperationError::invalid(
                "construct GPU depth clear value",
                "depth",
                None,
                GpuWorkOperationCause::OutOfRangeClearValue,
                "keep depth inside 0.0 through 1.0",
            ));
        }
        Ok(Self {
            bits: canonical_f32_bits(value),
        })
    }

    pub fn value(self) -> f32 {
        f32::from_bits(self.bits)
    }
}

impl fmt::Debug for GpuDepthClearValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("GpuDepthClearValue")
            .field(&self.value())
            .finish()
    }
}

impl PartialEq for GpuDepthClearValue {
    fn eq(&self, other: &Self) -> bool {
        self.bits == other.bits
    }
}

impl Eq for GpuDepthClearValue {}

impl PartialOrd for GpuDepthClearValue {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for GpuDepthClearValue {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.bits.cmp(&other.bits)
    }
}

impl Hash for GpuDepthClearValue {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.bits.hash(state);
    }
}

const fn canonical_f64_bits(value: f64) -> u64 {
    if value == 0.0 {
        0.0_f64.to_bits()
    } else {
        value.to_bits()
    }
}

const fn canonical_f32_bits(value: f32) -> u32 {
    if value == 0.0 {
        0.0_f32.to_bits()
    } else {
        value.to_bits()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuColorAttachmentLoad {
    Load,
    Clear(GpuColorClearValue),
}

impl GpuColorAttachmentLoad {
    pub const fn kind(self) -> GpuAttachmentLoadKind {
        match self {
            Self::Load => GpuAttachmentLoadKind::Load,
            Self::Clear(_) => GpuAttachmentLoadKind::Clear,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuDepthAttachmentLoad {
    Load,
    Clear(GpuDepthClearValue),
}

impl GpuDepthAttachmentLoad {
    pub const fn kind(self) -> GpuAttachmentLoadKind {
        match self {
            Self::Load => GpuAttachmentLoadKind::Load,
            Self::Clear(_) => GpuAttachmentLoadKind::Clear,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuStencilClearValue(u8);

impl GpuStencilClearValue {
    pub fn new(value: u32) -> Result<Self, GpuWorkOperationError> {
        let value = u8::try_from(value).map_err(|_| {
            GpuWorkOperationError::invalid(
                "construct GPU stencil clear value",
                format!("stencil={value}"),
                None,
                GpuWorkOperationCause::OutOfRangeClearValue,
                "keep the Stencil8 clear value inside 0 through 255",
            )
        })?;
        Ok(Self(value))
    }

    pub const fn value(self) -> u32 {
        self.0 as u32
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuStencilAttachmentLoad {
    Load,
    Clear(GpuStencilClearValue),
}

impl GpuStencilAttachmentLoad {
    pub const fn kind(self) -> GpuAttachmentLoadKind {
        match self {
            Self::Load => GpuAttachmentLoadKind::Load,
            Self::Clear(_) => GpuAttachmentLoadKind::Clear,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuDepthAttachmentState {
    access: GpuDepthStencilAccess,
    load: GpuDepthAttachmentLoad,
    store: GpuAttachmentStore,
}

impl GpuDepthAttachmentState {
    pub fn new(
        access: GpuDepthStencilAccess,
        load: GpuDepthAttachmentLoad,
        store: GpuAttachmentStore,
    ) -> Result<Self, GpuWorkOperationError> {
        validate_read_only_attachment_state(
            "construct GPU depth attachment state",
            "depth",
            access,
            load.kind(),
            store,
        )?;
        Ok(Self {
            access,
            load,
            store,
        })
    }

    pub const fn access(self) -> GpuDepthStencilAccess {
        self.access
    }
    pub const fn load(self) -> GpuDepthAttachmentLoad {
        self.load
    }
    pub const fn store(self) -> GpuAttachmentStore {
        self.store
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuStencilAttachmentState {
    access: GpuDepthStencilAccess,
    load: GpuStencilAttachmentLoad,
    store: GpuAttachmentStore,
}

impl GpuStencilAttachmentState {
    pub fn new(
        access: GpuDepthStencilAccess,
        load: GpuStencilAttachmentLoad,
        store: GpuAttachmentStore,
    ) -> Result<Self, GpuWorkOperationError> {
        validate_read_only_attachment_state(
            "construct GPU stencil attachment state",
            "stencil",
            access,
            load.kind(),
            store,
        )?;
        Ok(Self {
            access,
            load,
            store,
        })
    }

    pub const fn access(self) -> GpuDepthStencilAccess {
        self.access
    }
    pub const fn load(self) -> GpuStencilAttachmentLoad {
        self.load
    }
    pub const fn store(self) -> GpuAttachmentStore {
        self.store
    }
}

fn validate_read_only_attachment_state(
    operation: &'static str,
    aspect: &'static str,
    access: GpuDepthStencilAccess,
    load_kind: GpuAttachmentLoadKind,
    store: GpuAttachmentStore,
) -> Result<(), GpuWorkOperationError> {
    if access == GpuDepthStencilAccess::ReadOnly
        && (load_kind != GpuAttachmentLoadKind::Load || store != GpuAttachmentStore::Store)
    {
        return Err(GpuWorkOperationError::invalid(
            operation,
            aspect,
            None,
            GpuWorkOperationCause::InvalidAttachment,
            "use canonical Load + Store semantics for a read-only aspect, or select read-write access before clearing or discarding",
        ));
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuMultisampleResolveTarget {
    destination: GpuTextureViewHandle,
    access: GpuTextureAccess,
}

impl GpuMultisampleResolveTarget {
    pub fn new(destination: GpuTextureViewHandle) -> Result<Self, GpuWorkOperationError> {
        validate_attachment_view(
            &destination,
            "construct GPU multisample resolve target",
            GpuWorkOperationCause::InvalidMultisampleResolve,
            "use one explicit 2D texture view selecting exactly one mip and one array layer",
        )?;
        let label = destination
            .descriptor()
            .texture()
            .descriptor()
            .common()
            .label()
            .as_str()
            .to_string();
        let access = GpuTextureAccess::new(
            GpuTextureAccessResource::TextureView(destination.clone()),
            destination.descriptor().subresources(),
            GpuTextureAccessKind::MultisampleResolveDestination,
        )
        .map_err(|source| {
            GpuWorkOperationError::from_access(
                "construct GPU multisample resolve target",
                label,
                GpuWorkOperationCause::InvalidMultisampleResolve,
                "provide a checked single-sampled color-attachment destination view",
                source,
            )
        })?;
        Ok(Self {
            destination,
            access,
        })
    }

    pub fn destination(&self) -> &GpuTextureViewHandle {
        &self.destination
    }

    pub fn subresources(&self) -> GpuTextureSubresourceRange {
        self.access.normalized_subresources()
    }

    pub fn access(&self) -> &GpuTextureAccess {
        &self.access
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuRenderColorAttachment {
    source: GpuTextureViewHandle,
    load: GpuColorAttachmentLoad,
    store: GpuAttachmentStore,
    resolve_target: Option<GpuMultisampleResolveTarget>,
    source_access: GpuTextureAccess,
}

impl GpuRenderColorAttachment {
    pub fn new(
        source: GpuTextureViewHandle,
        load: GpuColorAttachmentLoad,
        store: GpuAttachmentStore,
        resolve_target: Option<GpuMultisampleResolveTarget>,
    ) -> Result<Self, GpuWorkOperationError> {
        validate_attachment_view(
            &source,
            "construct GPU render color attachment",
            GpuWorkOperationCause::InvalidAttachment,
            "use one explicit 2D color-attachment view selecting exactly one mip and one array layer",
        )?;
        let label = source
            .descriptor()
            .texture()
            .descriptor()
            .common()
            .label()
            .as_str()
            .to_string();
        let source_access = GpuTextureAccess::new(
            GpuTextureAccessResource::TextureView(source.clone()),
            source.descriptor().subresources(),
            GpuTextureAccessKind::ColorAttachment {
                load_kind: load.kind(),
                store,
            },
        )
        .map_err(|source| {
            GpuWorkOperationError::from_access(
                "construct GPU render color attachment",
                label.clone(),
                GpuWorkOperationCause::InvalidAttachment,
                "provide a checked color attachment view with compatible descriptor usage",
                source,
            )
        })?;
        if let Some(resolve) = &resolve_target {
            validate_multisample_resolve(&source_access, resolve)?;
        }
        Ok(Self {
            source,
            load,
            store,
            resolve_target,
            source_access,
        })
    }

    pub fn source(&self) -> &GpuTextureViewHandle {
        &self.source
    }

    pub fn subresources(&self) -> GpuTextureSubresourceRange {
        self.source_access.normalized_subresources()
    }

    pub const fn load(&self) -> GpuColorAttachmentLoad {
        self.load
    }

    pub const fn store(&self) -> GpuAttachmentStore {
        self.store
    }

    pub fn resolve_target(&self) -> Option<&GpuMultisampleResolveTarget> {
        self.resolve_target.as_ref()
    }

    pub fn source_access(&self) -> &GpuTextureAccess {
        &self.source_access
    }
}

fn validate_multisample_resolve(
    source: &GpuTextureAccess,
    destination: &GpuMultisampleResolveTarget,
) -> Result<(), GpuWorkOperationError> {
    let source_texture = source.normalized_texture();
    let destination_texture = destination.access().normalized_texture();
    let label = source_texture
        .descriptor()
        .common()
        .label()
        .as_str()
        .to_string();
    let source_range = source.normalized_subresources();
    let destination_range = destination.access().normalized_subresources();
    let same_shape = source_range.mip_level_count() == destination_range.mip_level_count()
        && source_range.array_layer_count() == destination_range.array_layer_count()
        && source_range.aspect() == GpuTextureAspect::Color
        && destination_range.aspect() == GpuTextureAspect::Color
        && mip_extent(source_texture, source_range.base_mip_level())
            == mip_extent(destination_texture, destination_range.base_mip_level());
    let valid = source_texture.descriptor().sample_count() > 1
        && destination_texture.descriptor().sample_count() == 1
        && effective_texture_format(source.resource())
            == effective_view_format(destination.destination())
        && source_texture.descriptor().dimension() == destination_texture.descriptor().dimension()
        && same_shape
        && source_texture != destination_texture;
    if !valid {
        return Err(GpuWorkOperationError::invalid(
            "validate GPU multisample resolve",
            label,
            Some(source_texture.diagnostic_identity()),
            GpuWorkOperationCause::InvalidMultisampleResolve,
            "use non-aliasing multisampled source and single-sampled destination attachment views with matching color format, extent, and subresources",
        ));
    }
    Ok(())
}

fn effective_texture_format(resource: &GpuTextureAccessResource) -> GpuTextureFormat {
    match resource {
        GpuTextureAccessResource::Texture(texture) => texture.descriptor().format(),
        GpuTextureAccessResource::TextureView(view) => effective_view_format(view),
    }
}

fn effective_view_format(view: &GpuTextureViewHandle) -> GpuTextureFormat {
    view.descriptor()
        .format()
        .unwrap_or_else(|| view.descriptor().texture().descriptor().format())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct GpuAttachmentAspectAccessSpec {
    aspect: GpuTextureAspect,
    access: GpuDepthStencilAccess,
    load_kind: GpuAttachmentLoadKind,
    store: GpuAttachmentStore,
}

fn attachment_aspect_access_specs(
    depth: Option<GpuDepthAttachmentState>,
    stencil: Option<GpuStencilAttachmentState>,
) -> impl Iterator<Item = GpuAttachmentAspectAccessSpec> {
    let depth = depth.map(|state| GpuAttachmentAspectAccessSpec {
        aspect: GpuTextureAspect::DepthOnly,
        access: state.access(),
        load_kind: state.load().kind(),
        store: state.store(),
    });
    let stencil = stencil.map(|state| GpuAttachmentAspectAccessSpec {
        aspect: GpuTextureAspect::StencilOnly,
        access: state.access(),
        load_kind: state.load().kind(),
        store: state.store(),
    });
    depth.into_iter().chain(stencil)
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuRenderDepthStencilAttachment {
    source: GpuTextureViewHandle,
    depth: Option<GpuDepthAttachmentState>,
    stencil: Option<GpuStencilAttachmentState>,
    depth_access: Option<GpuTextureAccess>,
    stencil_access: Option<GpuTextureAccess>,
}

impl GpuRenderDepthStencilAttachment {
    pub fn new(
        source: GpuTextureViewHandle,
        depth: Option<GpuDepthAttachmentState>,
        stencil: Option<GpuStencilAttachmentState>,
    ) -> Result<Self, GpuWorkOperationError> {
        validate_attachment_view(
            &source,
            "construct GPU render depth/stencil attachment",
            GpuWorkOperationCause::InvalidAttachment,
            "use one explicit 2D depth/stencil attachment view selecting exactly one mip and one array layer",
        )?;
        let texture = source.descriptor().texture();
        let format = effective_view_format(&source);
        let label = texture.descriptor().common().label().as_str().to_string();
        if depth.is_none() && stencil.is_none() {
            return Err(GpuWorkOperationError::invalid(
                "construct GPU render depth/stencil attachment",
                label,
                Some(texture.diagnostic_identity()),
                GpuWorkOperationCause::InvalidAttachment,
                "provide at least one depth or stencil attachment state",
            ));
        }
        if depth.is_some() && !format.is_depth() {
            return Err(GpuWorkOperationError::invalid(
                "construct GPU render depth/stencil attachment",
                label,
                Some(texture.diagnostic_identity()),
                GpuWorkOperationCause::InvalidAttachment,
                "use a view whose effective format contains a depth aspect before configuring depth attachment state",
            ));
        }
        if stencil.is_some() && !format.is_stencil() {
            return Err(GpuWorkOperationError::invalid(
                "construct GPU render depth/stencil attachment",
                label,
                Some(texture.diagnostic_identity()),
                GpuWorkOperationCause::InvalidAttachment,
                "use a view whose effective format contains a stencil aspect before configuring stencil attachment state",
            ));
        }
        if texture
            .descriptor()
            .usages()
            .contains(GpuTextureUsage::TransientAttachment)
            && (depth.is_some() != format.is_depth() || stencil.is_some() != format.is_stencil())
        {
            return Err(GpuWorkOperationError::invalid(
                "construct GPU render depth/stencil attachment",
                label,
                Some(texture.diagnostic_identity()),
                GpuWorkOperationCause::InvalidAttachment,
                "provide writable Clear + Discard state for every depth/stencil aspect present in a transient attachment format",
            ));
        }

        let make_access = |aspect, access, load_kind, store| {
            let selected = source.descriptor().subresources();
            let range = GpuTextureSubresourceRange::new(
                texture.descriptor().common().label(),
                selected.base_mip_level(),
                selected.mip_level_count(),
                selected.base_array_layer(),
                selected.array_layer_count(),
                aspect,
            )
            .expect("validated attachment subresource counts remain valid");
            GpuTextureAccess::new(
                GpuTextureAccessResource::TextureView(source.clone()),
                range,
                GpuTextureAccessKind::DepthStencilAttachment {
                    access,
                    load_kind,
                    store,
                },
            )
            .map_err(|source| {
                GpuWorkOperationError::from_access(
                    "construct GPU render depth/stencil attachment",
                    label.clone(),
                    GpuWorkOperationCause::InvalidAttachment,
                    "provide attachment state only for aspects selected by a compatible depth/stencil view",
                    source,
                )
            })
        };

        let mut depth_access = None;
        let mut stencil_access = None;
        for spec in attachment_aspect_access_specs(depth, stencil) {
            let access = make_access(spec.aspect, spec.access, spec.load_kind, spec.store)?;
            match spec.aspect {
                GpuTextureAspect::DepthOnly => depth_access = Some(access),
                GpuTextureAspect::StencilOnly => stencil_access = Some(access),
                GpuTextureAspect::All | GpuTextureAspect::Color => {
                    unreachable!("depth/stencil attachment access specs are aspect-specific")
                }
            }
        }

        Ok(Self {
            source,
            depth,
            stencil,
            depth_access,
            stencil_access,
        })
    }

    pub fn source(&self) -> &GpuTextureViewHandle {
        &self.source
    }

    pub const fn depth(&self) -> Option<GpuDepthAttachmentState> {
        self.depth
    }

    pub const fn stencil(&self) -> Option<GpuStencilAttachmentState> {
        self.stencil
    }

    pub fn depth_access(&self) -> Option<&GpuTextureAccess> {
        self.depth_access.as_ref()
    }

    pub fn stencil_access(&self) -> Option<&GpuTextureAccess> {
        self.stencil_access.as_ref()
    }
}

fn validate_attachment_view(
    view: &GpuTextureViewHandle,
    operation: &'static str,
    cause: GpuWorkOperationCause,
    correction: &'static str,
) -> Result<(), GpuWorkOperationError> {
    let descriptor = view.descriptor();
    let subresources = descriptor.subresources();
    if descriptor.dimension() != GpuTextureViewDimension::D2
        || subresources.mip_level_count() != 1
        || subresources.array_layer_count() != 1
    {
        return Err(GpuWorkOperationError::invalid(
            operation,
            descriptor.common().label().as_str(),
            Some(view.diagnostic_identity()),
            cause,
            correction,
        ));
    }
    Ok(())
}

#[cfg(test)]
mod depth_stencil_aspect_tests {
    use super::*;

    #[test]
    fn combined_depth_stencil_access_specs_remain_independent() {
        let depth = GpuDepthAttachmentState::new(
            GpuDepthStencilAccess::ReadOnly,
            GpuDepthAttachmentLoad::Load,
            GpuAttachmentStore::Store,
        )
        .unwrap();
        let stencil = GpuStencilAttachmentState::new(
            GpuDepthStencilAccess::ReadWrite,
            GpuStencilAttachmentLoad::Clear(GpuStencilClearValue::new(7).unwrap()),
            GpuAttachmentStore::Discard,
        )
        .unwrap();

        let specs = attachment_aspect_access_specs(Some(depth), Some(stencil)).collect::<Vec<_>>();
        assert_eq!(
            specs,
            [
                GpuAttachmentAspectAccessSpec {
                    aspect: GpuTextureAspect::DepthOnly,
                    access: GpuDepthStencilAccess::ReadOnly,
                    load_kind: GpuAttachmentLoadKind::Load,
                    store: GpuAttachmentStore::Store,
                },
                GpuAttachmentAspectAccessSpec {
                    aspect: GpuTextureAspect::StencilOnly,
                    access: GpuDepthStencilAccess::ReadWrite,
                    load_kind: GpuAttachmentLoadKind::Clear,
                    store: GpuAttachmentStore::Discard,
                },
            ]
        );
    }
}
