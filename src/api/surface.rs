use super::{GpuContextAffinity, GpuTextureFormat, GpuTextureUsage};
use core::fmt;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};
use std::collections::BTreeSet;
use std::num::NonZeroU64;
use std::sync::atomic::{AtomicU64, Ordering};

/// Safe host target accepted by the backend-neutral RunenGPU surface API.
///
/// The contract is intentionally the standardized raw-window-handle display/window pair rather
/// than a WGPU or window-system object. The private backend retains the producer and performs its
/// own target conversion.
#[cfg(not(target_arch = "wasm32"))]
pub trait GpuSurfaceTarget: HasDisplayHandle + HasWindowHandle + Send + Sync + 'static {}

#[cfg(not(target_arch = "wasm32"))]
impl<T> GpuSurfaceTarget for T where T: HasDisplayHandle + HasWindowHandle + Send + Sync + 'static {}

/// Safe host target accepted by the backend-neutral RunenGPU surface API.
///
/// Web targets deliberately do not inherit native-only thread-safety requirements.
#[cfg(target_arch = "wasm32")]
pub trait GpuSurfaceTarget: HasDisplayHandle + HasWindowHandle + 'static {}

#[cfg(target_arch = "wasm32")]
impl<T> GpuSurfaceTarget for T where T: HasDisplayHandle + HasWindowHandle + 'static {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuSurfaceId(NonZeroU64);

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuSurfaceGeneration(NonZeroU64);

impl GpuSurfaceGeneration {
    pub(crate) const fn first() -> Self {
        Self(NonZeroU64::MIN)
    }

    pub(crate) fn next(self, surface: GpuSurfaceId) -> Result<Self, GpuSurfaceError> {
        let next = self.0.get().checked_add(1).and_then(NonZeroU64::new);
        next.map(Self).ok_or_else(|| {
            GpuSurfaceError::new(
                GpuSurfaceErrorCategory::GenerationExhausted,
                Some(surface),
                "surface generation counter exhausted",
            )
        })
    }
}

/// Opaque operational surface reference.
///
/// The process-local ID is only correlation identity. Affinity and generation are carried with the
/// reference so foreign-context and stale-generation rejection never requires a global registry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GpuSurfaceHandle {
    id: GpuSurfaceId,
    affinity: GpuContextAffinity,
    generation: GpuSurfaceGeneration,
}

impl GpuSurfaceHandle {
    pub(crate) const fn new(
        id: GpuSurfaceId,
        affinity: GpuContextAffinity,
        generation: GpuSurfaceGeneration,
    ) -> Self {
        Self {
            id,
            affinity,
            generation,
        }
    }

    pub const fn id(self) -> GpuSurfaceId {
        self.id
    }

    pub const fn affinity(self) -> GpuContextAffinity {
        self.affinity
    }

    pub const fn generation(self) -> GpuSurfaceGeneration {
        self.generation
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuSurfacePresentMode {
    Fifo,
    FifoRelaxed,
    Immediate,
    Mailbox,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuSurfaceAlphaMode {
    Opaque,
    PreMultiplied,
    PostMultiplied,
    Inherit,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuSurfaceCapabilities {
    formats: Vec<GpuTextureFormat>,
    usages: Vec<GpuTextureUsage>,
    present_modes: Vec<GpuSurfacePresentMode>,
    alpha_modes: Vec<GpuSurfaceAlphaMode>,
}

impl GpuSurfaceCapabilities {
    pub(crate) fn from_normalized_facts(
        formats: Vec<GpuTextureFormat>,
        usages: Vec<GpuTextureUsage>,
        present_modes: Vec<GpuSurfacePresentMode>,
        alpha_modes: Vec<GpuSurfaceAlphaMode>,
    ) -> Self {
        Self {
            formats,
            usages,
            present_modes,
            alpha_modes,
        }
    }

    pub fn formats(&self) -> &[GpuTextureFormat] {
        &self.formats
    }

    pub fn usages(&self) -> &[GpuTextureUsage] {
        &self.usages
    }

    pub fn present_modes(&self) -> &[GpuSurfacePresentMode] {
        &self.present_modes
    }

    pub fn alpha_modes(&self) -> &[GpuSurfaceAlphaMode] {
        &self.alpha_modes
    }

    pub fn supports_format(&self, format: GpuTextureFormat) -> bool {
        self.formats.contains(&format)
    }

    pub fn supports_usage(&self, usage: GpuTextureUsage) -> bool {
        self.usages.contains(&usage)
    }

    pub fn supports_present_mode(&self, mode: GpuSurfacePresentMode) -> bool {
        self.present_modes.contains(&mode)
    }

    pub fn supports_alpha_mode(&self, mode: GpuSurfaceAlphaMode) -> bool {
        self.alpha_modes.contains(&mode)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuSurfaceConfiguration {
    width: u32,
    height: u32,
    format: GpuTextureFormat,
    usages: Vec<GpuTextureUsage>,
    present_mode: GpuSurfacePresentMode,
    alpha_mode: GpuSurfaceAlphaMode,
    desired_maximum_frame_latency: u32,
    view_formats: Vec<GpuTextureFormat>,
}

impl GpuSurfaceConfiguration {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        width: u32,
        height: u32,
        format: GpuTextureFormat,
        usages: impl IntoIterator<Item = GpuTextureUsage>,
        present_mode: GpuSurfacePresentMode,
        alpha_mode: GpuSurfaceAlphaMode,
        desired_maximum_frame_latency: u32,
        view_formats: impl IntoIterator<Item = GpuTextureFormat>,
    ) -> Result<Self, GpuSurfaceConfigurationError> {
        if width == 0 || height == 0 {
            return Err(GpuSurfaceConfigurationError::new(
                GpuSurfaceConfigurationCause::ZeroExtent,
                "provide a nonzero surface width and height",
            ));
        }
        if format.is_depth() {
            return Err(GpuSurfaceConfigurationError::new(
                GpuSurfaceConfigurationCause::InvalidFormat,
                "choose a normalized color format for presentation",
            ));
        }
        if desired_maximum_frame_latency == 0 {
            return Err(GpuSurfaceConfigurationError::new(
                GpuSurfaceConfigurationCause::ZeroFrameLatency,
                "request at least one frame of presentation latency",
            ));
        }

        let usages = usages.into_iter().collect::<BTreeSet<_>>();
        if usages.is_empty() {
            return Err(GpuSurfaceConfigurationError::new(
                GpuSurfaceConfigurationCause::EmptyUsage,
                "declare at least one surface texture usage",
            ));
        }
        if usages.iter().any(|usage| {
            !matches!(
                usage,
                GpuTextureUsage::ColorAttachment
                    | GpuTextureUsage::CopySource
                    | GpuTextureUsage::CopyDestination
            )
        }) {
            return Err(GpuSurfaceConfigurationError::new(
                GpuSurfaceConfigurationCause::UnsupportedUsage,
                "G7A surfaces permit only color-attachment and supported copy usages",
            ));
        }

        let view_formats = view_formats.into_iter().collect::<BTreeSet<_>>();
        if view_formats
            .iter()
            .any(|view_format| !surface_view_format_is_compatible(format, *view_format))
        {
            return Err(GpuSurfaceConfigurationError::new(
                GpuSurfaceConfigurationCause::InvalidViewFormat,
                "use the surface format or its normalized sRGB/non-sRGB pair as a view format",
            ));
        }

        Ok(Self {
            width,
            height,
            format,
            usages: usages.into_iter().collect(),
            present_mode,
            alpha_mode,
            desired_maximum_frame_latency,
            view_formats: view_formats.into_iter().collect(),
        })
    }

    pub const fn width(&self) -> u32 {
        self.width
    }

    pub const fn height(&self) -> u32 {
        self.height
    }

    pub const fn format(&self) -> GpuTextureFormat {
        self.format
    }

    pub fn usages(&self) -> &[GpuTextureUsage] {
        &self.usages
    }

    pub const fn present_mode(&self) -> GpuSurfacePresentMode {
        self.present_mode
    }

    pub const fn alpha_mode(&self) -> GpuSurfaceAlphaMode {
        self.alpha_mode
    }

    pub const fn desired_maximum_frame_latency(&self) -> u32 {
        self.desired_maximum_frame_latency
    }

    pub fn view_formats(&self) -> &[GpuTextureFormat] {
        &self.view_formats
    }
}

fn surface_view_format_is_compatible(
    format: GpuTextureFormat,
    view_format: GpuTextureFormat,
) -> bool {
    format == view_format
        || matches!(
            (format, view_format),
            (
                GpuTextureFormat::Rgba8Unorm,
                GpuTextureFormat::Rgba8UnormSrgb
            ) | (
                GpuTextureFormat::Rgba8UnormSrgb,
                GpuTextureFormat::Rgba8Unorm
            ) | (
                GpuTextureFormat::Bgra8Unorm,
                GpuTextureFormat::Bgra8UnormSrgb
            ) | (
                GpuTextureFormat::Bgra8UnormSrgb,
                GpuTextureFormat::Bgra8Unorm
            )
        )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GpuSurfaceConfigurationCause {
    ZeroExtent,
    InvalidFormat,
    EmptyUsage,
    UnsupportedUsage,
    ZeroFrameLatency,
    InvalidViewFormat,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuSurfaceConfigurationError {
    cause: GpuSurfaceConfigurationCause,
    correction: &'static str,
}

impl GpuSurfaceConfigurationError {
    const fn new(cause: GpuSurfaceConfigurationCause, correction: &'static str) -> Self {
        Self { cause, correction }
    }

    pub const fn cause(&self) -> GpuSurfaceConfigurationCause {
        self.cause
    }

    pub const fn correction(&self) -> &'static str {
        self.correction
    }
}

impl fmt::Display for GpuSurfaceConfigurationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "invalid GPU surface configuration ({:?}); correction: {}",
            self.cause, self.correction
        )
    }
}

impl std::error::Error for GpuSurfaceConfigurationError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum GpuSurfaceErrorCategory {
    BackendCreationFailure,
    AdapterIncompatible,
    UnknownSurface,
    ForeignContext,
    StaleGeneration,
    UnsupportedFormat,
    UnsupportedUsage,
    UnsupportedPresentMode,
    UnsupportedAlphaMode,
    UnsupportedViewFormat,
    ContextOrDeviceUnavailableOrLost,
    IdentityExhausted,
    GenerationExhausted,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuSurfaceError {
    category: GpuSurfaceErrorCategory,
    surface: Option<GpuSurfaceId>,
    detail: Option<String>,
}

impl GpuSurfaceError {
    pub(crate) fn new(
        category: GpuSurfaceErrorCategory,
        surface: Option<GpuSurfaceId>,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            category,
            surface,
            detail: super::context::sanitized_diagnostic(detail.into()),
        }
    }

    pub const fn category(&self) -> GpuSurfaceErrorCategory {
        self.category
    }

    pub const fn surface(&self) -> Option<GpuSurfaceId> {
        self.surface
    }

    pub fn detail(&self) -> Option<&str> {
        self.detail.as_deref()
    }
}

impl fmt::Display for GpuSurfaceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "GPU surface operation failed ({:?})",
            self.category
        )?;
        if let Some(detail) = &self.detail {
            write!(formatter, ": {detail}")?;
        }
        Ok(())
    }
}

impl std::error::Error for GpuSurfaceError {}

#[derive(Debug)]
pub(crate) struct GpuSurfaceIdAllocator {
    next: AtomicU64,
}

impl GpuSurfaceIdAllocator {
    const fn new(next: u64) -> Self {
        Self {
            next: AtomicU64::new(next),
        }
    }

    fn allocate(&self) -> Result<GpuSurfaceId, GpuSurfaceError> {
        let value = self
            .next
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |current| {
                (current != 0).then_some(if current == u64::MAX { 0 } else { current + 1 })
            })
            .map_err(|_| {
                GpuSurfaceError::new(
                    GpuSurfaceErrorCategory::IdentityExhausted,
                    None,
                    "surface identifier allocator exhausted",
                )
            })?;
        Ok(GpuSurfaceId(
            NonZeroU64::new(value).expect("surface identifier allocator never returns zero"),
        ))
    }
}

static PRODUCTION_SURFACE_IDS: GpuSurfaceIdAllocator = GpuSurfaceIdAllocator::new(1);

pub(crate) fn allocate_surface_id() -> Result<GpuSurfaceId, GpuSurfaceError> {
    PRODUCTION_SURFACE_IDS.allocate()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn configuration_normalizes_set_like_usage_and_view_format_inputs() {
        let configuration = GpuSurfaceConfiguration::new(
            1280,
            720,
            GpuTextureFormat::Bgra8Unorm,
            [
                GpuTextureUsage::CopySource,
                GpuTextureUsage::ColorAttachment,
                GpuTextureUsage::CopySource,
            ],
            GpuSurfacePresentMode::Fifo,
            GpuSurfaceAlphaMode::Opaque,
            2,
            [
                GpuTextureFormat::Bgra8UnormSrgb,
                GpuTextureFormat::Bgra8Unorm,
                GpuTextureFormat::Bgra8UnormSrgb,
            ],
        )
        .unwrap();

        assert_eq!(
            configuration.usages(),
            &[
                GpuTextureUsage::ColorAttachment,
                GpuTextureUsage::CopySource
            ]
        );
        assert_eq!(
            configuration.view_formats(),
            &[
                GpuTextureFormat::Bgra8Unorm,
                GpuTextureFormat::Bgra8UnormSrgb
            ]
        );
    }

    #[test]
    fn configuration_rejects_product_policy_and_non_surface_resource_usages() {
        assert!(matches!(
            GpuSurfaceConfiguration::new(
                0,
                720,
                GpuTextureFormat::Bgra8Unorm,
                [GpuTextureUsage::ColorAttachment],
                GpuSurfacePresentMode::Fifo,
                GpuSurfaceAlphaMode::Opaque,
                2,
                [],
            ),
            Err(error) if error.cause() == GpuSurfaceConfigurationCause::ZeroExtent
        ));
        assert!(matches!(
            GpuSurfaceConfiguration::new(
                1280,
                720,
                GpuTextureFormat::Depth32Float,
                [GpuTextureUsage::ColorAttachment],
                GpuSurfacePresentMode::Fifo,
                GpuSurfaceAlphaMode::Opaque,
                2,
                [],
            ),
            Err(error) if error.cause() == GpuSurfaceConfigurationCause::InvalidFormat
        ));
        assert!(matches!(
            GpuSurfaceConfiguration::new(
                1280,
                720,
                GpuTextureFormat::Bgra8Unorm,
                [GpuTextureUsage::Sampled],
                GpuSurfacePresentMode::Fifo,
                GpuSurfaceAlphaMode::Opaque,
                2,
                [],
            ),
            Err(error) if error.cause() == GpuSurfaceConfigurationCause::UnsupportedUsage
        ));
    }

    #[test]
    fn isolated_surface_allocator_proves_nonzero_uniqueness_and_exhaustion() {
        let allocator = GpuSurfaceIdAllocator::new(1);
        let first = allocator.allocate().unwrap();
        let second = allocator.allocate().unwrap();
        assert_ne!(first, second);

        let exhausted = GpuSurfaceIdAllocator::new(u64::MAX);
        assert!(exhausted.allocate().is_ok());
        assert!(matches!(
            exhausted.allocate(),
            Err(error) if error.category() == GpuSurfaceErrorCategory::IdentityExhausted
        ));
    }
}
