#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum GpuSamplerClass {
    Filtering,
    NonFiltering,
    Comparison,
}
