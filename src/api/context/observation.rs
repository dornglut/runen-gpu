use super::descriptor::{
    GpuAdapterClass, GpuAlignmentKind, GpuBackendFamily, GpuContextDescriptor, GpuFormatRole,
    GpuLimitConstraint, GpuLimitKind,
};
use crate::GpuTextureFormat;

impl GpuContextDescriptor {
    /// Iterates the normalized backend allowlist in canonical backend-family order.
    ///
    /// An empty iterator means the request does not constrain backend families.
    pub fn backend_allowlist(&self) -> impl ExactSizeIterator<Item = GpuBackendFamily> + '_ {
        self.allowed_backends().iter().copied()
    }

    /// Iterates explicitly preferred backend families from most to least preferred.
    ///
    /// An empty iterator means the request has no explicit backend preference. The internal
    /// ranking representation remains private and is not part of the public RunenGPU contract.
    pub fn backend_preference_order(&self) -> impl ExactSizeIterator<Item = GpuBackendFamily> + '_ {
        let mut ordered = self
            .backend_preference()
            .iter()
            .map(|(&backend, &priority)| (priority, backend))
            .collect::<Vec<_>>();
        ordered.sort_unstable();
        ordered.into_iter().map(|(_, backend)| backend)
    }

    /// Iterates the normalized adapter-class allowlist in canonical adapter-class order.
    ///
    /// An empty iterator means the request does not constrain adapter classes.
    pub fn adapter_class_allowlist(&self) -> impl ExactSizeIterator<Item = GpuAdapterClass> + '_ {
        self.allowed_adapter_classes().iter().copied()
    }

    /// Iterates normalized limit constraints in canonical limit-kind order.
    pub fn limit_constraints(
        &self,
    ) -> impl ExactSizeIterator<Item = (GpuLimitKind, GpuLimitConstraint)> + '_ {
        self.limits
            .iter()
            .map(|(&kind, &constraint)| (kind, constraint))
    }

    /// Iterates required texture-format roles in canonical `(format, role)` order.
    pub fn format_role_requirements(
        &self,
    ) -> impl ExactSizeIterator<Item = (GpuTextureFormat, GpuFormatRole)> + '_ {
        self.format_roles.iter().copied()
    }

    /// Iterates normalized maximum alignment requirements in canonical alignment-kind order.
    pub fn alignment_requirements(
        &self,
    ) -> impl ExactSizeIterator<Item = (GpuAlignmentKind, u64)> + '_ {
        self.alignments
            .iter()
            .map(|(&kind, &maximum)| (kind, maximum))
    }
}
