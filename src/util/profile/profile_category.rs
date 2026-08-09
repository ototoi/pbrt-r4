use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, Copy)]
#[repr(u32)]
enum EProfileCategory {
    SceneConstruction,
    AccelConstruction,
    TextureLoading,
    MIPMapCreation,
    IntegratorRender,
    RayIntegratorLi,
    SPPMCameraPass,
    SPPMGridConstruction,
    SPPMPhotonPass,
    SPPMStatsUpdate,
    BDPTGenerateSubpath,
    BDPTConnectSubpaths,
    LightDistribLookup,
    LightDistribSpinWait,
    LightDistribCreation,
    DirectLighting,
    EstimateDirect,
    SampleLightImportance,
    ComputeBSDF,
    ComputeLightImportance,
    SampleLight,
    BSDFEvaluation,
    BSDFSampling,
    BSDFPdf,
    BSSRDFEvaluation,
    BSSRDFSampling,
    PhaseFuncEvaluation,
    PhaseFuncSampling,
    AccelIntersect,
    AccelIntersectP,
    GeometricPrimitiveIntersect,
    GeometricPrimitiveIntersectP,
    LightSample,
    LightPdf,
    MediumSample,
    MediumTr,
    TriIntersect,
    TriIntersectP,
    CurveIntersect,
    CurveIntersectP,
    ShapeIntersect,
    ShapeIntersectP,
    GenerateCameraRay,
    MergeFilmTile,
    SplatFilm,
    AddFilmSample,
    StartPixel,
    GetSample,
    TexFiltTrilerp,
    TexFiltEWA,
    TexFiltPtex,
    NumProfCategories,
}

#[derive(Debug, Default, Clone, Copy)]
pub struct ProfileCategory(pub u32);

#[allow(non_upper_case_globals)]
impl ProfileCategory {
    pub const SceneConstruction: Self = Self(EProfileCategory::SceneConstruction as u32);
    pub const AccelConstruction: Self = Self(EProfileCategory::AccelConstruction as u32);
    pub const TextureLoading: Self = Self(EProfileCategory::TextureLoading as u32);
    pub const MIPMapCreation: Self = Self(EProfileCategory::MIPMapCreation as u32);
    pub const IntegratorRender: Self = Self(EProfileCategory::IntegratorRender as u32);
    pub const RayIntegratorLi: Self = Self(EProfileCategory::RayIntegratorLi as u32);
    pub const SPPMCameraPass: Self = Self(EProfileCategory::SPPMCameraPass as u32);
    pub const SPPMGridConstruction: Self = Self(EProfileCategory::SPPMGridConstruction as u32);
    pub const SPPMPhotonPass: Self = Self(EProfileCategory::SPPMPhotonPass as u32);
    pub const SPPMStatsUpdate: Self = Self(EProfileCategory::SPPMStatsUpdate as u32);
    pub const BDPTGenerateSubpath: Self = Self(EProfileCategory::BDPTGenerateSubpath as u32);
    pub const BDPTConnectSubpaths: Self = Self(EProfileCategory::BDPTConnectSubpaths as u32);
    pub const LightDistribLookup: Self = Self(EProfileCategory::LightDistribLookup as u32);
    pub const LightDistribSpinWait: Self = Self(EProfileCategory::LightDistribSpinWait as u32);
    pub const LightDistribCreation: Self = Self(EProfileCategory::LightDistribCreation as u32);
    pub const DirectLighting: Self = Self(EProfileCategory::DirectLighting as u32);
    pub const EstimateDirect: Self = Self(EProfileCategory::EstimateDirect as u32);
    pub const SampleLightImportance: Self = Self(EProfileCategory::SampleLightImportance as u32);
    pub const ComputeBSDF: Self = Self(EProfileCategory::ComputeBSDF as u32);
    pub const ComputeLightImportance: Self = Self(EProfileCategory::ComputeLightImportance as u32);
    pub const SampleBSDFImportance: Self = Self(EProfileCategory::SampleLight as u32);
    pub const BSDFEvaluation: Self = Self(EProfileCategory::BSDFEvaluation as u32);
    pub const BSDFSampling: Self = Self(EProfileCategory::BSDFSampling as u32);
    pub const BSDFPdf: Self = Self(EProfileCategory::BSDFPdf as u32);
    pub const BSSRDFEvaluation: Self = Self(EProfileCategory::BSSRDFEvaluation as u32);
    pub const BSSRDFSampling: Self = Self(EProfileCategory::BSSRDFSampling as u32);
    pub const PhaseFuncEvaluation: Self = Self(EProfileCategory::PhaseFuncEvaluation as u32);
    pub const PhaseFuncSampling: Self = Self(EProfileCategory::PhaseFuncSampling as u32);
    pub const AccelIntersect: Self = Self(EProfileCategory::AccelIntersect as u32);
    pub const AccelIntersectP: Self = Self(EProfileCategory::AccelIntersectP as u32);
    pub const GeometricPrimitiveIntersect: Self =
        Self(EProfileCategory::GeometricPrimitiveIntersect as u32);
    pub const GeometricPrimitiveIntersectP: Self =
        Self(EProfileCategory::GeometricPrimitiveIntersectP as u32);
    pub const LightSample: Self = Self(EProfileCategory::LightSample as u32);
    pub const LightPdf: Self = Self(EProfileCategory::LightPdf as u32);
    pub const MediumSample: Self = Self(EProfileCategory::MediumSample as u32);
    pub const MediumTr: Self = Self(EProfileCategory::MediumTr as u32);
    pub const TriIntersect: Self = Self(EProfileCategory::TriIntersect as u32);
    pub const TriIntersectP: Self = Self(EProfileCategory::TriIntersectP as u32);
    pub const CurveIntersect: Self = Self(EProfileCategory::CurveIntersect as u32);
    pub const CurveIntersectP: Self = Self(EProfileCategory::CurveIntersectP as u32);
    pub const ShapeIntersect: Self = Self(EProfileCategory::ShapeIntersect as u32);
    pub const ShapeIntersectP: Self = Self(EProfileCategory::ShapeIntersectP as u32);
    pub const GenerateCameraRay: Self = Self(EProfileCategory::GenerateCameraRay as u32);
    pub const MergeFilmTile: Self = Self(EProfileCategory::MergeFilmTile as u32);
    pub const SplatFilm: Self = Self(EProfileCategory::SplatFilm as u32);
    pub const AddFilmSample: Self = Self(EProfileCategory::AddFilmSample as u32);
    pub const StartPixel: Self = Self(EProfileCategory::StartPixel as u32);
    pub const GetSample: Self = Self(EProfileCategory::GetSample as u32);
    pub const TexFiltTrilerp: Self = Self(EProfileCategory::TexFiltTrilerp as u32);
    pub const TexFiltEWA: Self = Self(EProfileCategory::TexFiltEWA as u32);
    pub const TexFiltPtex: Self = Self(EProfileCategory::TexFiltPtex as u32);
    pub const NumProfCategories: Self = Self(EProfileCategory::NumProfCategories as u32);
}

const PROF_NAMES: [&str; 53] = [
    "Scene parsing and creation",
    "Acceleration structure creation",
    "Texture loading",
    "MIP map generation",
    "Integrator::Render()",
    "RayIntegrator::Li()",
    "SPPM camera pass",
    "SPPM grid construction",
    "SPPM photon pass",
    "SPPM photon statistics update",
    "BDPT subpath generation",
    "BDPT subpath connections",
    "SpatialLightDistribution lookup",
    "SpatialLightDistribution spin wait",
    "SpatialLightDistribution creation",
    "Direct lighting",
    "Estimate Direct",
    "Sample LightImportance",
    "Compute BSDF",
    "Compute LightImportance",
    "Sample BSDFImportance",
    "BSDF::f()",
    "BSDF::Sample_f()",
    "BSDF::PDF()",
    "BSSRDF::f()",
    "BSSRDF::Sample_f()",
    "PhaseFunction::p()",
    "PhaseFunction::Sample_p()",
    "Accelerator::Intersect()",
    "Accelerator::IntersectP()",
    "GeometricPrimitive::Intersect()",
    "GeometricPrimitive::IntersectP()",
    "Light::Sample_*()",
    "Light::Pdf()",
    "Medium::Sample()",
    "Medium::Tr()",
    "Triangle::Intersect()",
    "Triangle::IntersectP()",
    "Curve::Intersect()",
    "Curve::IntersectP()",
    "Other Shape::Intersect()",
    "Other Shape::IntersectP()",
    "Material::ComputeScatteringFunctions()",
    "Camera::GenerateRay[Differential]()",
    "Film::MergeTile()",
    "Film::AddSplat()",
    "Film::AddSample()",
    "Sampler::StartPixelSample()",
    "Sampler::GetSample[12]D()",
    "MIPMap::Lookup() (trilinear)",
    "MIPMap::Lookup() (EWA)",
    "Ptex lookup",
    "Num prof categories",
];

impl Display for ProfileCategory {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let index = self.0 as usize;
        if index < PROF_NAMES.len() {
            return write!(f, "{}", PROF_NAMES[index]);
        } else {
            return write!(f, "Unknown");
        }
    }
}

fn prof_to_bits(p: ProfileCategory) -> u64 {
    return 1u64 << p.0 as u64;
}

impl ProfileCategory {
    pub fn to_bits(&self) -> u64 {
        return prof_to_bits(*self);
    }
}

pub type Prof = ProfileCategory;
