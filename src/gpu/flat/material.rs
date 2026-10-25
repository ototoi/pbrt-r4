use super::{
    AttributeKind, AttributeRef, MeasuredBsdfResources, Scene, TabulatedBSSRDFTable, BSSRDF,
    INVALID_INDEX,
};
use crate::util::error::PbrtError;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MaterialRoot {
    pub node_offset: u32,
    pub node_count: u32,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MaterialKind {
    Diffuse,
    Dielectric,
    ThinDielectric,
    DiffuseTransmission,
    ConductorEtaK,
    ConductorReflectance,
    Mix,
    CoatedDiffuse,
    CoatedConductor,
    Measured,
    Subsurface,
    Hair,
}

impl MaterialKind {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "diffuse" => Some(Self::Diffuse),
            "dielectric" => Some(Self::Dielectric),
            "thindielectric" => Some(Self::ThinDielectric),
            "diffusetransmission" => Some(Self::DiffuseTransmission),
            "conductor_eta_k" => Some(Self::ConductorEtaK),
            "conductor_reflectance" => Some(Self::ConductorReflectance),
            "mix" => Some(Self::Mix),
            "coateddiffuse" => Some(Self::CoatedDiffuse),
            "coatedconductor" => Some(Self::CoatedConductor),
            "measured" => Some(Self::Measured),
            "subsurface" => Some(Self::Subsurface),
            "hair" => Some(Self::Hair),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Diffuse => "diffuse",
            Self::Dielectric => "dielectric",
            Self::ThinDielectric => "thindielectric",
            Self::DiffuseTransmission => "diffusetransmission",
            Self::ConductorEtaK => "conductor_eta_k",
            Self::ConductorReflectance => "conductor_reflectance",
            Self::Mix => "mix",
            Self::CoatedDiffuse => "coateddiffuse",
            Self::CoatedConductor => "coatedconductor",
            Self::Measured => "measured",
            Self::Subsurface => "subsurface",
            Self::Hair => "hair",
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct MaterialNode {
    pub kind: MaterialKind,
    pub source_kind: String,
    pub attributes: Vec<AttributeRef>,
    pub parent: u32,
    pub parent_slot: u32,
    pub child0: u32,
    pub child1: u32,
    pub displacement_texture_root: u32,
    pub bssrdf_index: u32,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MaterialResources {
    pub roots: Vec<MaterialRoot>,
    pub nodes: Vec<MaterialNode>,
    pub bssrdfs: Vec<BSSRDF>,
    pub bssrdf_tables: Vec<TabulatedBSSRDFTable>,
    pub measured_bsdfs: MeasuredBsdfResources,
}

impl MaterialResources {
    pub fn validate(&self) -> Result<(), PbrtError> {
        for node in &self.nodes {
            if node.kind == MaterialKind::Subsurface {
                if node.bssrdf_index as usize >= self.bssrdfs.len() {
                    return Err(PbrtError::error(
                        "Subsurface material references an invalid BSSRDF.",
                    ));
                }
            } else if node.bssrdf_index != INVALID_INDEX {
                return Err(PbrtError::error(
                    "Only subsurface materials may reference a BSSRDF.",
                ));
            }
            validate_material_attributes(node)?;
        }
        if self
            .roots
            .iter()
            .any(|root| root.node_offset as usize >= self.nodes.len())
        {
            return Err(PbrtError::error("Flat material root node is missing."));
        }
        Ok(())
    }
}

fn expected_material_attributes(kind: MaterialKind) -> &'static [AttributeKind] {
    use AttributeKind::{Measured, Scalar, Spectrum};
    match kind {
        MaterialKind::Diffuse => &[Spectrum],
        MaterialKind::Dielectric => &[Spectrum, Scalar, Scalar, Scalar],
        MaterialKind::Subsurface => &[Spectrum, Scalar, Scalar, Scalar, Spectrum, Spectrum],
        MaterialKind::ThinDielectric => &[Spectrum],
        MaterialKind::DiffuseTransmission => &[Spectrum, Spectrum, Scalar],
        MaterialKind::ConductorEtaK => &[Spectrum, Spectrum, Scalar, Scalar],
        MaterialKind::ConductorReflectance => &[Spectrum, Scalar, Scalar],
        MaterialKind::Mix => &[Scalar],
        MaterialKind::CoatedDiffuse => &[
            Scalar, Spectrum, Scalar, Scalar, Scalar, Spectrum, Spectrum, Scalar, Scalar, Scalar,
        ],
        MaterialKind::CoatedConductor => &[
            Scalar, Spectrum, Scalar, Scalar, Scalar, Spectrum, Scalar, Scalar, Spectrum, Spectrum,
            Scalar, Scalar, Scalar, Spectrum, Scalar,
        ],
        MaterialKind::Measured => &[Measured],
        // [beta_m, beta_n, alpha, eta, sigma_a, reflectance, eumelanin,
        // pheomelanin, sigma_a_mode] in `docs/webgpu-hair-material-design_ja.md`
        // §2.2; the WGSL evaluator reads the same order.
        MaterialKind::Hair => &[
            Scalar, Scalar, Scalar, Scalar, Spectrum, Spectrum, Scalar, Scalar, Scalar,
        ],
    }
}

fn validate_material_attributes(node: &MaterialNode) -> Result<(), PbrtError> {
    let expected = expected_material_attributes(node.kind);
    let matches = node.attributes.len() == expected.len()
        && node
            .attributes
            .iter()
            .zip(expected)
            .all(|(actual, &expected)| {
                actual.kind == expected
                    || (actual.kind == AttributeKind::Texture
                        && matches!(expected, AttributeKind::Scalar | AttributeKind::Spectrum))
            });
    if matches {
        return Ok(());
    }
    let actual = node
        .attributes
        .iter()
        .map(|attribute| {
            format!(
                "{}:{}",
                attribute.name,
                format_attribute_kind(attribute.kind)
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    Err(PbrtError::error(&format!(
        "Material \"{}\" kind \"{}\" has invalid attributes: [{actual}].",
        node.source_kind,
        node.kind.name()
    )))
}

fn format_attribute_kind(kind: AttributeKind) -> &'static str {
    match kind {
        AttributeKind::Scalar => "scalar",
        AttributeKind::Spectrum => "spectrum",
        AttributeKind::Texture => "texture",
        AttributeKind::Measured => "measured",
    }
}

pub fn max_attributes_eval_work_items_per_surface(scene: &Scene) -> Result<u32, PbrtError> {
    scene.instances.iter().try_fold(0, |maximum, instance| {
        if instance.material_root == INVALID_INDEX {
            return Ok(maximum);
        }
        let layout = scene
            .materials
            .roots
            .get(instance.material_root as usize)
            .ok_or_else(|| PbrtError::error("Instance material tree layout is missing."))?;
        Ok(maximum.max(layout.node_count))
    })
}

pub fn max_texture_eval_results_per_surface(scene: &Scene) -> Result<u32, PbrtError> {
    scene.instances.iter().try_fold(0, |maximum, instance| {
        if instance.material_root == INVALID_INDEX {
            return Ok(maximum);
        }
        let layout = scene
            .materials
            .roots
            .get(instance.material_root as usize)
            .ok_or_else(|| PbrtError::error("Instance material tree layout is missing."))?;
        let start = layout.node_offset as usize;
        let end = start
            .checked_add(layout.node_count as usize)
            .ok_or_else(|| PbrtError::error("Material tree layout range overflowed."))?;
        let count = scene
            .materials
            .nodes
            .get(start..end)
            .ok_or_else(|| PbrtError::error("Material tree layout is outside the node table."))?
            .iter()
            .map(|node| {
                node.attributes
                    .iter()
                    .filter(|a| a.kind == AttributeKind::Texture)
                    .count() as u32
            })
            .try_fold(0u32, |sum, count| {
                sum.checked_add(count)
                    .ok_or_else(|| PbrtError::error("Flat texture result count overflowed."))
            })?;
        Ok(maximum.max(count))
    })
}

pub const EVENT_REFLECTION: u32 = 1 << 0;
pub const EVENT_TRANSMISSION: u32 = 1 << 1;
pub const EVENT_DIFFUSE: u32 = 1 << 2;
pub const EVENT_GLOSSY: u32 = 1 << 3;
pub const EVENT_SPECULAR: u32 = 1 << 4;
