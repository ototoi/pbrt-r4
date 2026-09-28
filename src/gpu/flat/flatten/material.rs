use super::super::texture::TextureRootSpec;
use super::material_attributes::build_material_attributes;
use super::{
    push_scalar_attribute, push_spectrum_attribute, AttributeKind, AttributeRef, FlatBuilder,
    UnsupportedTexturePolicy,
};
use crate::gpu::flat::{MaterialNode, MaterialRoot, INVALID_INDEX};
use crate::gpu::node::{Material as NodeMaterial, TextureComponent, TextureKind, TextureNode};
use crate::util::error::PbrtError;
use crate::util::spectrum::{Spectrum, SpectrumType};
use std::sync::Arc;

#[derive(Clone, Debug)]
pub struct MaterialSourceNode {
    pub kind: String,
    pub source_kind: String,
    pub attributes: Vec<AttributeRef>,
    pub children: Vec<u32>,
    pub displacement_texture_root: u32,
}

pub fn build_material_roots(
    source_nodes: &[MaterialSourceNode],
    root_source_nodes: &[u32],
) -> Result<(Vec<MaterialRoot>, Vec<MaterialNode>, Vec<u32>), PbrtError> {
    fn append_node(
        source_node_index: u32,
        parent: u32,
        parent_slot: u32,
        source_nodes: &[MaterialSourceNode],
        visiting: &mut [bool],
        nodes: &mut Vec<MaterialNode>,
    ) -> Result<u32, PbrtError> {
        let source_index = usize::try_from(source_node_index)
            .map_err(|_| PbrtError::error("Flat material source index does not fit usize."))?;
        let source = source_nodes.get(source_index).ok_or_else(|| {
            PbrtError::error("Flat material child is outside the source-node table.")
        })?;
        if visiting[source_index] {
            return Err(PbrtError::error("Flat material graph contains a cycle."));
        }
        visiting[source_index] = true;

        let node_index = u32::try_from(nodes.len())
            .map_err(|_| PbrtError::error("Flat material tree exceeds u32."))?;
        nodes.push(MaterialNode {
            kind: source.kind.clone(),
            source_kind: source.source_kind.clone(),
            attributes: source.attributes.clone(),
            parent,
            parent_slot,
            child0: INVALID_INDEX,
            child1: INVALID_INDEX,
            displacement_texture_root: source.displacement_texture_root,
        });
        if source.children.len() > 2 {
            return Err(PbrtError::error(
                "Flat material has more than two material children.",
            ));
        }
        for (slot, child) in source.children.iter().enumerate() {
            let child_index = append_node(
                *child,
                node_index,
                slot as u32,
                source_nodes,
                visiting,
                nodes,
            )?;
            if slot == 0 {
                nodes[node_index as usize].child0 = child_index;
            } else {
                nodes[node_index as usize].child1 = child_index;
            }
        }
        visiting[source_index] = false;
        Ok(node_index)
    }

    let mut layouts = Vec::new();
    let mut nodes = Vec::new();
    let mut source_to_root = vec![INVALID_INDEX; source_nodes.len()];
    for &root in root_source_nodes {
        if root == INVALID_INDEX {
            continue;
        }
        let root_index = usize::try_from(root)
            .map_err(|_| PbrtError::error("Flat material root does not fit usize."))?;
        let mapped_root = source_to_root.get_mut(root_index).ok_or_else(|| {
            PbrtError::error("Flat instance references a material outside the source-node table.")
        })?;
        if *mapped_root != INVALID_INDEX {
            continue;
        }
        let layout_index = u32::try_from(layouts.len())
            .map_err(|_| PbrtError::error("Flat material layout table exceeds u32."))?;
        *mapped_root = layout_index;
        let node_offset = u32::try_from(nodes.len())
            .map_err(|_| PbrtError::error("Flat material tree table exceeds u32."))?;
        let mut tree_nodes = Vec::new();
        append_node(
            root,
            INVALID_INDEX,
            INVALID_INDEX,
            source_nodes,
            &mut vec![false; source_nodes.len()],
            &mut tree_nodes,
        )?;
        let node_count = u32::try_from(tree_nodes.len())
            .map_err(|_| PbrtError::error("Flat material tree exceeds u32."))?;
        for node in &mut tree_nodes {
            if node.parent != INVALID_INDEX {
                node.parent += node_offset;
            }
            if node.child0 != INVALID_INDEX {
                node.child0 += node_offset;
            }
            if node.child1 != INVALID_INDEX {
                node.child1 += node_offset;
            }
        }
        nodes.extend(tree_nodes);
        layouts.push(MaterialRoot {
            node_offset,
            node_count,
        });
    }
    Ok((layouts, nodes, source_to_root))
}

pub fn register_material_source(
    source_material: &Arc<NodeMaterial>,
    builder: &mut FlatBuilder,
    material_kind: Option<&str>,
) -> Result<u32, PbrtError> {
    if let Some(index) = builder
        .source_materials
        .iter()
        .position(|material| Arc::ptr_eq(material, source_material))
    {
        return u32::try_from(index).map_err(|_| {
            PbrtError::error(
                "The flattened material source-node table exceeds the u32 index range.",
            )
        });
    }
    let source_kind = source_material.kind.as_str();
    let requested_kind = if source_kind == "alphamask" {
        source_kind
    } else {
        material_kind.unwrap_or(source_kind)
    };
    let supported = matches!(
        requested_kind,
        "alphamask"
            | "diffuse"
            | "dielectric"
            | "thindielectric"
            | "conductor_eta_k"
            | "conductor_reflectance"
            | "diffusetransmission"
            | "mix"
            | "coateddiffuse"
            | "coatedconductor"
            | "measured"
    );
    let texture_fallback = if requested_kind != "diffusetransmission"
        && requested_kind != "alphamask"
        && supported
        && has_texture_attribute(source_material)
    {
        match UnsupportedTexturePolicy::from_environment()? {
            UnsupportedTexturePolicy::Error => false,
            UnsupportedTexturePolicy::DiagnosticMagenta => {
                log::warn!(
                    "GPU material \"{}\" contains unsupported textures; using diffuse reflectance (1, 0, 1).",
                    source_material.name
                );
                true
            }
        }
    } else {
        false
    };
    let (kind, mut attributes) = if texture_fallback {
        let magenta = Spectrum::from_rgb(&[1.0, 0.0, 1.0], SpectrumType::Albedo);
        (
            "diffuse",
            vec![push_spectrum_attribute(builder, "reflectance", &magenta)?],
        )
    } else if !supported {
        log::warn!(
            concat!(
                "GPU material \"{}\" of kind \"{}\" is unsupported; ",
                "using diffuse reflectance (1, 1, 0)."
            ),
            source_material.name,
            requested_kind,
        );
        ("diffuse", {
            let yellow = Spectrum::from_rgb(&[1.0, 1.0, 0.0], SpectrumType::Albedo);
            vec![push_spectrum_attribute(builder, "reflectance", &yellow)?]
        })
    } else {
        if requested_kind == "alphamask" {
            if source_material.material_attributes.len() != 1 {
                return Err(PbrtError::error(&format!(
                    "AlphaMask material \"{}\" must have exactly one child material.",
                    source_material.name
                )));
            }
            if source_material.texture_attributes.len() > 1
                || source_material
                    .texture_attributes
                    .iter()
                    .any(|(name, texture)| {
                        name != "alpha"
                            || texture.components.iter().any(|component| match component {
                                TextureComponent::Texture(texture) => {
                                    texture.kind != TextureKind::Float
                                }
                                TextureComponent::Mapping(_) => false,
                            })
                    })
            {
                return Err(PbrtError::error(&format!(
                    "AlphaMask material \"{}\" must have one float alpha texture at most.",
                    source_material.name
                )));
            }
            let attribute = if let Some((_, texture)) = source_material.texture_attributes.first() {
                AttributeRef {
                    kind: AttributeKind::Texture,
                    index: intern_texture_root(builder, Arc::clone(texture), 0)?,
                    name: "alpha".to_string(),
                }
            } else {
                let alpha = source_material.params.get_one_float("alpha", 1.0) as f32;
                if !alpha.is_finite() {
                    return Err(PbrtError::error(&format!(
                        "AlphaMask material \"{}\" has a non-finite alpha value.",
                        source_material.name
                    )));
                }
                push_scalar_attribute(builder, "alpha", alpha)?
            };
            (requested_kind, vec![attribute])
        } else if requested_kind == "measured" {
            let filename = source_material.params.get_one_string("filename", "");
            if filename.is_empty() {
                return Err(PbrtError::error(&format!(
                    "Material \"{}\" requires a measured BSDF filename.",
                    source_material.name
                )));
            }
            let index = builder
                .measured_bsdf_library
                .intern(std::path::Path::new(&filename))
                .map_err(|error| {
                    PbrtError::error(&format!(
                        "Unable to build GPU measured material \"{}\": {error}",
                        source_material.name
                    ))
                })?;
            (
                requested_kind,
                vec![AttributeRef {
                    kind: AttributeKind::Measured,
                    index,
                    name: "filename".to_string(),
                }],
            )
        } else {
            (
                requested_kind,
                build_material_attributes(source_material, requested_kind, builder)?,
            )
        }
    };
    let displacement_texture_root = if let Some((_, texture_node)) = source_material
        .texture_attributes
        .iter()
        .find(|(name, _)| name == "displacement")
    {
        let texture = texture_node
            .components
            .iter()
            .find_map(|component| match component {
                TextureComponent::Texture(texture) => Some(texture),
                TextureComponent::Mapping(_) => None,
            });
        let Some(texture) = texture else {
            return Err(PbrtError::error(&format!(
                "Material \"{}\" displacement has no texture component.",
                source_material.name
            )));
        };
        if texture.kind != TextureKind::Float {
            return Err(PbrtError::error(&format!(
                "Material \"{}\" displacement texture must be Float.",
                source_material.name
            )));
        }
        validate_bump_texture_filters(&source_material.name, texture_node)?;
        intern_texture_root(builder, Arc::clone(texture_node), 0)?
    } else {
        INVALID_INDEX
    };
    for (name, texture_node) in &source_material.texture_attributes {
        if name == "displacement" {
            continue;
        }
        if kind == "diffusetransmission" {
            if matches!(name.as_str(), "reflectance" | "Kd" | "transmittance" | "Kt") {
                continue;
            }
            return Err(PbrtError::error(&format!(
                "Material \"{}\" has unsupported diffusetransmission texture attribute \"{name}\".",
                source_material.name
            )));
        }
        let texture = texture_node
            .components
            .iter()
            .find_map(|component| match component {
                TextureComponent::Texture(texture) => Some(texture),
                TextureComponent::Mapping(_) => None,
            });
        if texture.is_some() && !attributes.iter().any(|attribute| attribute.name == *name) {
            let index = intern_texture_root(builder, texture_node.clone(), 0)?;
            attributes.push(AttributeRef {
                kind: AttributeKind::Texture,
                index,
                name: name.clone(),
            });
        }
    }
    let mut material_children = Vec::new();
    if matches!(kind, "coateddiffuse" | "coatedconductor") {
        let child_kinds: &[&str] = if kind == "coateddiffuse" {
            &["dielectric", "diffuse"]
        } else {
            &["dielectric", "conductor"]
        };
        for child_kind in child_kinds {
            let mut child_params = source_material.params.clone();
            // The synthetic substrate material must not inherit texture
            // references from the coating itself.  The coating stores its
            // textured albedo/eta/k as attributes on the wrapper; retaining
            // those parameter references would make the synthetic child try
            // to evaluate them a second time.
            for key in child_params.get_keys() {
                if child_params.get_key_type(&key) == "texture" {
                    child_params.remove_parameter(&key);
                }
            }
            let child = Arc::new(NodeMaterial {
                name: format!("{}:{}", source_material.name, child_kind),
                kind: if *child_kind == "conductor" {
                    if source_material
                        .params
                        .get_keys()
                        .iter()
                        .any(|key| source_material.params.get_key_name(key) == "reflectance")
                        || source_material
                            .texture_attributes
                            .iter()
                            .any(|(name, _)| name == "reflectance")
                    {
                        "conductor_reflectance".to_string()
                    } else {
                        "conductor_eta_k".to_string()
                    }
                } else {
                    (*child_kind).to_string()
                },
                params: child_params,
                material_attributes: Vec::new(),
                texture_attributes: Vec::new(),
            });
            material_children.push(register_material_source(&child, builder, None)?);
        }
    } else if kind == "mix" {
        for (_, child) in &source_material.material_attributes {
            material_children.push(register_material_source(child, builder, None)?);
        }
        if material_children.len() != 2 {
            return Err(PbrtError::error(
                "GPU mix material must contain exactly two material references.",
            ));
        }
    } else if kind == "alphamask" {
        let child = &source_material.material_attributes[0].1;
        material_children.push(register_material_source(child, builder, material_kind)?);
    }
    let index = u32::try_from(builder.material_source_nodes.len()).map_err(|_| {
        PbrtError::error("The flattened material source-node table exceeds the u32 index range.")
    })?;
    builder.material_source_nodes.push(MaterialSourceNode {
        kind: kind.to_string(),
        source_kind: source_kind.to_string(),
        attributes: attributes.clone(),
        children: material_children,
        displacement_texture_root,
    });
    builder.source_materials.push(Arc::clone(source_material));
    Ok(index)
}

fn validate_bump_texture_filters(
    material_name: &str,
    texture_node: &Arc<TextureNode>,
) -> Result<(), PbrtError> {
    for component in &texture_node.components {
        let TextureComponent::Texture(texture) = component else {
            continue;
        };
        if texture.name != "imagemap" {
            continue;
        }
        let filter = texture.params.get_one_string("filter", "bilinear");
        if !matches!(
            filter.as_str(),
            "point" | "nearest" | "bilinear" | "trilinear"
        ) {
            return Err(PbrtError::error(&format!(
                "Material \"{material_name}\" displacement texture \"{}\" uses unsupported image filter \"{filter}\".",
                texture_node.name
            )));
        }
    }
    for child in &texture_node.children {
        validate_bump_texture_filters(material_name, child)?;
    }
    Ok(())
}

pub fn diffuse_reflectance(source_material: &NodeMaterial) -> Result<Spectrum, PbrtError> {
    let default_reflectance = Spectrum::from(0.5);
    spectrum_attribute(
        source_material,
        "reflectance",
        &default_reflectance,
        SpectrumType::Albedo,
    )
}

pub fn texture_attribute_ref(
    source_material: &NodeMaterial,
    key: &str,
    builder: &mut FlatBuilder,
) -> Result<Option<AttributeRef>, PbrtError> {
    texture_attribute_ref_with_spectrum_type(source_material, key, builder, 0)
}

pub fn texture_attribute_ref_with_spectrum_type(
    source_material: &NodeMaterial,
    key: &str,
    builder: &mut FlatBuilder,
    spectrum_type: u32,
) -> Result<Option<AttributeRef>, PbrtError> {
    let Some((name, node)) = source_material
        .texture_attributes
        .iter()
        .find(|(name, _)| name == key)
    else {
        return Ok(None);
    };
    let texture = node
        .components
        .iter()
        .find_map(|component| match component {
            TextureComponent::Texture(texture) => Some(texture),
            TextureComponent::Mapping(_) => None,
        });
    if texture.is_none() {
        return Err(PbrtError::error(&format!(
            "Texture attribute \"{key}\" has no texture component."
        )));
    }
    let index = intern_texture_root(builder, node.clone(), spectrum_type)?;
    Ok(Some(AttributeRef {
        kind: AttributeKind::Texture,
        index,
        name: name.clone(),
    }))
}

pub fn texture_attribute_ref_unbounded(
    source_material: &NodeMaterial,
    key: &str,
    builder: &mut FlatBuilder,
) -> Result<Option<AttributeRef>, PbrtError> {
    texture_attribute_ref_with_spectrum_type(source_material, key, builder, 1)
}

pub fn intern_texture_root(
    builder: &mut FlatBuilder,
    texture_node: Arc<TextureNode>,
    spectrum_type: u32,
) -> Result<u32, PbrtError> {
    let texture = texture_node
        .components
        .iter()
        .find_map(|component| match component {
            TextureComponent::Texture(texture) => Some(texture),
            TextureComponent::Mapping(_) => None,
        })
        .ok_or_else(|| PbrtError::error("Texture root has no texture component."))?;
    let spectrum_type = match spectrum_type {
        1 => SpectrumType::Unbounded,
        2 => SpectrumType::Illuminant,
        _ => SpectrumType::Albedo,
    };
    let key = (
        Arc::as_ptr(&texture_node) as usize,
        match texture.kind {
            TextureKind::Float => 0,
            TextureKind::Spectrum => match spectrum_type {
                SpectrumType::Albedo => 1,
                SpectrumType::Unbounded => 2,
                SpectrumType::Illuminant => 3,
            },
        },
    );
    if let Some(&index) = builder.texture_roots_by_key.get(&key) {
        return Ok(index);
    }
    let index = u32::try_from(builder.texture_root_specs.len())
        .map_err(|_| PbrtError::error("Flat texture root table exceeds u32."))?;
    builder.texture_root_specs.push(match texture.kind {
        TextureKind::Float => TextureRootSpec::Float { node: texture_node },
        TextureKind::Spectrum => TextureRootSpec::Spectrum {
            node: texture_node,
            spectrum_type,
        },
    });
    builder.texture_roots_by_key.insert(key, index);
    Ok(index)
}

pub fn reject_scalar_textures(
    source_material: &NodeMaterial,
    keys: &[&str],
) -> Result<(), PbrtError> {
    if let Some(key) = source_material
        .params
        .get_keys()
        .iter()
        .find_map(|stored_key| {
            let is_texture = source_material.params.get_key_type(stored_key) == "texture";
            let name = source_material.params.get_key_name(stored_key);
            (is_texture && keys.iter().any(|key| *key == name)).then_some(name)
        })
    {
        return Err(PbrtError::error(&format!(
            "Material \"{}\" uses unsupported scalar texture attribute \"{key}\".",
            source_material.name
        )));
    }
    Ok(())
}

fn has_texture_attribute(source_material: &NodeMaterial) -> bool {
    source_material.params.get_keys().iter().any(|key| {
        source_material.params.get_key_type(key) == "texture"
            && source_material.params.get_key_name(key) != "displacement"
    })
}

pub fn spectrum_attribute(
    source_material: &NodeMaterial,
    key: &str,
    default: &Spectrum,
    spectrum_type: SpectrumType,
) -> Result<Spectrum, PbrtError> {
    let has_texture = source_material.params.get_keys().iter().any(|stored_key| {
        source_material.params.get_key_type(stored_key) == "texture"
            && source_material.params.get_key_name(stored_key) == key
    });
    if !has_texture {
        return Ok(source_material.params.get_one_spectrum(key, default));
    }
    match UnsupportedTexturePolicy::from_environment()? {
        UnsupportedTexturePolicy::Error => Err(PbrtError::error(&format!(
            "Material \"{}\" uses unsupported texture attribute \"{key}\".",
            source_material.name
        ))),
        UnsupportedTexturePolicy::DiagnosticMagenta => {
            log::warn!(
                "Material \"{}\" texture attribute \"{key}\" uses diagnostic magenta.",
                source_material.name
            );
            Ok(Spectrum::from_rgb(&[1.0, 0.0, 1.0], spectrum_type))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::paramdict::ParameterDictionary;

    #[test]
    fn displacement_float_texture_is_kept_as_a_separate_material_root() {
        let displacement = Arc::new(TextureNode {
            name: "height".to_string(),
            components: vec![TextureComponent::Texture(crate::gpu::node::Texture {
                name: "constant".to_string(),
                kind: TextureKind::Float,
                params: Default::default(),
            })],
            children: Vec::new(),
        });
        let material = Arc::new(NodeMaterial {
            name: "bumped".to_string(),
            kind: "diffuse".to_string(),
            params: Default::default(),
            material_attributes: Vec::new(),
            texture_attributes: vec![("displacement".to_string(), displacement)],
        });
        let mut builder = FlatBuilder::default();

        let source_index = register_material_source(&material, &mut builder, None).unwrap();
        let source = &builder.material_source_nodes[source_index as usize];
        assert_eq!(source.displacement_texture_root, 0);
        assert!(source
            .attributes
            .iter()
            .all(|attribute| attribute.name != "displacement"));

        let (_, flat_nodes, _) =
            build_material_roots(&builder.material_source_nodes, &[source_index]).unwrap();
        assert_eq!(flat_nodes[0].displacement_texture_root, 0);
        assert_eq!(builder.texture_root_specs.len(), 1);
        assert!(matches!(
            builder.texture_root_specs[0],
            TextureRootSpec::Float { .. }
        ));
    }

    #[test]
    fn displacement_texture_rejects_filters_without_gpu_footprint_support() {
        let mut texture_params = ParameterDictionary::new();
        texture_params.add_string("filter", "ewa");
        let displacement = Arc::new(TextureNode {
            name: "height-image".to_string(),
            components: vec![TextureComponent::Texture(crate::gpu::node::Texture {
                name: "imagemap".to_string(),
                kind: TextureKind::Float,
                params: texture_params,
            })],
            children: Vec::new(),
        });
        let material = Arc::new(NodeMaterial {
            name: "bumped-material".to_string(),
            kind: "diffuse".to_string(),
            params: Default::default(),
            material_attributes: Vec::new(),
            texture_attributes: vec![("displacement".to_string(), displacement)],
        });

        let error = register_material_source(&material, &mut FlatBuilder::default(), None)
            .unwrap_err()
            .to_string();

        assert!(error.contains("bumped-material"));
        assert!(error.contains("height-image"));
        assert!(error.contains("ewa"));
    }
}
