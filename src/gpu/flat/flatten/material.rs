use super::super::texture::TextureRootSpec;
use super::material_attributes::build_material_attributes;
use super::{
    push_spectrum_attribute, AttributeKind, AttributeRef, FlatBuilder, UnsupportedTexturePolicy,
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
    let requested_kind = material_kind.unwrap_or(&source_material.kind);
    let source_kind = source_material.kind.as_str();
    let supported = matches!(
        requested_kind,
        "diffuse"
            | "dielectric"
            | "thindielectric"
            | "conductor"
            | "mix"
            | "coateddiffuse"
            | "coatedconductor"
    );
    let texture_fallback = if supported && has_texture_attribute(source_material) {
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
        (
            requested_kind,
            build_material_attributes(source_material, requested_kind, builder)?,
        )
    };
    for (name, texture_node) in &source_material.texture_attributes {
        // Displacement is consumed during CPU shape realization and is not a
        // material-evaluation attribute in the WebGPU backend.
        if name == "displacement" {
            continue;
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
                kind: (*child_kind).to_string(),
                params: child_params,
                material_attributes: Vec::new(),
                texture_attributes: Vec::new(),
            });
            material_children.push(register_material_source(&child, builder, Some(child_kind))?);
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
    }
    let index = u32::try_from(builder.material_source_nodes.len()).map_err(|_| {
        PbrtError::error("The flattened material source-node table exceeds the u32 index range.")
    })?;
    builder.material_source_nodes.push(MaterialSourceNode {
        kind: kind.to_string(),
        source_kind: source_kind.to_string(),
        attributes: attributes.clone(),
        children: material_children,
    });
    builder.source_materials.push(Arc::clone(source_material));
    Ok(index)
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
