use super::{FlatBuilder, ImageViewKey};
use crate::gpu::flat::TextureNode as FlatTextureNode;
use crate::gpu::node::{
    ColorSpace, TextureComponent, TextureKind as NodeTextureKind, TextureMapping, TextureNode,
    Transform,
};
use crate::gpu::texture::{ImageFilterMode, ImageValueType, ImageView, ImageWrapMode, Mipmap};
use crate::util::error::PbrtError;
use crate::util::spectrum::Spectrum;
use std::collections::HashSet;
use std::sync::Arc;

pub fn register_texture_node(
    node: &Arc<TextureNode>,
    builder: &mut FlatBuilder,
) -> Result<u32, PbrtError> {
    validate_texture_graph(node, 0, &mut HashSet::new())?;
    let key = Arc::as_ptr(node) as usize;
    if let Some(&index) = builder.texture_nodes_by_ptr.get(&key) {
        return Ok(index);
    }
    let index = u32::try_from(builder.texture_nodes.len())
        .map_err(|_| PbrtError::error("Flat texture node table exceeds u32."))?;
    builder.texture_nodes_by_ptr.insert(key, index);
    builder.texture_nodes.push(FlatTextureNode {
        name: node.name.clone(),
        kind: 0,
        implementation: String::new(),
        first_child: 0,
        child_count: 0,
        image_view: None,
        mapping: Transform::default().matrix,
        color_space: 0,
        operation: 0,
        mapping_kind: 0,
        constant_value: [0.0; 4],
    });
    builder.texture_source_nodes.push(node.clone());
    let mut kind = 0;
    let mut implementation = String::new();
    let mut mipmap = None;
    let mut image_view = None;
    let mut mapping = Transform::default().matrix;
    let mut swrap_mode = 0;
    let mut twrap_mode = 0;
    let mut filter_mode = 0;
    let mut color_space = 0;
    let mut operation = 0;
    let mut mapping_kind = 0;
    let mut constant_value = [0.0; 4];
    for component in &node.components {
        if let TextureComponent::Texture(texture) = component {
            kind = match texture.kind {
                NodeTextureKind::Float => 0,
                NodeTextureKind::Spectrum => 1,
            };
            implementation = texture.name.clone();
            mipmap = texture.mipmap.clone();
            if texture.kind == NodeTextureKind::Float {
                if let Some(source) = mipmap.clone() {
                    mipmap = Some(intern_projected_float_mipmap(builder, &source)?);
                }
            }
            color_space = mipmap
                .as_ref()
                .map(|mipmap| match mipmap.color_space {
                    ColorSpace::Unknown => 0,
                    ColorSpace::Srgb => 0,
                    ColorSpace::Aces2065 => 1,
                    ColorSpace::DciP3 => 2,
                    ColorSpace::Rec2020 => 3,
                })
                .unwrap_or(0);
            let wrap = texture.params.get_one_string("wrap", "repeat");
            swrap_mode = sampler_wrap_mode(&texture.params.get_one_string("swrap", &wrap));
            twrap_mode = sampler_wrap_mode(&texture.params.get_one_string("twrap", &wrap));
            filter_mode =
                sampler_filter_mode(&texture.params.get_one_string("filter", "bilinear"))?;
            if texture.name == "constant" {
                operation = 1;
                if texture.kind == NodeTextureKind::Float {
                    constant_value[0] = texture.params.get_one_float("value", 0.0) as f32;
                    constant_value[1] = constant_value[0];
                    constant_value[2] = constant_value[0];
                } else {
                    let rgb = texture
                        .params
                        .get_one_spectrum("value", &Spectrum::from(0.0))
                        .to_rgb();
                    constant_value = [rgb[0] as f32, rgb[1] as f32, rgb[2] as f32, 0.0];
                }
                constant_value[3] =
                    (constant_value[0] + constant_value[1] + constant_value[2]) / 3.0;
            } else if texture.name == "imagemap" {
                // ImageTexture's scale/invert are applied after sampling.
                constant_value[0] = texture.params.get_one_float("scale", 1.0) as f32;
                constant_value[1] = if texture.params.get_one_bool("invert", false) {
                    1.0
                } else {
                    0.0
                };
            } else if texture.name == "scale" {
                operation = 2;
                let default_scale = texture.params.get_one_float("value", 1.0);
                constant_value[0] = texture.params.get_one_float("scale", default_scale) as f32;
            } else if texture.name == "mix" {
                operation = 3;
                constant_value[0] = texture.params.get_one_float("amount", 0.5) as f32;
            } else if texture.name == "directionmix" {
                operation = 5;
            } else if texture.name == "fbm" {
                operation = 7;
                constant_value[0] = texture.params.get_one_float("roughness", 0.5) as f32;
                constant_value[1] = texture.params.get_one_int("octaves", 8) as f32;
            } else if texture.name == "wrinkled" {
                operation = 8;
                constant_value[0] = texture.params.get_one_float("roughness", 0.5) as f32;
                constant_value[1] = texture.params.get_one_int("octaves", 8) as f32;
            } else if texture.name == "windy" {
                operation = 9;
            } else if texture.name == "dots" {
                operation = 6;
            } else if texture.name == "bilerp" {
                operation = 10;
            } else if texture.name == "marble" {
                operation = 11;
                constant_value[0] = texture.params.get_one_float("roughness", 0.5) as f32;
                constant_value[1] = texture.params.get_one_int("octaves", 8) as f32;
                constant_value[2] = texture.params.get_one_float("scale", 1.0) as f32;
                constant_value[3] = texture.params.get_one_float("variation", 0.2) as f32;
            } else if texture.name.contains("checkerboard") {
                operation = if texture.name == "checkerboard3d"
                    || texture.params.get_one_int("dimension", 2) == 3
                {
                    12
                } else {
                    4
                };
            }
            builder
                .texture_nodes_by_name
                .insert((texture.kind, node.name.clone()), index);
        }
        if let TextureComponent::Mapping(mapping_component) = component {
            if let TextureMapping::Uv(uv) = mapping_component {
                mapping[0] = uv.uscale;
                mapping[5] = uv.vscale;
                mapping[3] = uv.udelta;
                mapping[7] = uv.vdelta;
            } else if let TextureMapping::PointTransform(transform) = mapping_component {
                mapping = transform.matrix;
                if operation == 5 {
                    constant_value[0] = transform.matrix[0];
                    constant_value[1] = transform.matrix[1];
                    constant_value[2] = transform.matrix[2];
                }
            } else {
                match mapping_component {
                    TextureMapping::Planar(transform) => {
                        mapping = transform.matrix;
                        mapping_kind = 1;
                    }
                    TextureMapping::Spherical(transform) => {
                        mapping = transform.matrix;
                        mapping_kind = 2;
                    }
                    TextureMapping::Cylindrical(transform) => {
                        mapping = transform.matrix;
                        mapping_kind = 3;
                    }
                    _ => {}
                }
            }
        }
    }
    let mut child_indices = Vec::with_capacity(node.children.len());
    for child in &node.children {
        child_indices.push(register_texture_node(child, builder)?);
    }
    // Child registration is recursive and may append the descendants' child
    // ranges first.  Capture the parent's range only after that recursion so
    // `first_child` points at the indices appended below, not at a descendant
    // range.
    let first_child = u32::try_from(builder.texture_child_indices.len())
        .map_err(|_| PbrtError::error("Flat texture child table exceeds u32."))?;
    builder
        .texture_child_indices
        .extend(child_indices.iter().copied());
    let child_count = u32::try_from(node.children.len())
        .map_err(|_| PbrtError::error("Flat texture child table exceeds u32."))?;
    let valid_child_count = match operation {
        0 | 1 | 7 | 8 | 9 | 11 => child_count == 0,
        2 => child_count == 1 || child_count == 2,
        3 | 5 => child_count == 2 || (operation == 3 && child_count == 3),
        4 | 6 | 12 => child_count == 2,
        10 => child_count == 4,
        _ => false,
    };
    if !valid_child_count {
        return Err(PbrtError::error(&format!(
            "Texture node \"{}\" has {} children for operation {}.",
            node.name, child_count, operation
        )));
    }
    if kind == 1 && mipmap.is_none() {
        let mut graph_color_space = None;
        for child_index in &child_indices {
            let child = &builder.texture_nodes[*child_index as usize];
            if child.kind != 1 || (child.image_view.is_none() && child.color_space == 0) {
                continue;
            }
            if let Some(previous) = graph_color_space {
                if previous != child.color_space {
                    return Err(PbrtError::error(&format!(
                        "Spectrum texture node \"{}\" mixes incompatible color spaces.",
                        node.name
                    )));
                }
            } else {
                graph_color_space = Some(child.color_space);
            }
        }
        if let Some(graph_color_space) = graph_color_space {
            color_space = graph_color_space;
        }
    }
    if implementation == "imagemap" {
        if let Some(resource) = &mipmap {
            let wrap_mode = |mode| match mode {
                1 => ImageWrapMode::Clamp,
                2 => ImageWrapMode::Black,
                _ => ImageWrapMode::Repeat,
            };
            let filter = match filter_mode {
                0 => ImageFilterMode::Nearest,
                1 => ImageFilterMode::Bilinear,
                _ => ImageFilterMode::Trilinear,
            };
            let view = ImageView {
                mipmap: resource.clone(),
                value_type: if kind == 0 {
                    ImageValueType::Float
                } else {
                    ImageValueType::LinearRgb
                },
                swrap: wrap_mode(swrap_mode),
                twrap: wrap_mode(twrap_mode),
                filter,
                scale: constant_value[0],
                invert: constant_value[1] > 0.5,
            };
            image_view = Some(intern_image_view(builder, view)?);
            constant_value[0] = 0.0;
            constant_value[1] = 0.0;
        }
    }
    let flat = &mut builder.texture_nodes[index as usize];
    flat.kind = kind;
    flat.implementation = implementation;
    flat.first_child = first_child;
    flat.child_count = child_count;
    flat.image_view = image_view;
    flat.mapping = mapping;
    flat.color_space = color_space;
    flat.operation = operation;
    flat.mapping_kind = mapping_kind;
    flat.constant_value = constant_value;
    Ok(index)
}

pub fn intern_projected_float_mipmap(
    builder: &mut FlatBuilder,
    source: &Arc<Mipmap>,
) -> Result<Arc<Mipmap>, PbrtError> {
    builder
        .image_compiler
        .compile(source, ImageValueType::Float)
}

pub fn intern_image_view(builder: &mut FlatBuilder, view: ImageView) -> Result<u32, PbrtError> {
    let key = ImageViewKey {
        mipmap: Arc::as_ptr(&view.mipmap) as usize,
        value_type: view.value_type,
        swrap: view.swrap,
        twrap: view.twrap,
        filter: view.filter,
        scale: view.scale.to_bits(),
        invert: view.invert,
    };
    if let Some(&index) = builder.image_views_by_key.get(&key) {
        return Ok(index);
    }
    let index = u32::try_from(builder.image_views.len())
        .map_err(|_| PbrtError::error("Flat image view table exceeds u32."))?;
    builder.image_views.push(view);
    builder.image_views_by_key.insert(key, index);
    Ok(index)
}

const MAX_TEXTURE_GRAPH_DEPTH: usize = 32;

fn validate_texture_graph(
    node: &Arc<TextureNode>,
    depth: usize,
    visiting: &mut HashSet<usize>,
) -> Result<(), PbrtError> {
    if depth >= MAX_TEXTURE_GRAPH_DEPTH {
        return Err(PbrtError::error(&format!(
            "Texture graph exceeds the WebGPU evaluator depth limit of {}.",
            MAX_TEXTURE_GRAPH_DEPTH
        )));
    }
    let key = Arc::as_ptr(node) as usize;
    if !visiting.insert(key) {
        return Err(PbrtError::error("Texture graph contains a cycle."));
    }
    for child in &node.children {
        validate_texture_graph(child, depth + 1, visiting)?;
    }
    visiting.remove(&key);
    Ok(())
}

fn sampler_wrap_mode(mode: &str) -> u32 {
    match mode {
        "clamp" => 1,
        "black" => 2,
        _ => 0,
    }
}

fn sampler_filter_mode(mode: &str) -> Result<u32, PbrtError> {
    match mode {
        "point" | "nearest" => Ok(0),
        "bilinear" | "linear" => Ok(1),
        "trilinear" => Ok(2),
        "ewa" => Err(PbrtError::error(
            "GPU texture filter \"ewa\" is not supported yet.",
        )),
        other => Err(PbrtError::error(&format!(
            "Unknown GPU texture filter \"{other}\"."
        ))),
    }
}

#[cfg(test)]
mod texture_projection_tests {
    use super::super::material::intern_texture_root;
    use super::FlatBuilder;
    use super::{intern_image_view, intern_projected_float_mipmap};
    use crate::gpu::texture::project_float_mipmap;
    use crate::gpu::texture::{
        ColorSpace, ImageFilterMode, ImageValueType, ImageView, ImageWrapMode, Mipmap,
        MipmapEncoding, MipmapLevel, MipmapLevelData,
    };
    use std::sync::Arc;

    #[test]
    fn float_projection_normalizes_channels_and_encoding() {
        let source = Arc::new(Mipmap {
            levels: vec![MipmapLevel {
                resolution: [2, 1],
                channels: 3,
                data: MipmapLevelData::F32(vec![0.0, 0.3, 0.6, 0.5, 0.5, 0.5]),
            }],
            color_space: ColorSpace::Unknown,
            encoding: MipmapEncoding::Linear,
        });
        let projected = project_float_mipmap(&source).unwrap();
        assert_eq!(projected.encoding, MipmapEncoding::Linear);
        assert_eq!(projected.levels[0].channels, 1);
        assert_eq!(
            projected.levels[0].data,
            MipmapLevelData::F32(vec![0.3, 0.5])
        );
    }

    #[test]
    fn float_projection_decodes_srgb_before_extracting_alpha() {
        let source = Arc::new(Mipmap {
            levels: vec![MipmapLevel {
                resolution: [1, 1],
                channels: 4,
                data: MipmapLevelData::U8(vec![128, 64, 32, 200]),
            }],
            color_space: ColorSpace::Unknown,
            encoding: MipmapEncoding::SrgbEncoded,
        });
        let projected = project_float_mipmap(&source).unwrap();
        let MipmapLevelData::F32(values) = &projected.levels[0].data else {
            panic!("projection must produce F32 data");
        };
        assert!((values[0] - 200.0 / 255.0).abs() < 1e-6);
    }

    #[test]
    fn texture_roots_share_nodes_but_keep_spectrum_interpretation() {
        let mut builder = FlatBuilder::default();
        let albedo = intern_texture_root(&mut builder, 7, 0).unwrap();
        let same_albedo = intern_texture_root(&mut builder, 7, 0).unwrap();
        let unbounded = intern_texture_root(&mut builder, 7, 1).unwrap();

        assert_eq!(albedo, same_albedo);
        assert_ne!(albedo, unbounded);
        assert_eq!(builder.texture_roots.len(), 2);
        assert_eq!(builder.texture_roots[albedo as usize].texture_node, 7);
        assert_eq!(builder.texture_roots[unbounded as usize].spectrum_type, 1);
    }

    #[test]
    fn float_projection_is_shared_by_source_identity() {
        let source = Arc::new(Mipmap {
            levels: vec![MipmapLevel {
                resolution: [1, 1],
                channels: 3,
                data: MipmapLevelData::F32(vec![0.25, 0.5, 0.75]),
            }],
            color_space: ColorSpace::Unknown,
            encoding: MipmapEncoding::Linear,
        });
        let mut builder = FlatBuilder::default();
        let first = intern_projected_float_mipmap(&mut builder, &source).unwrap();
        let second = intern_projected_float_mipmap(&mut builder, &source).unwrap();

        assert!(Arc::ptr_eq(&first, &second));
    }

    #[test]
    fn image_views_share_only_identical_sampling_interpretations() {
        let mipmap = Arc::new(Mipmap {
            levels: vec![MipmapLevel {
                resolution: [1, 1],
                channels: 3,
                data: MipmapLevelData::F32(vec![0.25, 0.5, 0.75]),
            }],
            color_space: ColorSpace::Unknown,
            encoding: MipmapEncoding::Linear,
        });
        let view = ImageView {
            mipmap,
            value_type: ImageValueType::LinearRgb,
            swrap: ImageWrapMode::Repeat,
            twrap: ImageWrapMode::Clamp,
            filter: ImageFilterMode::Bilinear,
            scale: 1.0,
            invert: false,
        };
        let mut builder = FlatBuilder::default();
        let first = intern_image_view(&mut builder, view.clone()).unwrap();
        let same = intern_image_view(&mut builder, view.clone()).unwrap();
        let mut different_filter = view;
        different_filter.filter = ImageFilterMode::Trilinear;
        let different = intern_image_view(&mut builder, different_filter).unwrap();

        assert_eq!(first, same);
        assert_ne!(first, different);
        assert_eq!(builder.image_views.len(), 2);
        assert!(Arc::ptr_eq(
            &builder.image_views[first as usize].mipmap,
            &builder.image_views[different as usize].mipmap
        ));
    }
}
