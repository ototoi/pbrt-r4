pub mod common;
pub mod debug_target;
pub mod parse_target;
pub mod print_target;
pub mod read_file;
pub mod remove_comment;
pub mod scene_builder;
pub mod to_ply_target;
pub mod upgrade;

pub use debug_target::DebugTarget;
pub use parse_target::ParseTarget;
pub use print_target::PrintTarget;
pub use scene_builder::SceneBuilder;
pub use to_ply_target::ToPlyTarget;

use self::common::*;
use self::read_file::{read_file_with_include, read_file_without_include};
use self::remove_comment::remove_comment;
use crate::paramdict::ParameterDictionary;
use crate::util::base::Float;
use crate::util::error::*;

use nom::bytes;
use nom::character;
use nom::multi;
use nom::number;
use nom::sequence;
use nom::IResult;

fn search_pbrt_file(dir: &std::path::Path) -> Option<std::path::PathBuf> {
    let entries: Vec<std::fs::DirEntry> = match std::fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(_) => return None,
    }
    .filter_map(|f| f.ok())
    .collect();
    for entry in entries {
        let path = entry.path();
        if path.is_file() {
            let Some(filename) = path.file_name() else {
                continue;
            };
            if filename.to_string_lossy().ends_with(".pbrt") {
                return Some(path);
            }
        }
    }
    return None;
}

fn parse_targz(filename: &str, context: &mut dyn ParseTarget) -> Result<(), PbrtError> {
    let tmp_dir = tempfile::tempdir()?;
    let tmp_dir_path = tmp_dir.path();

    let tar_gz = std::fs::File::open(filename)?;
    let tar = flate2::read::GzDecoder::new(tar_gz);
    let mut archive = tar::Archive::new(tar);
    archive.unpack(tmp_dir_path)?;

    let entries: Vec<std::fs::DirEntry> = std::fs::read_dir(tmp_dir_path)
        .map_err(|e| PbrtError::from(format!("failed to inspect extracted archive: {}", e)))?
        .filter_map(|f| f.ok())
        .collect();
    for entry in entries {
        let path = entry.path();
        if let Some(path) = search_pbrt_file(&path) {
            let path = path.to_string_lossy();
            let s = read_file_with_include(&path)?;
            return parse_string_core(&s, context);
        }
    }
    return Err(PbrtError::from(std::io::Error::from(
        std::io::ErrorKind::NotFound,
    )));
}

pub fn parse_file(filename: &str, context: &mut dyn ParseTarget) -> Result<(), PbrtError> {
    if filename.ends_with(".tar.gz") {
        return parse_targz(filename, context);
    } else {
        let s = read_file_with_include(filename)?;
        return parse_string_core(&s, context);
    }
}

pub fn parse_string(s: &str, context: &mut dyn ParseTarget) -> Result<(), PbrtError> {
    let ops = parse_opnodes(s)?;
    return evaluate_opnodes(&ops, context);
}

/// Parses a legacy scene and evaluates its v4-compatible upgraded operations.
pub fn parse_string_upgraded(s: &str, context: &mut dyn ParseTarget) -> Result<(), PbrtError> {
    let mut ops = parse_opnodes(s)?;
    upgrade::upgrade_opnodes(&mut ops)?;
    evaluate_opnodes(&ops, context)
}

pub fn parse_file_without_include(
    filename: &str,
    context: &mut dyn ParseTarget,
) -> Result<(), PbrtError> {
    let s = read_file_without_include(filename)?;
    return parse_string_core(&s, context);
}

/// Reads, upgrades, and evaluates a scene file. Includes are expanded before
/// the upgrade pass so that material and film directives are handled uniformly.
pub fn parse_file_upgraded(filename: &str, context: &mut dyn ParseTarget) -> Result<(), PbrtError> {
    let s = read_file_with_include(filename)?;
    parse_string_upgraded(&s, context)
}
//-----------------------------------

pub fn parse_string_core(s: &str, context: &mut dyn ParseTarget) -> Result<(), PbrtError> {
    let ops = parse_opnodes_core(s)?;
    return evaluate_opnodes(&ops, context);
}

fn parse_opnodes(s: &str) -> Result<Vec<OPNode>, PbrtError> {
    let r = remove_comment(s);
    match r {
        Ok((_, s)) => {
            return parse_opnodes_core(&s);
        }
        Err(e) => {
            return Err(PbrtError::from(e.to_string()));
        }
    }
}

fn parse_opnodes_core(s: &str) -> Result<Vec<OPNode>, PbrtError> {
    let result = nom::combinator::all_consuming(multi::many0(sequence::delimited(
        space0,
        parse_operation,
        space0,
    )))(s);
    match result {
        Ok((_, nodes)) => {
            return Ok(nodes);
        }
        Err(e) => {
            return Err(PbrtError::from(e.to_string()));
        }
    }
}

pub struct OPNode {
    pub name: String,
    pub args: Option<ParameterDictionary>,
    pub params: Option<ParameterDictionary>,
}

impl OPNode {
    pub fn new(
        name: &str,
        args: Option<ParameterDictionary>,
        params: Option<ParameterDictionary>,
    ) -> Self {
        OPNode {
            name: String::from(name),
            args,
            params,
        }
    }
}

fn evaluate_opnodes(ops: &[OPNode], context: &mut dyn ParseTarget) -> Result<(), PbrtError> {
    for op in ops {
        let opname: &str = &op.name;
        match opname {
            "Identity" => {
                //fn identity(&mut self);
                context.identity();
            }
            "Translate" => {
                //fn translate(&mut self, dx: Float, dy: Float, dz: Float);
                let Some(args) = op.args.as_ref() else {
                    return Err(PbrtError::error("Translate requires arguments."));
                };
                let vec = args.get_floats("args");
                if vec.len() != 3 {
                    let msg = format!("{} required {} arguments", opname, 3);
                    return Err(PbrtError::error(&msg));
                }
                context.translate(vec[0], vec[1], vec[2]);
            }
            "Rotate" => {
                //fn rotate(&mut self, angle: Float, ax: Float, ay: Float, az: Float);
                let Some(args) = op.args.as_ref() else {
                    return Err(PbrtError::error("Rotate requires arguments."));
                };
                let vec = args.get_floats("args");
                if vec.len() != 4 {
                    let msg = format!("{} required {} arguments", opname, 4);
                    return Err(PbrtError::error(&msg));
                }
                context.rotate(vec[0], vec[1], vec[2], vec[3]);
            }
            "Scale" => {
                let Some(args) = op.args.as_ref() else {
                    return Err(PbrtError::error("Scale requires arguments."));
                };
                let vec = args.get_floats("args");
                if vec.len() != 3 {
                    let msg = format!("{} required {} arguments", opname, 3);
                    return Err(PbrtError::error(&msg));
                }
                context.scale(vec[0], vec[1], vec[2]);
            }
            "LookAt" => {
                let Some(args) = op.args.as_ref() else {
                    return Err(PbrtError::error("LookAt requires arguments."));
                };
                let vec = args.get_floats("args");
                if vec.len() != 9 {
                    let msg = format!("{} required {} arguments", opname, 9);
                    return Err(PbrtError::error(&msg));
                }
                context.look_at(
                    vec[0], vec[1], vec[2], vec[3], vec[4], vec[5], vec[6], vec[7], vec[8],
                );
            }
            "ConcatTransform" => {
                let Some(args) = op.args.as_ref() else {
                    return Err(PbrtError::error("ConcatTransform requires arguments."));
                };
                let vec = args.get_floats("arg1");
                if vec.len() != 16 {
                    let msg = format!("{} required {} arguments", opname, 16);
                    return Err(PbrtError::error(&msg));
                }
                context.concat_transform(&vec);
            }
            "Transform" => {
                let Some(args) = op.args.as_ref() else {
                    return Err(PbrtError::error("Transform requires arguments."));
                };
                let vec = args.get_floats("arg1");
                if vec.len() != 16 {
                    let msg = format!("{} required {} arguments", opname, 16);
                    return Err(PbrtError::error(&msg));
                }
                context.transform(&vec);
            }
            "CoordinateSystem" => {
                let Some(args) = op.args.as_ref() else {
                    return Err(PbrtError::error("CoordinateSystem requires arguments."));
                };
                let vec = args.get_strings("arg1");
                let Some(name) = vec.first() else {
                    return Err(PbrtError::error("CoordinateSystem requires a name."));
                };
                context.coordinate_system(name);
            }
            "CoordSysTransform" => {
                let Some(args) = op.args.as_ref() else {
                    return Err(PbrtError::error("CoordSysTransform requires arguments."));
                };
                let vec = args.get_strings("arg1");
                let Some(name) = vec.first() else {
                    return Err(PbrtError::error("CoordSysTransform requires a name."));
                };
                context.coord_sys_transform(name);
            }
            "ColorSpace" => {
                let Some(args) = op.args.as_ref() else {
                    return Err(PbrtError::error("ColorSpace requires arguments."));
                };
                let vec = args.get_strings("arg1");
                let Some(name) = vec.first() else {
                    return Err(PbrtError::error("ColorSpace requires a name."));
                };
                context.color_space(name);
            }
            "Option" => {
                let Some(args) = op.args.as_ref() else {
                    return Err(PbrtError::error("Option requires arguments."));
                };
                let name_vec = args.get_strings("arg1");
                let value_vec = args.get_strings("arg2");
                let Some(name) = name_vec.first() else {
                    return Err(PbrtError::error("Option requires a name."));
                };
                let Some(value) = value_vec.first() else {
                    return Err(PbrtError::error("Option requires a value."));
                };
                context.option(name, value);
            }
            "ActiveTransformAll" => {
                context.active_transform_all();
            }
            "ActiveTransformEndTime" => {
                context.active_transform_end_time();
            }
            "ActiveTransformStartTime" => {
                context.active_transform_start_time();
            }
            "PixelFilter" => {
                let Some(args) = op.args.as_ref() else {
                    return Err(PbrtError::error("PixelFilter requires arguments."));
                };
                let vec = args.get_strings("arg1");
                let Some(name) = vec.first() else {
                    return Err(PbrtError::error("PixelFilter requires a name."));
                };
                let Some(params) = op.params.as_ref() else {
                    return Err(PbrtError::error("PixelFilter requires parameters."));
                };
                context.pixel_filter(name, params);
            }
            "Film" => {
                let Some(args) = op.args.as_ref() else {
                    return Err(PbrtError::error("Film requires arguments."));
                };
                let vec = args.get_strings("arg1");
                let Some(name) = vec.first() else {
                    return Err(PbrtError::error("Film requires a name."));
                };
                let Some(params) = op.params.as_ref() else {
                    return Err(PbrtError::error("Film requires parameters."));
                };
                context.film(name, params);
            }
            "Sampler" => {
                let Some(args) = op.args.as_ref() else {
                    return Err(PbrtError::error("Sampler requires arguments."));
                };
                let vec = args.get_strings("arg1");
                let Some(name) = vec.first() else {
                    return Err(PbrtError::error("Sampler requires a name."));
                };
                let Some(params) = op.params.as_ref() else {
                    return Err(PbrtError::error("Sampler requires parameters."));
                };
                context.sampler(name, params);
            }
            "Accelerator" => {
                let Some(args) = op.args.as_ref() else {
                    return Err(PbrtError::error("Accelerator requires arguments."));
                };
                let vec = args.get_strings("arg1");
                let Some(name) = vec.first() else {
                    return Err(PbrtError::error("Accelerator requires a name."));
                };
                let Some(params) = op.params.as_ref() else {
                    return Err(PbrtError::error("Accelerator requires parameters."));
                };
                context.accelerator(name, params);
            }
            "Integrator" => {
                let Some(args) = op.args.as_ref() else {
                    return Err(PbrtError::error("Integrator requires arguments."));
                };
                let vec = args.get_strings("arg1");
                let Some(name) = vec.first() else {
                    return Err(PbrtError::error("Integrator requires a name."));
                };
                let Some(params) = op.params.as_ref() else {
                    return Err(PbrtError::error("Integrator requires parameters."));
                };
                context.integrator(name, params);
            }
            "Camera" => {
                let Some(args) = op.args.as_ref() else {
                    return Err(PbrtError::error("Camera requires arguments."));
                };
                let vec = args.get_strings("arg1");
                let Some(name) = vec.first() else {
                    return Err(PbrtError::error("Camera requires a name."));
                };
                let Some(params) = op.params.as_ref() else {
                    return Err(PbrtError::error("Camera requires parameters."));
                };
                context.camera(name, params);
            }
            "MakeNamedMedium" => {
                let Some(args) = op.args.as_ref() else {
                    return Err(PbrtError::error("MakeNamedMedium requires arguments."));
                };
                let vec = args.get_strings("arg1");
                let Some(name) = vec.first() else {
                    return Err(PbrtError::error("MakeNamedMedium requires a name."));
                };
                let Some(params) = op.params.as_ref() else {
                    return Err(PbrtError::error("MakeNamedMedium requires parameters."));
                };
                context.make_named_medium(name, params);
            }
            "MediumInterface" => {
                let Some(args) = op.args.as_ref() else {
                    return Err(PbrtError::error("MediumInterface requires arguments."));
                };
                let vec1 = args.get_strings("arg1");
                let vec2 = args.get_strings("arg2");
                let Some(inside_name) = vec1.first() else {
                    return Err(PbrtError::error(
                        "MediumInterface requires an inside medium name.",
                    ));
                };
                let Some(outside_name) = vec2.first() else {
                    return Err(PbrtError::error(
                        "MediumInterface requires an outside medium name.",
                    ));
                };
                context.medium_interface(inside_name, outside_name);
            }
            "WorldBegin" => {
                context.world_begin();
            }
            "Attribute" => {
                let Some(args) = op.args.as_ref() else {
                    return Err(PbrtError::error("Attribute requires arguments."));
                };
                let vec = args.get_strings("arg1");
                let Some(target) = vec.first() else {
                    return Err(PbrtError::error("Attribute requires a target."));
                };
                let Some(params) = op.params.as_ref() else {
                    return Err(PbrtError::error("Attribute requires parameters."));
                };
                context.attribute(target, params);
            }
            "AttributeBegin" => {
                context.attribute_begin();
            }
            "AttributeEnd" => {
                context.attribute_end();
            }
            "TransformBegin" => {
                context.transform_begin();
            }
            "TransformEnd" => {
                context.transform_end();
            }
            "Texture" => {
                let Some(args) = op.args.as_ref() else {
                    return Err(PbrtError::error("Texture requires arguments."));
                };
                let name_values = args.get_strings("arg1");
                let Some(name) = name_values.first() else {
                    return Err(PbrtError::error("Texture requires a name."));
                };
                let tp_values = args.get_strings("arg2");
                let Some(tp) = tp_values.first() else {
                    return Err(PbrtError::error("Texture requires a type."));
                };
                let tex_name_values = args.get_strings("arg3");
                let Some(tex_name) = tex_name_values.first() else {
                    return Err(PbrtError::error("Texture requires a texture target."));
                };
                let Some(params) = op.params.as_ref() else {
                    return Err(PbrtError::error("Texture requires parameters."));
                };
                context.texture(&name, &tp, &tex_name, params);
            }
            "Material" => {
                let Some(args) = op.args.as_ref() else {
                    return Err(PbrtError::error("Material requires arguments."));
                };
                let vec = args.get_strings("arg1");
                let Some(name) = vec.first() else {
                    return Err(PbrtError::error("Material requires a name."));
                };
                let Some(params) = op.params.as_ref() else {
                    return Err(PbrtError::error("Material requires parameters."));
                };
                context.material(name, params);
            }
            "MakeNamedMaterial" => {
                let Some(args) = op.args.as_ref() else {
                    return Err(PbrtError::error("MakeNamedMaterial requires arguments."));
                };
                let vec = args.get_strings("arg1");
                let Some(name) = vec.first() else {
                    return Err(PbrtError::error("MakeNamedMaterial requires a name."));
                };
                let Some(params) = op.params.as_ref() else {
                    return Err(PbrtError::error("MakeNamedMaterial requires parameters."));
                };
                context.make_named_material(name, params);
            }
            "NamedMaterial" => {
                let Some(args) = op.args.as_ref() else {
                    return Err(PbrtError::error("NamedMaterial requires arguments."));
                };
                let vec = args.get_strings("arg1");
                let Some(name) = vec.first() else {
                    return Err(PbrtError::error("NamedMaterial requires a name."));
                };
                context.named_material(name);
            }
            "LightSource" => {
                let Some(args) = op.args.as_ref() else {
                    return Err(PbrtError::error("LightSource requires arguments."));
                };
                let vec = args.get_strings("arg1");
                let Some(name) = vec.first() else {
                    return Err(PbrtError::error("LightSource requires a name."));
                };
                let Some(params) = op.params.as_ref() else {
                    return Err(PbrtError::error("LightSource requires parameters."));
                };
                context.light_source(name, params);
            }
            "AreaLightSource" => {
                let Some(args) = op.args.as_ref() else {
                    return Err(PbrtError::error("AreaLightSource requires arguments."));
                };
                let vec = args.get_strings("arg1");
                let Some(name) = vec.first() else {
                    return Err(PbrtError::error("AreaLightSource requires a name."));
                };
                let Some(params) = op.params.as_ref() else {
                    return Err(PbrtError::error("AreaLightSource requires parameters."));
                };
                context.area_light_source(name, params);
            }
            "Shape" => {
                let Some(args) = op.args.as_ref() else {
                    return Err(PbrtError::error("Shape requires arguments."));
                };
                let vec = args.get_strings("arg1");
                let Some(name) = vec.first() else {
                    return Err(PbrtError::error("Shape requires a name."));
                };
                let Some(params) = op.params.as_ref() else {
                    return Err(PbrtError::error("Shape requires parameters."));
                };
                context.shape(name, params);
            }
            "ReverseOrientation" => {
                context.reverse_orientation();
            }
            "ObjectBegin" => {
                let Some(args) = op.args.as_ref() else {
                    return Err(PbrtError::error("ObjectBegin requires arguments."));
                };
                let vec = args.get_strings("arg1");
                let Some(name) = vec.first() else {
                    return Err(PbrtError::error("ObjectBegin requires a name."));
                };
                context.object_begin(name);
            }
            "ObjectEnd" => {
                context.object_end();
            }
            "ObjectInstance" => {
                let Some(args) = op.args.as_ref() else {
                    return Err(PbrtError::error("ObjectInstance requires arguments."));
                };
                let vec = args.get_strings("arg1");
                let Some(name) = vec.first() else {
                    return Err(PbrtError::error("ObjectInstance requires a name."));
                };
                context.object_instance(name);
            }
            "WorldEnd" => {
                context.world_end();
            }
            "WorkDirBegin" => {
                let Some(args) = op.args.as_ref() else {
                    return Err(PbrtError::error("WorkDirBegin requires arguments."));
                };
                let vec = args.get_strings("arg1");
                let Some(path) = vec.first() else {
                    return Err(PbrtError::error("WorkDirBegin requires a path."));
                };
                context.work_dir_begin(path);
            }
            "WorkDirEnd" => {
                context.work_dir_end();
            }
            "Include" => {
                let Some(args) = op.args.as_ref() else {
                    return Err(PbrtError::error("Include requires arguments."));
                };
                let vec = args.get_strings("arg1");
                let Some(filename) = vec.first() else {
                    return Err(PbrtError::error("Include requires a filename."));
                };
                let Some(params) = op.params.as_ref() else {
                    return Err(PbrtError::error("Include requires parameters."));
                };
                context.include(filename, params);
            }
            "Import" => {
                let Some(args) = op.args.as_ref() else {
                    return Err(PbrtError::error("Import requires arguments."));
                };
                let vec = args.get_strings("arg1");
                let Some(filename) = vec.first() else {
                    return Err(PbrtError::error("Import requires a filename."));
                };
                let Some(params) = op.params.as_ref() else {
                    return Err(PbrtError::error("Import requires parameters."));
                };
                context.import(filename, params);
            }
            _ => {
                let msg = format!("Unexpected token: {}", opname);
                return Err(PbrtError::error(&msg));
            }
        }
    }
    return Ok(());
}

//-----------------------------------

fn parse_operation(s: &str) -> IResult<&str, OPNode> {
    return nom::branch::alt((
        nom::branch::alt((
            parse_identity,
            parse_translate,
            parse_rotate,
            parse_scale,
            parse_look_at,
            parse_concat_transform,
            parse_transform,
            parse_coordinate_system,
            parse_coord_sys_transform,
            parse_color_space,
            parse_active_transform,
            parse_transform_times,
        )),
        nom::branch::alt((
            parse_pixel_filter,
            parse_film,
            parse_sampler,
            parse_accelerator,
            parse_integrator,
            parse_camera,
            parse_make_named_medium,
            parse_medium_interface,
            parse_option,
        )),
        nom::branch::alt((
            parse_world_begin,
            parse_attribute,
            parse_attribute_begin,
            parse_attribute_end,
            parse_transform_begin,
            parse_transform_end,
            parse_texture,
            parse_material,
            parse_make_named_material,
            parse_named_material,
            parse_light_source,
            parse_area_light_source,
            parse_shape,
            parse_reverse_orientation,
            parse_object_begin,
            parse_object_end,
            parse_object_instance,
            parse_world_end,
            parse_include,
            parse_import,
        )),
        nom::branch::alt((parse_work_dir_begin, parse_work_dir_end)),
    ))(s);
}

fn parse_op_void<'a>(s: &'a str, name: &str) -> IResult<&'a str, OPNode> {
    let (s, _) = sequence::terminated(bytes::complete::tag(name), space0)(s)?;
    return Ok((s, OPNode::new(name, None, None)));
}

fn parse_op_float_n<'a>(s: &'a str, opname: &str, n: usize) -> IResult<&'a str, OPNode> {
    let (s, (op, a)) = nom::branch::permutation((
        sequence::terminated(bytes::complete::tag(opname), space1),
        multi::count(
            sequence::terminated(number::complete::recognize_float, space0),
            n,
        ),
    ))(s)?;
    let mut args = ParameterDictionary::new();
    let v: Vec<Float> = a
        .iter()
        .map(|x| (*x).parse::<f32>().map(|v| v as Float))
        .collect::<Result<_, _>>()
        .map_err(|_| nom::Err::Failure(nom::error::Error::new(s, nom::error::ErrorKind::Fail)))?;
    args.add_floats("args", &v);
    return Ok((s, OPNode::new(op, Some(args), None)));
}

fn parse_op_floats<'a>(s: &'a str, opname: &str) -> IResult<&'a str, OPNode> {
    let (s, (op, a)) = nom::branch::permutation((
        sequence::terminated(bytes::complete::tag(opname), space1),
        sequence::delimited(
            character::complete::char('['),
            sequence::delimited(
                character::complete::multispace0,
                multi::separated_list1(
                    character::complete::multispace1,
                    number::complete::recognize_float,
                ),
                character::complete::multispace0,
            ),
            character::complete::char(']'),
        ),
    ))(s)?;
    let mut args = ParameterDictionary::new();
    let v: Vec<Float> = a
        .iter()
        .map(|x| (*x).parse::<f32>().map(|v| v as Float))
        .collect::<Result<_, _>>()
        .map_err(|_| nom::Err::Failure(nom::error::Error::new(s, nom::error::ErrorKind::Fail)))?;
    args.add_floats("arg1", &v);
    return Ok((s, OPNode::new(op, Some(args), None)));
}

fn parse_op_string<'a>(s: &'a str, opname: &'a str) -> IResult<&'a str, OPNode> {
    let (s, (op, name)) = nom::branch::permutation((
        sequence::terminated(bytes::complete::tag(opname), space1),
        sequence::terminated(string_literal, space0),
    ))(s)?;
    let mut args = ParameterDictionary::new();
    args.add_string("arg1", name);
    return Ok((s, OPNode::new(op, Some(args), None)));
}

fn parse_op_string_string<'a>(s: &'a str, opname: &str) -> IResult<&'a str, OPNode> {
    let (s, (op, b, c)) = nom::branch::permutation((
        sequence::terminated(bytes::complete::tag(opname), space1),
        sequence::terminated(string_literal, space1),
        sequence::terminated(string_literal, space0),
    ))(s)?;
    let mut args = ParameterDictionary::new();
    args.add_string("arg1", b);
    args.add_string("arg2", c);
    return Ok((s, OPNode::new(op, Some(args), None)));
}

// `Option "name" value` — a quoted name followed by a single bare value
// token (bool / number / quoted string). The value is stored unquoted in
// `arg2`.
fn parse_op_string_value<'a>(s: &'a str, opname: &str) -> IResult<&'a str, OPNode> {
    let (s, (op, name, value)) = sequence::tuple((
        sequence::terminated(bytes::complete::tag(opname), space1),
        sequence::terminated(string_literal, space1),
        sequence::terminated(parse_literal, space0),
    ))(s)?;
    let mut args = ParameterDictionary::new();
    args.add_string("arg1", name);
    args.add_string("arg2", value);
    return Ok((s, OPNode::new(op, Some(args), None)));
}

fn parse_op_string_params<'a>(s: &'a str, opname: &str) -> IResult<&'a str, OPNode> {
    let (s, (op, a, params)) = nom::branch::permutation((
        sequence::terminated(bytes::complete::tag(opname), space1),
        sequence::terminated(string_literal, space0),
        parse_params,
    ))(s)?;
    let mut args = ParameterDictionary::new();
    args.add_string("arg1", a);
    return Ok((s, OPNode::new(op, Some(args), Some(params))));
}

fn parse_op_string_string_string_params<'a>(s: &'a str, opname: &str) -> IResult<&'a str, OPNode> {
    let (s, (op, a, params)) = nom::branch::permutation((
        sequence::terminated(bytes::complete::tag(opname), space1),
        multi::count(sequence::terminated(string_literal, space0), 3),
        parse_params,
    ))(s)?;
    let mut args = ParameterDictionary::new();
    args.add_string("arg1", a[0]);
    args.add_string("arg2", a[1]);
    args.add_string("arg3", a[2]);
    return Ok((s, OPNode::new(op, Some(args), Some(params))));
}

fn parse_identity(s: &str) -> IResult<&str, OPNode> {
    return parse_op_void(s, "Identity");
}
//fn translate(&mut self, dx: Float, dy: Float, dz: Float);
fn parse_translate(s: &str) -> IResult<&str, OPNode> {
    return parse_op_float_n(s, "Translate", 3);
}

//fn rotate(&mut self, angle: Float, ax: Float, ay: Float, az: Float);
fn parse_rotate(s: &str) -> IResult<&str, OPNode> {
    return parse_op_float_n(s, "Rotate", 4);
}

//fn scale(&mut self, sx: Float, sy: Float, sz: Float);
fn parse_scale(s: &str) -> IResult<&str, OPNode> {
    return parse_op_float_n(s, "Scale", 3);
}

fn parse_look_at(s: &str) -> IResult<&str, OPNode> {
    return parse_op_float_n(s, "LookAt", 9);
}

fn parse_concat_transform(s: &str) -> IResult<&str, OPNode> {
    return parse_op_floats(s, "ConcatTransform");
}

fn parse_transform(s: &str) -> IResult<&str, OPNode> {
    return parse_op_floats(s, "Transform");
}

//fn coordinate_system(&mut self, name: &str);
fn parse_coordinate_system(s: &str) -> IResult<&str, OPNode> {
    return parse_op_string(s, "CoordinateSystem");
}

//fn coord_sys_transform(&mut self, name: &str);
fn parse_coord_sys_transform(s: &str) -> IResult<&str, OPNode> {
    return parse_op_string(s, "CoordSysTransform");
}

fn parse_color_space(s: &str) -> IResult<&str, OPNode> {
    return parse_op_string(s, "ColorSpace");
}

fn parse_option(s: &str) -> IResult<&str, OPNode> {
    return parse_op_string_value(s, "Option");
}

fn parse_active_transform(s: &str) -> IResult<&str, OPNode> {
    let (s, (op, t)) = nom::branch::permutation((
        sequence::terminated(bytes::complete::tag("ActiveTransform"), space1),
        sequence::terminated(
            nom::branch::alt((
                bytes::complete::tag("All"),
                bytes::complete::tag("EndTime"),
                bytes::complete::tag("StartTime"),
            )),
            space0,
        ),
    ))(s)?;

    //fn active_transform_all(&mut self);
    //fn active_transform_end_time(&mut self);
    //fn active_transform_start_time(&mut self);
    let name = String::from(op) + t;
    return Ok((s, OPNode::new(&name, None, None)));
}

//fn transform_times(&mut self, start: Float, end: Float);
fn parse_transform_times(s: &str) -> IResult<&str, OPNode> {
    return parse_op_float_n(s, "TransformTimes", 2);
}

fn parse_pixel_filter(s: &str) -> IResult<&str, OPNode> {
    return parse_op_string_params(s, "PixelFilter");
}

fn parse_film(s: &str) -> IResult<&str, OPNode> {
    return parse_op_string_params(s, "Film");
}

fn parse_sampler(s: &str) -> IResult<&str, OPNode> {
    return parse_op_string_params(s, "Sampler");
}

fn parse_accelerator(s: &str) -> IResult<&str, OPNode> {
    return parse_op_string_params(s, "Accelerator");
}

fn parse_integrator(s: &str) -> IResult<&str, OPNode> {
    return parse_op_string_params(s, "Integrator");
}

fn parse_camera(s: &str) -> IResult<&str, OPNode> {
    return parse_op_string_params(s, "Camera");
}

fn parse_make_named_medium(s: &str) -> IResult<&str, OPNode> {
    return parse_op_string_params(s, "MakeNamedMedium");
}

fn parse_medium_interface(s: &str) -> IResult<&str, OPNode> {
    return parse_op_string_string(s, "MediumInterface");
}

fn parse_world_begin(s: &str) -> IResult<&str, OPNode> {
    return parse_op_void(s, "WorldBegin");
}

fn parse_attribute(s: &str) -> IResult<&str, OPNode> {
    return parse_op_string_params(s, "Attribute");
}

fn parse_attribute_begin(s: &str) -> IResult<&str, OPNode> {
    return parse_op_void(s, "AttributeBegin");
}

fn parse_attribute_end(s: &str) -> IResult<&str, OPNode> {
    return parse_op_void(s, "AttributeEnd");
}

fn parse_transform_begin(s: &str) -> IResult<&str, OPNode> {
    return parse_op_void(s, "TransformBegin");
}

fn parse_transform_end(s: &str) -> IResult<&str, OPNode> {
    return parse_op_void(s, "TransformEnd");
}

fn parse_texture(s: &str) -> IResult<&str, OPNode> {
    return parse_op_string_string_string_params(s, "Texture");
}

fn parse_material(s: &str) -> IResult<&str, OPNode> {
    return parse_op_string_params(s, "Material");
}

fn parse_make_named_material(s: &str) -> IResult<&str, OPNode> {
    return parse_op_string_params(s, "MakeNamedMaterial");
}

fn parse_named_material(s: &str) -> IResult<&str, OPNode> {
    return parse_op_string(s, "NamedMaterial");
}

fn parse_light_source(s: &str) -> IResult<&str, OPNode> {
    return parse_op_string_params(s, "LightSource");
}

fn parse_area_light_source(s: &str) -> IResult<&str, OPNode> {
    return parse_op_string_params(s, "AreaLightSource");
}

fn parse_shape(s: &str) -> IResult<&str, OPNode> {
    return parse_op_string_params(s, "Shape");
}

fn parse_reverse_orientation(s: &str) -> IResult<&str, OPNode> {
    return parse_op_void(s, "ReverseOrientation");
}

fn parse_object_begin(s: &str) -> IResult<&str, OPNode> {
    return parse_op_string(s, "ObjectBegin");
}

fn parse_object_end(s: &str) -> IResult<&str, OPNode> {
    return parse_op_void(s, "ObjectEnd");
}

fn parse_object_instance(s: &str) -> IResult<&str, OPNode> {
    return parse_op_string(s, "ObjectInstance");
}

fn parse_world_end(s: &str) -> IResult<&str, OPNode> {
    return parse_op_void(s, "WorldEnd");
}

fn parse_work_dir_begin(s: &str) -> IResult<&str, OPNode> {
    return parse_op_string(s, "WorkDirBegin");
}

fn parse_work_dir_end(s: &str) -> IResult<&str, OPNode> {
    return parse_op_void(s, "WorkDirEnd");
}

fn parse_include(s: &str) -> IResult<&str, OPNode> {
    return parse_op_string_params(s, "Include");
}

fn parse_import(s: &str) -> IResult<&str, OPNode> {
    return parse_op_string_params(s, "Import");
}
