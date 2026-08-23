use super::parse_target::*;
use super::ParsedParameterVector;
use crate::util::base::*;
use std::cell::{Cell, RefCell};

pub struct Operation {
    pub name: String,
    pub args: Vec<String>,
}

impl Operation {
    pub fn new(op: &str, args: &[String]) -> Self {
        Operation {
            name: op.to_string(),
            args: args.to_vec(),
        }
    }
}

#[derive(Default)]
pub struct DebugTarget {
    pub operations: RefCell<Vec<Operation>>,
    pub indent: Cell<i32>,
}

impl DebugTarget {
    pub fn new() -> Self {
        DebugTarget {
            operations: RefCell::<Vec<Operation>>::new(Vec::<Operation>::new()),
            indent: Cell::<i32>::new(0),
        }
    }
    pub fn inc_indent(&mut self) {
        self.indent.set(self.indent.get() + 1);
    }
    pub fn dec_indent(&mut self) {
        self.indent.set(self.indent.get() - 1);
    }

    pub fn get_indent(&self) -> String {
        let mut s = String::new();
        let count = self.indent.get();
        for _ in 0..count {
            s += "    ";
        }
        return s;
    }
}

impl ParseTarget for DebugTarget {
    fn cleanup(&mut self) {
        println!("{}cleanup", self.get_indent());
        let v = vec![];
        self.operations
            .borrow_mut()
            .push(Operation::new("Cleanup", &v));
    }
    fn identity(&mut self) {
        println!("{}identity", self.get_indent());
        let v = vec![String::from("")];
        self.operations
            .borrow_mut()
            .push(Operation::new("Identitiy", &v));
    }
    fn translate(&mut self, dx: Float, dy: Float, dz: Float) {
        println!("{}translate:[{dx}, {dy}, {dz}]", self.get_indent());
        let v = vec![String::from("")];
        self.operations
            .borrow_mut()
            .push(Operation::new("Translate", &v));
    }
    fn rotate(&mut self, angle: Float, ax: Float, ay: Float, az: Float) {
        println!("{}rotate:[{angle}, {ax}, {ay}, {az}]", self.get_indent());
        let v = vec![String::from("")];
        self.operations
            .borrow_mut()
            .push(Operation::new("Rotate", &v));
    }
    fn scale(&mut self, sx: Float, sy: Float, sz: Float) {
        println!("{}scale:[{sx}, {sy}, {sz}]", self.get_indent());
        let v = vec![String::from("")];
        self.operations
            .borrow_mut()
            .push(Operation::new("Scale", &v));
    }
    fn look_at(
        &mut self,
        ex: Float,
        ey: Float,
        ez: Float,
        lx: Float,
        ly: Float,
        lz: Float,
        ux: Float,
        uy: Float,
        uz: Float,
    ) {
        println!(
            "{}look_at:[{ex}, {ey}, {ez}, {lx}, {ly}, {lz}, {ux}, {uy}, {uz}]",
            self.get_indent()
        );
        let v = vec![String::from("")];
        self.operations
            .borrow_mut()
            .push(Operation::new("LookAt", &v));
    }

    fn concat_transform(&mut self, _tansform: &[Float]) {
        println!("{}concat_transform", self.get_indent());
        let v = vec![String::from("")];
        self.operations
            .borrow_mut()
            .push(Operation::new("ConcatTransform", &v));
    }
    fn transform(&mut self, _tansform: &[Float]) {
        println!("{}transform", self.get_indent());
        let v = vec![String::from("")];
        self.operations
            .borrow_mut()
            .push(Operation::new("Transform", &v));
    }
    fn color_space(&mut self, name: &str) {
        println!("{}color_space:\"{name}\"", self.get_indent());
        let v = vec![String::from(name)];
        self.operations
            .borrow_mut()
            .push(Operation::new("ColorSpace", &v));
    }
    fn option(&mut self, name: &str, value: &str) {
        println!("{}option:\"{name}\" {value}", self.get_indent());
        let v = vec![String::from(name), String::from(value)];
        self.operations
            .borrow_mut()
            .push(Operation::new("Option", &v));
    }
    fn coordinate_system(&mut self, name: &str) {
        println!("{}coordinate_system:\"{name}\"", self.get_indent());
        let v = vec![String::from(name)];
        self.operations
            .borrow_mut()
            .push(Operation::new("CoordinateSystem", &v));
    }
    fn coord_sys_transform(&mut self, name: &str) {
        println!("{}coord_sys_transform:\"{name}\"", self.get_indent());
        let v = vec![String::from(name)];
        self.operations
            .borrow_mut()
            .push(Operation::new("CoordSysTransform", &v));
    }
    fn active_transform_all(&mut self) {
        println!("{}active_transform_all", self.get_indent());
        let v = vec![String::from("")];
        self.operations
            .borrow_mut()
            .push(Operation::new("ActiveTransformAll", &v));
    }
    fn active_transform_end_time(&mut self) {
        println!("{}active_transform_end_time", self.get_indent());
        let v = vec![String::from("")];
        self.operations
            .borrow_mut()
            .push(Operation::new("ActiveTransformEndTime", &v));
    }
    fn active_transform_start_time(&mut self) {
        println!("{}active_transform_start_time", self.get_indent());
        let v = vec![String::from("")];
        self.operations
            .borrow_mut()
            .push(Operation::new("ActiveTransformStartTime", &v));
    }
    fn transform_times(&mut self, _start: Float, _end: Float) {
        println!("{}transform_times", self.get_indent());
        let v = vec![String::from("")];
        self.operations
            .borrow_mut()
            .push(Operation::new("TransformTimes", &v));
    }
    fn pixel_filter(&mut self, name: &str, _params: ParsedParameterVector) {
        println!("{}pixel_filter:\"{name}\"", self.get_indent());
        let v = vec![String::from(name)];
        self.operations
            .borrow_mut()
            .push(Operation::new("PixelFilter", &v));
    }
    fn film(&mut self, name: &str, _params: ParsedParameterVector) {
        println!("{}film:\"{name}\"", self.get_indent());
        let v = vec![String::from(name)];
        self.operations
            .borrow_mut()
            .push(Operation::new("Film", &v));
    }
    fn sampler(&mut self, name: &str, _params: ParsedParameterVector) {
        println!("{}sampler:\"{name}\"", self.get_indent());
        let v = vec![String::from(name)];
        self.operations
            .borrow_mut()
            .push(Operation::new("Sampler", &v));
    }
    fn accelerator(&mut self, name: &str, _params: ParsedParameterVector) {
        println!("{}accelerator:\"{name}\"", self.get_indent());
        let v = vec![String::from(name)];
        self.operations
            .borrow_mut()
            .push(Operation::new("Accelerator", &v));
    }
    fn integrator(&mut self, name: &str, _params: ParsedParameterVector) {
        println!("{}integrator:\"{name}\"", self.get_indent());
        let v = vec![String::from(name)];
        self.operations
            .borrow_mut()
            .push(Operation::new("Integrator", &v));
    }
    fn camera(&mut self, name: &str, _params: ParsedParameterVector) {
        println!("{}camera:\"{name}\"", self.get_indent());
        let v = vec![String::from("")];
        self.operations
            .borrow_mut()
            .push(Operation::new("Camera", &v));
    }
    fn make_named_medium(&mut self, name: &str, _params: ParsedParameterVector) {
        println!("{}make_named_medium:\"{name}\"", self.get_indent());
        let v = vec![String::from("")];
        self.operations
            .borrow_mut()
            .push(Operation::new("NamedMedium", &v));
    }
    fn medium_interface(&mut self, inside_name: &str, outside_name: &str) {
        println!(
            "{}medium_interface:\"{inside_name}\", \"{outside_name}\"",
            self.get_indent()
        );
        let v = vec![String::from("")];
        self.operations
            .borrow_mut()
            .push(Operation::new("MediumInterface", &v));
    }
    fn world_begin(&mut self) {
        println!("{}world_begin", self.get_indent());
        let v = vec![String::from("")];
        self.operations
            .borrow_mut()
            .push(Operation::new("WorldBegin", &v));
        self.inc_indent();
    }
    fn attribute(&mut self, target: &str, _params: ParsedParameterVector) {
        println!("{}attribute:\"{target}\"", self.get_indent());
        let v = vec![String::from(target)];
        self.operations
            .borrow_mut()
            .push(Operation::new("Attribute", &v));
    }
    fn attribute_begin(&mut self) {
        println!("{}attribute_begin", self.get_indent());
        let v = vec![String::from("")];
        self.operations
            .borrow_mut()
            .push(Operation::new("AttributeBegin", &v));
        self.inc_indent();
    }
    fn attribute_end(&mut self) {
        self.dec_indent();
        println!("{}attribute_end", self.get_indent());
        let v = vec![String::from("")];
        self.operations
            .borrow_mut()
            .push(Operation::new("AttributeEnd", &v));
    }
    fn transform_begin(&mut self) {
        println!("{}transform_begin", self.get_indent());
        let v = vec![String::from("")];
        self.operations
            .borrow_mut()
            .push(Operation::new("TransformBegin", &v));
        self.inc_indent();
    }
    fn transform_end(&mut self) {
        self.dec_indent();
        println!("{}transform_end", self.get_indent());
        let v = vec![String::from("")];
        self.operations
            .borrow_mut()
            .push(Operation::new("TransformEnd", &v));
    }
    fn texture(&mut self, name: &str, t: &str, tex_name: &str, _params: ParsedParameterVector) {
        println!(
            "{}texture:\"{}\", \"{}\", \"{}\"",
            self.get_indent(),
            name,
            t,
            tex_name
        );
        let v = vec![String::from("")];
        self.operations
            .borrow_mut()
            .push(Operation::new("Texture", &v));
    }
    fn material(&mut self, name: &str, _params: ParsedParameterVector) {
        println!("{}material:\"{name}\"", self.get_indent());
        let v = vec![String::from(name)];
        self.operations
            .borrow_mut()
            .push(Operation::new("Material", &v));
    }
    fn make_named_material(&mut self, name: &str, _params: ParsedParameterVector) {
        println!("{}make_named_material:\"{name}\"", self.get_indent());
        let v = vec![String::from(name)];
        self.operations
            .borrow_mut()
            .push(Operation::new("MakeNamedMaterial", &v));
    }
    fn named_material(&mut self, name: &str) {
        println!("{}named_material:\"{name}\"", self.get_indent());
        let v = vec![String::from("")];
        self.operations
            .borrow_mut()
            .push(Operation::new("NamedMaterial", &v));
    }
    fn light_source(&mut self, name: &str, _params: ParsedParameterVector) {
        println!("{}light_source:\"{name}\"", self.get_indent());
        let v = vec![String::from(name)];
        self.operations
            .borrow_mut()
            .push(Operation::new("LightSource", &v));
    }
    fn area_light_source(&mut self, name: &str, _params: ParsedParameterVector) {
        println!("{}area_light_source:\"{name}\"", self.get_indent());
        let v = vec![String::from(name)];
        self.operations
            .borrow_mut()
            .push(Operation::new("AreaLightSource", &v));
    }
    fn shape(&mut self, name: &str, _params: ParsedParameterVector) {
        println!("{}shape:\"{name}\"", self.get_indent());
        let v = vec![String::from(name)];
        self.operations
            .borrow_mut()
            .push(Operation::new("Shape", &v));
    }
    fn reverse_orientation(&mut self) {
        println!("{}reverse_orientation", self.get_indent());
        let v = vec![String::from("")];
        self.operations
            .borrow_mut()
            .push(Operation::new("ReverseOrientation", &v));
    }
    fn object_begin(&mut self, name: &str) {
        println!("{}object_begin:\"{name}\"", self.get_indent());
        let v = vec![String::from(name)];
        self.operations
            .borrow_mut()
            .push(Operation::new("ObjectBegin", &v));
        self.inc_indent();
    }
    fn object_end(&mut self) {
        self.dec_indent();
        println!("{}object_end", self.get_indent());
        let v = vec![String::from("")];
        self.operations
            .borrow_mut()
            .push(Operation::new("ObjectEnd", &v));
    }
    fn object_instance(&mut self, name: &str) {
        println!("{}object_instance:\"{name}\"", self.get_indent());
        let v = vec![String::from(name)];
        self.operations
            .borrow_mut()
            .push(Operation::new("ObjectInstance", &v));
    }
    fn world_end(&mut self) {
        self.dec_indent();
        println!("{}world_end", self.get_indent());
        let v = vec![String::from("")];
        self.operations
            .borrow_mut()
            .push(Operation::new("WorldEnd", &v));
    }
    fn parse_file(&mut self, file_name: &str) {
        println!("{}parse_file:\"{file_name}\"", self.get_indent());
        let v = vec![String::from("")];
        self.operations
            .borrow_mut()
            .push(Operation::new("ParseFile", &v));
    }
    fn parse_string(&mut self, _s: &str) {
        println!("{}parse_string", self.get_indent());
        let v = vec![String::from("")];
        self.operations
            .borrow_mut()
            .push(Operation::new("ParseString", &v));
    }

    fn work_dir_begin(&mut self, path: &str) {
        println!("{}work_dir_begin:\"{path}\"", self.get_indent());
        let v = vec![String::from(path)];
        self.operations
            .borrow_mut()
            .push(Operation::new("WorkDirBegin", &v));
    }

    fn work_dir_end(&mut self) {
        println!("{}work_dir_end", self.get_indent());
        let v = vec![String::from("")];
        self.operations
            .borrow_mut()
            .push(Operation::new("WorkDirEnd", &v));
    }

    fn include(&mut self, filename: &str, _params: ParsedParameterVector) {
        println!("{}include:\"{filename}\"", self.get_indent());
        let v = vec![String::from(filename)];
        self.operations
            .borrow_mut()
            .push(Operation::new("Include", &v));
    }

    fn import(&mut self, filename: &str, _params: ParsedParameterVector) {
        println!("{}import:\"{filename}\"", self.get_indent());
        let v = vec![String::from(filename)];
        self.operations
            .borrow_mut()
            .push(Operation::new("Import", &v));
    }
}
