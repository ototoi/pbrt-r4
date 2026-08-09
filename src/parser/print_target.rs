use super::parse_target::*;
use crate::paramdict::*;
use crate::util::base::*;

use std::cell::{Cell, RefCell};
use std::io::Write;
use std::sync::Arc;

fn get_param_type(s: &str) -> (&str, &str) {
    let ss: Vec<&str> = s.split_ascii_whitespace().collect();
    if ss.len() == 2 {
        return (ss[0], ss[1]);
    } else if ss.len() == 1 {
        if let Some(t) = wellknown_params::find_type_from_key(ss[0]) {
            return (t, ss[0]);
        } else {
            return ("", ss[0]);
        }
    } else {
        return ("", s);
    }
}

fn get_type(s: &str) -> &str {
    let (t, _) = get_param_type(s);
    match t {
        "string" => "s",
        "spectrum" => "s",
        "texture" => "s",
        "bool" => "b",
        "integer" => "i",
        "point" | "point2" | "point3" | "point4" => "p",
        "normal" => "p",
        "vector" | "vector2" | "vector3" | "vector4" => "p",
        "color" => "p",
        "rgb" => "p",
        "blackbody" => "f",
        _ => "f",
    }
}

pub struct PrintTarget {
    pub writer: Arc<RefCell<dyn Write>>,
    pub omit_long_values: bool,
    pub indent: Cell<i32>,
}

impl PrintTarget {
    pub fn new(w: Arc<RefCell<dyn Write>>) -> Self {
        PrintTarget {
            writer: w,
            omit_long_values: false,
            indent: Cell::<i32>::new(0),
        }
    }

    pub fn new_with_params(w: Arc<RefCell<dyn Write>>, omit_long_values: bool) -> Self {
        PrintTarget {
            writer: w,
            omit_long_values,
            indent: Cell::<i32>::new(0),
        }
    }

    pub fn new_stdout(omit_long_values: bool) -> Self {
        PrintTarget {
            writer: Arc::new(RefCell::new(std::io::stdout())),
            omit_long_values,
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
        return self.get_indent_i(self.indent.get());
    }

    pub fn get_indent_i(&self, count: i32) -> String {
        let mut s = String::new();
        for _ in 0..count {
            s += "    ";
        }
        return s;
    }

    fn print(&self, s: &str) {
        _ = self.writer.borrow_mut().write_all(s.as_bytes());
    }

    fn convert_transform(&self, values: &[Float]) -> String {
        let mut s = String::from("[");
        let len = values.len();
        for i in 0..len {
            let v = values[i];
            s += &format!("{v}");
            if i != len - 1 {
                s += " ";
            }
        }
        s += "]";
        return s;
    }

    fn convert_values(&self, params: &ParameterDictionary, key: &str) -> String {
        let mut s = String::from("");
        let t = get_type(key);
        s += "[";
        match t {
            "s" => {
                let values = params.get_strings(key);
                let len = values.len();
                for i in 0..len {
                    let v = &values[i];
                    s += &format!("\"{v}\"");
                    if i != len - 1 {
                        s += " ";
                    }
                }
            }
            "b" => {
                let values = params.get_bools(key);
                let len = values.len();
                for i in 0..len {
                    let v = values[i];
                    if v {
                        s += "\"true\"";
                    } else {
                        s += "\"false\"";
                    }
                    if i != len - 1 {
                        s += " ";
                    }
                }
            }
            "i" => {
                let Some(values) = params.get_ints_ref(key) else {
                    return String::new();
                };
                if !self.omit_long_values || values.len() <= 16 {
                    let len = values.len();
                    for i in 0..len {
                        let v = values[i];
                        s += &format!("{v}");
                        if i != len - 1 {
                            s += " ";
                        }
                    }
                } else {
                    s += "...";
                }
            }
            "p" => {
                let Some(values) = params.get_points_ref(key) else {
                    return String::new();
                };
                if !self.omit_long_values || values.len() <= 16 {
                    let len = values.len();
                    for i in 0..len {
                        let v = values[i];
                        s += &format!("{v}");
                        if i != len - 1 {
                            s += " ";
                        }
                    }
                } else {
                    s += "...";
                }
            }
            _ => {
                let Some(values) = params.get_floats_ref(key) else {
                    return String::new();
                };
                if !self.omit_long_values || values.len() <= 16 {
                    let len = values.len();
                    for i in 0..len {
                        let v = values[i];
                        s += &format!("{v}");
                        if i != len - 1 {
                            s += " ";
                        }
                    }
                } else {
                    s += "...";
                }
            }
        }
        s += "]";
        return s;
    }

    fn convert_params(&self, params: &ParameterDictionary) -> String {
        let mut s = String::from("");
        let keys = params.get_keys();
        if !keys.is_empty() {
            let indent = self.get_indent_i(self.indent.get() + 1);
            for key in keys {
                let values = self.convert_values(params, &key);
                s += "\n";
                s += &format!("{indent}\"{key}\" {values}");
            }
        }
        return s;
    }

    fn with_params(&self, params: &ParameterDictionary) -> String {
        if !params.get_keys().is_empty() {
            return format!(" {}", self.convert_params(params));
        } else {
            return String::from("");
        }
    }
}

impl ParseTarget for PrintTarget {
    fn cleanup(&mut self) {
        self.print(&format!("{}Cleanup\n", self.get_indent()));
    }

    fn identity(&mut self) {
        self.print(&format!("{}Identity\n", self.get_indent()));
    }

    fn translate(&mut self, dx: Float, dy: Float, dz: Float) {
        self.print(&format!("{}Translate {dx} {dy} {dz}\n", self.get_indent()));
    }

    fn rotate(&mut self, angle: Float, ax: Float, ay: Float, az: Float) {
        self.print(&format!(
            "{}Rotate {angle} {ax} {ay} {az}\n",
            self.get_indent()
        ));
    }

    fn scale(&mut self, sx: Float, sy: Float, sz: Float) {
        self.print(&format!("{}Scale {sx} {sy} {sz}\n", self.get_indent()));
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
        self.print(&format!(
            "{}LookAt {ex} {ey} {ez} {lx} {ly} {lz} {ux} {uy} {uz}\n",
            self.get_indent()
        ));
    }

    fn concat_transform(&mut self, transform: &[Float]) {
        let s_transform = self.convert_transform(transform);
        self.print(&format!(
            "{}ConcatTransform {s_transform}\n",
            self.get_indent()
        ));
    }

    fn transform(&mut self, transform: &[Float]) {
        let s_transform = self.convert_transform(transform);
        self.print(&format!("{}Transform {s_transform}\n", self.get_indent()));
    }

    fn color_space(&mut self, name: &str) {
        self.print(&format!("{}ColorSpace \"{name}\"\n", self.get_indent()));
    }

    fn option(&mut self, name: &str, value: &str) {
        self.print(&format!("{}Option \"{name}\" {value}\n", self.get_indent()));
    }

    fn attribute(&mut self, target: &str, params: &ParameterDictionary) {
        let s_params = self.with_params(params);
        self.print(&format!(
            "{}Attribute \"{target}\"{s_params}\n",
            self.get_indent()
        ));
    }

    fn coordinate_system(&mut self, name: &str) {
        self.print(&format!(
            "{}CoordinateSystem \"{name}\"\n",
            self.get_indent()
        ));
    }

    fn coord_sys_transform(&mut self, name: &str) {
        self.print(&format!(
            "{}CoordSysTransform \"{name}\"\n",
            self.get_indent()
        ));
    }

    fn active_transform_all(&mut self) {
        self.print(&format!("{}ActiveTransfrom All\n", self.get_indent()));
    }

    fn active_transform_end_time(&mut self) {
        self.print(&format!("{}ActiveTransfrom EndTime\n", self.get_indent()));
    }

    fn active_transform_start_time(&mut self) {
        self.print(&format!("{}ActiveTransfrom StartTime\n", self.get_indent()));
    }

    fn transform_times(&mut self, start: Float, end: Float) {
        self.print(&format!(
            "{}TransformTimes {start} {end}\n",
            self.get_indent()
        ));
    }

    fn pixel_filter(&mut self, name: &str, params: &ParameterDictionary) {
        let s_params = self.with_params(params);
        self.print(&format!(
            "{}PixelFilter \"{name}\"{s_params}\n",
            self.get_indent()
        ));
    }

    fn film(&mut self, name: &str, params: &ParameterDictionary) {
        let s_params = self.with_params(params);
        self.print(&format!("{}Film \"{name}\"{s_params}\n", self.get_indent()));
    }

    fn sampler(&mut self, name: &str, params: &ParameterDictionary) {
        let s_params = self.with_params(params);
        self.print(&format!(
            "{}Sampler \"{name}\"{s_params}\n",
            self.get_indent()
        ));
    }

    fn accelerator(&mut self, name: &str, params: &ParameterDictionary) {
        let s_params = self.with_params(params);
        self.print(&format!(
            "{}Accelerator \"{name}\"{s_params}\n",
            self.get_indent()
        ));
    }

    fn integrator(&mut self, name: &str, params: &ParameterDictionary) {
        let s_params = self.with_params(params);
        self.print(&format!(
            "{}Integrator \"{name}\"{s_params}\n",
            self.get_indent()
        ));
    }

    fn camera(&mut self, name: &str, params: &ParameterDictionary) {
        let s_params = self.with_params(params);
        self.print(&format!(
            "{}Camera \"{name}\"{s_params}\n",
            self.get_indent()
        ));
    }

    fn make_named_medium(&mut self, name: &str, params: &ParameterDictionary) {
        let s_params = self.with_params(params);
        self.print(&format!(
            "{}NamedMedium \"{name}\"{s_params}\n",
            self.get_indent()
        ));
    }

    fn medium_interface(&mut self, inside_name: &str, outside_name: &str) {
        self.print(&format!(
            "{}MediumInterface \"{inside_name}\" \"{outside_name}\"\n",
            self.get_indent()
        ));
    }

    fn world_begin(&mut self) {
        self.print(&format!("{}WorldBegin\n", self.get_indent()));
        self.inc_indent();
    }

    fn attribute_begin(&mut self) {
        self.print(&format!("{}AttributeBegin\n", self.get_indent()));
        self.inc_indent();
    }

    fn attribute_end(&mut self) {
        self.dec_indent();
        self.print(&format!("{}AttributeEnd\n", self.get_indent()));
    }

    fn transform_begin(&mut self) {
        self.print(&format!("{}TransformBegin\n", self.get_indent()));
        self.inc_indent();
    }

    fn transform_end(&mut self) {
        self.dec_indent();
        self.print(&format!("{}TransformEnd\n", self.get_indent()));
    }

    fn texture(&mut self, name: &str, t: &str, tex_name: &str, params: &ParameterDictionary) {
        let s_params = self.with_params(params);
        self.print(&format!(
            "{}Texture \"{name}\" \"{t}\" \"{tex_name}\"{s_params}\n",
            self.get_indent()
        ));
    }

    fn material(&mut self, name: &str, params: &ParameterDictionary) {
        let s_params = self.with_params(params);
        self.print(&format!(
            "{}Material \"{name}\"{s_params}\n",
            self.get_indent()
        ));
    }

    fn make_named_material(&mut self, name: &str, params: &ParameterDictionary) {
        let s_params = self.with_params(params);
        self.print(&format!(
            "{}MakeNamedMaterial \"{name}\"{s_params}\n",
            self.get_indent()
        ));
    }

    fn named_material(&mut self, name: &str) {
        self.print(&format!("{}NamedMaterial \"{name}\"\n", self.get_indent()));
    }

    fn light_source(&mut self, name: &str, params: &ParameterDictionary) {
        let s_params = self.with_params(params);
        self.print(&format!(
            "{}LightSource \"{name}\"{s_params}\n",
            self.get_indent()
        ));
    }

    fn area_light_source(&mut self, name: &str, params: &ParameterDictionary) {
        let s_params = self.with_params(params);
        self.print(&format!(
            "{}AreaLightSource \"{name}\"{s_params}\n",
            self.get_indent()
        ));
    }

    fn shape(&mut self, name: &str, params: &ParameterDictionary) {
        let s_params = self.with_params(params);
        self.print(&format!(
            "{}Shape \"{name}\"{s_params}\n",
            self.get_indent()
        ));
    }

    fn reverse_orientation(&mut self) {
        self.print(&format!("{}ReverseOrientation\n", self.get_indent()));
    }

    fn object_begin(&mut self, name: &str) {
        self.print(&format!("{}ObjectBegin \"{name}\"\n", self.get_indent()));
        self.inc_indent();
    }

    fn object_end(&mut self) {
        self.dec_indent();
        self.print(&format!("{}ObjectEnd\n", self.get_indent()));
    }

    fn object_instance(&mut self, name: &str) {
        self.print(&format!("{}ObjectInstance \"{name}\"\n", self.get_indent()));
    }

    fn world_end(&mut self) {
        self.dec_indent();
        self.print(&format!("{}WorldEnd\n", self.get_indent()));
    }

    fn parse_file(&mut self, _file_name: &str) {}
    fn parse_string(&mut self, _s: &str) {}

    fn work_dir_begin(&mut self, _path: &str) {
        //Do not anything!
    }

    fn work_dir_end(&mut self) {
        //Do not anything!
    }

    fn include(&mut self, filename: &str, params: &ParameterDictionary) {
        let s_params = self.with_params(params);
        self.print(&format!(
            "{}Include \"{filename}\"{s_params}\n",
            self.get_indent()
        ));
    }

    fn import(&mut self, filename: &str, params: &ParameterDictionary) {
        let s_params = self.with_params(params);
        self.print(&format!(
            "{}Import \"{filename}\"{s_params}\n",
            self.get_indent()
        ));
    }
}
