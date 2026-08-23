use crate::parser::ParsedParameterVector;
use crate::util::base::*;

//Sized
pub trait ParseTarget {
    fn cleanup(&mut self);
    fn identity(&mut self);
    fn translate(&mut self, dx: Float, dy: Float, dz: Float);
    fn rotate(&mut self, angle: Float, ax: Float, ay: Float, az: Float);
    fn scale(&mut self, sx: Float, sy: Float, sz: Float);
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
    );
    fn concat_transform(&mut self, tansform: &[Float]);
    fn transform(&mut self, tansform: &[Float]);
    fn color_space(&mut self, name: &str);
    fn coordinate_system(&mut self, name: &str);
    fn coord_sys_transform(&mut self, name: &str);
    fn option(&mut self, name: &str, value: &str);
    fn active_transform_all(&mut self);
    fn active_transform_end_time(&mut self);
    fn active_transform_start_time(&mut self);
    fn transform_times(&mut self, start: Float, end: Float);
    fn pixel_filter(&mut self, name: &str, params: ParsedParameterVector);
    fn film(&mut self, name: &str, params: ParsedParameterVector);
    fn sampler(&mut self, name: &str, params: ParsedParameterVector);
    fn accelerator(&mut self, name: &str, params: ParsedParameterVector);
    fn integrator(&mut self, name: &str, params: ParsedParameterVector);
    fn camera(&mut self, name: &str, params: ParsedParameterVector);
    fn make_named_medium(&mut self, name: &str, params: ParsedParameterVector);
    fn medium_interface(&mut self, inside_name: &str, outside_name: &str);

    fn world_begin(&mut self);
    fn attribute(&mut self, target: &str, params: ParsedParameterVector);
    fn attribute_begin(&mut self);
    fn attribute_end(&mut self);
    fn transform_begin(&mut self);
    fn transform_end(&mut self);
    fn texture(&mut self, name: &str, _type: &str, tex_name: &str, params: ParsedParameterVector);
    fn material(&mut self, name: &str, params: ParsedParameterVector);
    fn make_named_material(&mut self, name: &str, params: ParsedParameterVector);
    fn named_material(&mut self, name: &str);
    fn light_source(&mut self, name: &str, params: ParsedParameterVector);
    fn area_light_source(&mut self, name: &str, params: ParsedParameterVector);
    fn shape(&mut self, name: &str, params: ParsedParameterVector);
    fn reverse_orientation(&mut self);
    fn object_begin(&mut self, name: &str);
    fn object_end(&mut self);
    fn object_instance(&mut self, name: &str);
    fn world_end(&mut self);

    fn parse_file(&mut self, filename: &str);
    fn parse_string(&mut self, s: &str);

    //----------------------------------------
    fn work_dir_begin(&mut self, _path: &str) {}
    fn work_dir_end(&mut self) {}
    fn include(&mut self, _filename: &str, _params: ParsedParameterVector) {}
    fn import(&mut self, filename: &str, params: ParsedParameterVector);
    //----------------------------------------
}
