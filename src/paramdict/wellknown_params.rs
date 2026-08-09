const WELLKNOWN_PARAMS: [(&str, &str, &str); 28] = [
    ("", "integer", "xresolution"),
    ("", "integer", "yresolution"),
    ("", "integer", "maxdepth"),
    ("", "integer", "pixelsamples"),
    ("", "integer", "indices"),
    ("", "string", "filename"),
    ("", "string", "mapname"),
    ("", "string", "normalmap"),
    ("", "string", "sensor"),
    ("", "float", "fov"),
    ("", "float", "radius"),
    ("", "float", "iso"),
    ("", "float", "lensradius"),
    ("", "float", "focaldistance"),
    ("", "float", "maxcomponentvalue"),
    ("", "float", "scale"),
    ("", "float", "uv"),
    ("", "color", "L"),
    ("", "blackbody", "L"),
    ("", "rgb", "L"),
    ("", "point", "P"),
    ("", "point3", "P"),
    ("", "normal", "N"),
    ("", "vector", "v"),
    ("", "vector3", "v"),
    ("", "texture", "Kd"),
    ("", "texture", "Kr"),
    ("", "color", "Kd"),
];

pub fn find_type_from_key(key: &str) -> Option<&str> {
    if let Some((_, t, _k)) = WELLKNOWN_PARAMS.iter().find(|(_a, _tt, kk)| -> bool {
        return *kk == key;
    }) {
        return Some(*t);
    } else {
        return None;
    }
}
