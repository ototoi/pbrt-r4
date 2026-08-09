use crate::util::base::*;

fn toa(v: &Vector3f) -> [Float; 3] {
    [v.x, v.y, v.z]
}

pub fn intersect_box(
    min: &Vector3f,
    max: &Vector3f,
    org: &Vector3f,
    dir: &Vector3f,
    t0: Float,
    t1: Float,
) -> (bool, Float, Float) {
    return intersect_box_array(&toa(min), &toa(max), &toa(org), &toa(dir), t0, t1);
}

pub fn intersect_box_array(
    min: &[Float; 3],
    max: &[Float; 3],
    org: &[Float; 3],
    dir: &[Float; 3],
    t0: Float,
    t1: Float,
) -> (bool, Float, Float) {
    let idir = [
        //1.0 / dir[0],
        //1.0 / dir[1],
        //1.0 / dir[2],
        Float::recip(dir[0]),
        Float::recip(dir[1]),
        Float::recip(dir[2]),
    ];
    let sign = [
        if dir[0].is_sign_negative() { 1 } else { 0 },
        if dir[1].is_sign_negative() { 1 } else { 0 },
        if dir[2].is_sign_negative() { 1 } else { 0 },
    ];
    if let Some((t0, t1)) = intersect_box_array_i(min, max, org, &idir, &sign, t0, t1) {
        return (true, t0, t1);
    } else {
        return (false, t0, t1);
    }
}

pub fn intersect_box_array_i(
    min: &[Float; 3],
    max: &[Float; 3],
    org: &[Float; 3],
    idir: &[Float; 3],
    sign: &[usize; 3],
    mut t0: Float,
    mut t1: Float,
) -> Option<(Float, Float)> {
    let bounds = [min, max];
    for i in 0..3 {
        t0 = Float::max(t0, (bounds[sign[i]][i] - org[i]) * idir[i]);
        t1 = Float::min(t1, (bounds[1 - sign[i]][i] - org[i]) * idir[i]);
    }
    if t0 <= t1 {
        return Some((t0, t1));
    } else {
        return None;
    }
}
