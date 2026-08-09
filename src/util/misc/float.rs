use crate::util::base::Float;

#[inline]
pub fn float_to_bits_32(f: f32) -> u32 {
    f.to_bits()
}

#[inline]
pub fn bits_to_float_32(ui: u32) -> f32 {
    f32::from_bits(ui)
}

#[inline]
pub fn float_to_bits_64(f: f64) -> u64 {
    f.to_bits()
}

#[inline]
pub fn bits_to_float_64(ui: u64) -> f64 {
    f64::from_bits(ui)
}

pub fn next_float_down_32(x: f32) -> f32 {
    let mut v = x;
    if f32::is_infinite(v) && v < 0.0 {
        return v;
    }
    if v == 0.0 {
        v = -0.0;
    }

    let mut ui = float_to_bits_32(v);
    if v > 0.0 {
        ui = ui.saturating_sub(1);
    } else {
        ui = ui.saturating_add(1);
    }
    return bits_to_float_32(ui);
}

pub fn next_float_up_32(x: f32) -> f32 {
    let mut v = x;
    if f32::is_infinite(v) && v > 0.0 {
        return v;
    }
    if v == -0.0 {
        v = 0.0;
    }
    let mut ui = float_to_bits_32(v);
    if v >= 0.0 {
        ui = ui.saturating_add(1);
    } else {
        ui = ui.saturating_sub(1);
    }
    return bits_to_float_32(ui);
}

pub fn next_float_down_64(x: f64) -> f64 {
    let mut v = x;
    if f64::is_infinite(v) && v < 0.0 {
        return v;
    }
    if v == 0.0 {
        v = -0.0;
    }

    let mut ui = float_to_bits_64(v);
    if v > 0.0 {
        ui = ui.saturating_sub(1);
    } else {
        ui = ui.saturating_add(1);
    }
    return bits_to_float_64(ui);
}

pub fn next_float_up_64(x: f64) -> f64 {
    let mut v = x;
    if f64::is_infinite(v) && v > 0.0 {
        return v;
    }
    if v == -0.0 {
        v = 0.0;
    }
    let mut ui = float_to_bits_64(v);
    if v >= 0.0 {
        ui = ui.saturating_add(1);
    } else {
        ui = ui.saturating_sub(1);
    }
    return bits_to_float_64(ui);
}

pub fn next_float_down(x: Float) -> Float {
    #[cfg(not(feature = "float-as-double"))]
    return next_float_down_32(x);
    #[cfg(feature = "float-as-double")]
    return next_float_down_64(x);
}

pub fn next_float_up(x: Float) -> Float {
    #[cfg(not(feature = "float-as-double"))]
    return next_float_up_32(x);
    #[cfg(feature = "float-as-double")]
    return next_float_up_64(x);
}
