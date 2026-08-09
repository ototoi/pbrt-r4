use crate::util::base::Float;

#[inline]
pub fn lerp(t: Float, a: Float, b: Float) -> Float {
    a + (b - a) * t
}

pub fn spectrum_samples_sorted(lambda: &[Float], _values: &[Float]) -> bool {
    if lambda.len() < 2 {
        return true;
    }
    for i in 0..(lambda.len() - 1) {
        if lambda[i] > lambda[i + 1] {
            return false;
        }
    }
    true
}

pub fn sort_spectrum_samples(lambda: &mut [Float], values: &mut [Float]) {
    let mut pairs: Vec<(Float, Float)> =
        lambda.iter().copied().zip(values.iter().copied()).collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    for (i, (wavelength, value)) in pairs.into_iter().enumerate() {
        lambda[i] = wavelength;
        values[i] = value;
    }
}

pub fn interpolate_spectrum_samples(lambda: &[Float], values: &[Float], l: Float) -> Float {
    debug_assert!(!lambda.is_empty());
    debug_assert_eq!(lambda.len(), values.len());

    if l <= lambda[0] {
        return values[0];
    }
    let last = lambda.len() - 1;
    if l >= lambda[last] {
        return values[last];
    }

    let mut low = 0usize;
    let mut high = last;
    while high - low > 1 {
        let mid = (low + high) / 2;
        if lambda[mid] <= l {
            low = mid;
        } else {
            high = mid;
        }
    }

    let t = (l - lambda[low]) / (lambda[low + 1] - lambda[low]);
    lerp(t, values[low], values[low + 1])
}

pub fn rgb_to_xyz(rgb: &[Float; 3]) -> [Float; 3] {
    [
        0.412453 * rgb[0] + 0.357580 * rgb[1] + 0.180423 * rgb[2],
        0.212671 * rgb[0] + 0.715160 * rgb[1] + 0.072169 * rgb[2],
        0.019334 * rgb[0] + 0.119193 * rgb[1] + 0.950227 * rgb[2],
    ]
}
