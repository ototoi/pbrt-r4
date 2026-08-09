use super::config::*;

#[inline]
fn lerp(t: Float, v1: Float, v2: Float) -> Float {
    return (1.0 - t) * v1 + t * v2;
}

pub fn average_spectrum_samples(
    lambda: &[Float],
    vals: &[Float],
    lambda_start: Float,
    lambda_end: Float,
) -> Float {
    let n = lambda.len();
    for i in 0..n - 1 {
        assert!(lambda[i] <= lambda[i + 1]);
    }
    assert!(lambda_start <= lambda_end);

    // Handle cases with out-of-bounds range or single sample only
    if lambda_end as Float <= lambda[0] {
        return vals[0];
    }
    if lambda_start as Float >= lambda[n - 1] {
        return vals[n - 1];
    }
    if n == 1 {
        return vals[0];
    }
    let mut sum: Float = 0.0;
    // Add contributions of constant segments before/after samples
    if lambda_start < lambda[0] {
        sum += vals[0] * (lambda[0] - lambda_start);
    }
    if lambda_end > lambda[n - 1] {
        sum += vals[n - 1] * (lambda_end - lambda[n - 1]);
    }

    // Advance to first relevant wavelength segment
    let mut i = 0;
    while lambda_start > lambda[i + 1] {
        i += 1;
    }
    assert!((i + 1) <= n);

    // Loop over wavelength sample segments and add contributions
    let interp = |w: Float, i: usize| {
        return lerp(
            (w - lambda[i]) / (lambda[i + 1] - lambda[i]),
            vals[i],
            vals[i + 1],
        );
    };
    while i + 1 < n && lambda_end >= lambda[i] {
        let seg_lambda_start = Float::max(lambda_start, lambda[i]);
        let seg_lambda_end = Float::min(lambda_end, lambda[i + 1]);
        sum += 0.5
            * (interp(seg_lambda_start, i) + interp(seg_lambda_end, i))
            * (seg_lambda_end - seg_lambda_start);
        i += 1;
    }
    return sum / (lambda_end - lambda_start);
}

pub fn sample_spectrum(lambda: &[Float], vals: &[Float]) -> [Float; N_SPECTRUM_SAMPLES] {
    let mut x: [Float; N_SPECTRUM_SAMPLES] = [0.0; N_SPECTRUM_SAMPLES];
    for i in 0..N_SPECTRUM_SAMPLES {
        let wl0 = lerp(
            (i as Float) / (N_SPECTRUM_SAMPLES as Float),
            LAMBDA_MIN as Float,
            LAMBDA_MAX as Float,
        );
        let wl1 = lerp(
            ((i + 1) as Float) / (N_SPECTRUM_SAMPLES as Float),
            LAMBDA_MIN as Float,
            LAMBDA_MAX as Float,
        );
        x[i] = average_spectrum_samples(lambda, vals, wl0, wl1);
    }
    return x;
}
