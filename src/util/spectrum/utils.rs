include!(concat!(env!("OUT_DIR"), "/spectrum_utils.rs"));

use crate::util::base::*;

pub fn spectrum_samples_sorted(lambda: &[Float], _vals: &[Float]) -> bool {
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

pub fn sort_spectrum_samples(lambda: &mut [Float], vals: &mut [Float]) {
    let n = lambda.len();
    let mut sort_vec: Vec<(Float, Float)> = Vec::with_capacity(n);
    for i in 0..n {
        sort_vec.push((lambda[i], vals[i]));
    }
    sort_vec.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
    for i in 0..n {
        lambda[i] = sort_vec[i].0;
        vals[i] = sort_vec[i].1;
    }
}

pub fn interpolate_spectrum_samples(lambda: &[Float], vals: &[Float], l: Float) -> Float {
    let n = lambda.len();
    if l <= lambda[0] {
        return vals[0];
    }
    if l >= lambda[n - 1] {
        return vals[n - 1];
    }
    let offset = find_interval(lambda, &|v, index| -> bool { v[index] <= l });
    let t = (l - lambda[offset]) / (lambda[offset + 1] - lambda[offset]);
    lerp(t, vals[offset], vals[offset + 1])
}

pub fn resample_linear_spectrum(
    lambda_in: &[Float],
    v_in: &[Float],
    lambda_min: Float,
    lambda_max: Float,
    v_out: &mut [Float],
) {
    let n_in = lambda_in.len();
    let n_out = v_out.len();
    assert!(n_out > 2);
    for i in 0..(n_in - 1) {
        assert!(lambda_in[i] <= lambda_in[i + 1]);
    }
    assert!(lambda_min < lambda_max);

    let delta = (lambda_max - lambda_min) / (n_out - 1) as Float;

    let lambda_in_clamped = |index: i32| -> Float {
        assert!(index >= -1 && index <= n_in as i32);
        if index == -1 {
            lambda_min - delta
        } else if index == n_in as i32 {
            lambda_max + delta
        } else {
            lambda_in[index as usize]
        }
    };

    let v_in_clamped = |index: i32| -> Float {
        assert!(index >= -1 && index <= n_in as i32);
        v_in[index.clamp(0, n_in as i32 - 1) as usize]
    };

    let resample = |lambda: Float| -> Float {
        if lambda + delta / 2.0 <= lambda_in[0] {
            return v_in[0];
        }
        if lambda - delta / 2.0 >= lambda_in[n_in - 1] {
            return v_in[n_in - 1];
        }
        if n_in == 1 {
            return v_in[0];
        }

        let start;
        let mut end;
        if lambda - delta < lambda_in[0] {
            start = -1;
        } else {
            start = find_interval(lambda_in, &|v, index| -> bool {
                v[index] <= lambda - delta
            }) as i32;
        }

        if lambda + delta > lambda_in[n_in - 1] {
            end = n_in as i32;
        } else {
            end = if start > 0 { start } else { 0 };
            while end < n_in as i32 && lambda + delta > lambda_in[end as usize] {
                end += 1;
            }
        }

        if end - start == 2
            && lambda_in_clamped(start) <= lambda - delta
            && lambda_in[(start + 1) as usize] == lambda
            && lambda_in_clamped(end) >= lambda + delta
        {
            return v_in[(start + 1) as usize];
        } else if end - start == 1 {
            let t = (lambda - lambda_in_clamped(start))
                / (lambda_in_clamped(end) - lambda_in_clamped(start));
            return lerp(t, v_in_clamped(start), v_in_clamped(end));
        }

        average_spectrum_samples(lambda_in, v_in, lambda - delta / 2.0, lambda + delta / 2.0)
    };

    for out_offset in 0..n_out {
        let lambda = lerp(
            out_offset as Float / (n_out as Float - 1.0),
            lambda_min,
            lambda_max,
        );
        v_out[out_offset] = resample(lambda);
    }
}
