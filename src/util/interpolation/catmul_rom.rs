use crate::util::base::*;

pub fn catmull_rom_weights(nodes: &[Float], x: Float) -> Option<(i32, [Float; 4])> {
    let size = nodes.len();
    // Return _false_ if _x_ is out of bounds
    if !(x >= nodes[0] && x <= nodes[size - 1]) {
        return None;
    }

    // Search for the interval _idx_ containing _x_
    let idx = find_interval(nodes, &|v, i| -> bool {
        return v[i] <= x;
    });

    assert!(idx < (size - 1));

    let offset = idx as i32 - 1;
    let x0 = nodes[idx];
    let x1 = nodes[idx + 1];

    assert!(x0 <= x);
    assert!(x <= x1, "{x} <= {x1}: [{x0}, {x}, {x1}], {idx}/{size}");

    // Compute the $t$ parameter and powers
    let t = (x - x0) / (x1 - x0);
    let t2 = t * t;
    let t3 = t2 * t;

    let mut weights = [0.0; 4];

    weights[1] = 2.0 * t3 - 3.0 * t2 + 1.0;
    weights[2] = -2.0 * t3 + 3.0 * t2;

    // Compute first node weight $w_0$
    if idx > 0 {
        let w0 = (t3 - 2.0 * t2 + t) * (x1 - x0) / (x1 - nodes[idx - 1]);
        weights[0] = -w0;
        weights[2] += w0;
    } else {
        let w0 = t3 - 2.0 * t2 + t;
        weights[0] = 0.0;
        weights[1] -= w0;
        weights[2] += w0;
    }

    // Compute last node weight $w_3$
    if idx + 2 < size {
        let w3 = (t3 - t2) * (x1 - x0) / (nodes[idx + 2] - x0);
        weights[1] -= w3;
        weights[3] = w3;
    } else {
        let w3 = t3 - t2;
        weights[1] -= w3;
        weights[2] += w3;
        weights[3] = 0.0;
    }

    return Some((offset, weights));
}

pub fn sample_catmull_rom_2d(
    nodes1: &[Float],
    nodes2: &[Float],
    values: &[Float],
    cdf: &[Float],
    alpha: Float,
    u: Float,
) -> Option<(Float, Float, Float)> {
    let (offset, weights) = catmull_rom_weights(nodes1, alpha)?;

    let size1 = nodes1.len();
    let size2 = nodes2.len();
    assert_eq!(values.len(), size1 * size2);
    assert_eq!(cdf.len(), size1 * size2);

    let interpolate = |array: &[Float], idx: usize| -> Float {
        assert!(idx < size2);
        assert!(array.len() == size1 * size2);
        let mut value = 0.0;
        for i in 0..4 {
            if weights[i as usize] != 0.0 {
                let y = i32::clamp(offset + i, 0, size1 as i32 - 1) as usize;
                value += array[y * size2 + idx] * weights[i as usize];
            }
        }
        return value;
    };

    let mut u = u;

    // Map _u_ to a spline interval by inverting the interpolated _cdf_
    let maximum = interpolate(cdf, size2 - 1);
    u *= maximum;
    let idx = find_interval_range((0, size2), cdf, &|v, i| {
        return interpolate(v, i) <= u;
    });

    assert!(interpolate(cdf, idx) <= u);
    assert!(u <= interpolate(cdf, idx + 1));

    assert!(idx < size2 - 1);

    let f0 = interpolate(values, idx);
    let f1 = interpolate(values, idx + 1);
    let x0 = nodes2[idx]; //
    let x1 = nodes2[idx + 1];
    let width = x1 - x0;

    // Re-scale _u_ using the interpolated _cdf_
    u = (u - interpolate(cdf, idx)) / width;

    // Approximate derivatives using finite differences of the interpolant
    let d0 = if idx > 0 {
        width * (f1 - interpolate(values, idx - 1)) / (x1 - nodes2[idx - 1])
    } else {
        f1 - f0
    };

    let d1 = if idx + 2 < size2 {
        width * (interpolate(values, idx + 2) - f0) / (nodes2[idx + 2] - x0)
    } else {
        f1 - f0
    };

    // Set initial guess for $t$ by importance sampling a linear interpolant
    let mut t = if f0 != f1 {
        (f0 - Float::sqrt(Float::max(0.0, f0 * f0 + 2.0 * u * (f1 - f0)))) / (f0 - f1)
    } else {
        u / f0
    };

    let mut a = 0.0;
    let mut b = 1.0;

    let mut fhat;

    loop {
        // Fall back to a bisection step when _t_ is out of bounds
        if !(t >= a && t <= b) {
            t = 0.5 * (a + b);
        }

        // Evaluate target function and its derivative in Horner form
        let t_fhat_ = t
            * (f0
                + t * (0.5 * d0
                    + t * ((1.0 / 3.0) * (-2.0 * d0 - d1) + f1 - f0
                        + t * (0.25 * (d0 + d1) + 0.5 * (f0 - f1)))));
        fhat = f0
            + t * (d0 + t * (-2.0 * d0 - d1 + 3.0 * (f1 - f0) + t * (d0 + d1 + 2.0 * (f0 - f1))));

        // Stop the iteration if converged
        if Float::abs(t_fhat_ - u) < 1e-6 || b - a < 1e-6 {
            break;
        }

        // Update bisection bounds using updated _t_
        if t_fhat_ - u < 0.0 {
            a = t;
        } else {
            b = t;
        }

        // Perform a Newton step
        t -= (t_fhat_ - u) / fhat;
    }

    let fval = fhat;
    let pdf = fhat / maximum;
    let value = x0 + width * t;
    return Some((value, fval, pdf));
}

pub fn integrate_catmull_rom(x: &[Float], values: &[Float]) -> Vec<Float> {
    let n = x.len();
    let mut cdf = vec![0.0; n];

    let mut sum = 0.0;
    for i in 0..(n - 1) {
        // Look up $x_i$ and function values of spline segment _i_
        let x0 = x[i];
        let x1 = x[i + 1];
        let f0 = values[i];
        let f1 = values[i + 1];
        let width = x1 - x0;

        // Approximate derivatives using finite differences
        let d0 = if i > 0 {
            width * (f1 - values[i - 1]) / (x1 - x[i - 1])
        } else {
            f1 - f0
        };

        let d1 = if (i + 2) < n {
            width * (values[i + 2] - f0) / (x[i + 2] - x0)
        } else {
            f1 - f0
        };
        // Keep a running sum and build a cumulative distribution function
        sum += ((d0 - d1) * (1.0 / 12.0) + (f0 + f1) * 0.5) * width;
        cdf[i + 1] = sum;
    }

    return cdf;
}

pub fn invert_catmull_rom(x: &[Float], values: &[Float], u: Float) -> Float {
    let n = values.len();
    // Stop when _u_ is out of bounds

    if u <= values[0] {
        return x[0];
    } else if values[n - 1] <= u {
        return x[n - 1];
    }

    // Map _u_ to a spline interval by inverting _values_
    let i = find_interval(values, &|v, i| -> bool {
        return v[i] <= u;
    });

    // Look up $x_i$ and function values of spline segment _i_
    let x0 = x[i];
    let x1 = x[i + 1];
    let f0 = values[i];
    let f1 = values[i + 1];
    let width = x1 - x0;

    // Approximate derivatives using finite differences
    let d0 = if i > 0 {
        width * (f1 - values[i - 1]) / (x1 - x[i - 1])
    } else {
        f1 - f0
    };

    let d1 = if i + 2 < n {
        width * (values[i + 2] - f0) / (x[i + 2] - x0)
    } else {
        f1 - f0
    };

    let mut a = 0.0;
    let mut b = 1.0;
    let mut t = 0.5;

    loop {
        // Fall back to a bisection step when _t_ is out of bounds
        if !(t >= a && t <= b) {
            t = 0.5 * (a + b);
        }

        // Compute powers of _t_
        let t2 = t * t;
        let t3 = t2 * t;

        // Set _Fhat_ using Equation (8.27)
        let t_fhat_ = (2.0 * t3 - 3.0 * t2 + 1.0) * f0
            + (-2.0 * t3 + 3.0 * t2) * f1
            + (t3 - 2.0 * t2 + t) * d0
            + (t3 - t2) * d1;

        // Set _fhat_ using Equation (not present)
        let fhat = (6.0 * t2 - 6.0 * t) * f0
            + (-6.0 * t2 + 6.0 * t) * f1
            + (3.0 * t2 - 4.0 * t + 1.0) * d0
            + (3.0 * t2 - 2.0 * t) * d1;

        // Stop the iteration if converged
        if Float::abs(t_fhat_ - u) < 1e-6 || b - a < 1e-6 {
            break;
        }

        // Update bisection bounds using updated _t_
        if t_fhat_ - u < 0.0 {
            a = t;
        } else {
            b = t;
        }

        // Perform a Newton step
        t -= (t_fhat_ - u) / fhat;
    }

    return x0 + t * width;
}
