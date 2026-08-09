use crate::util::base::*;

// Fourier Interpolation Definitions
pub fn evaluate_fourier(a: &[Float], cos_phi: Float) -> Float {
    let mut value = 0.0;
    // Initialize cosine iterates
    let mut cos_kminus_one_phi = cos_phi;
    let mut cos_kphi = 1.0;
    for k in 0..a.len() {
        // Add the current summand and update the cosine iterates
        value += a[k] * cos_kphi;
        let cos_kplus_one_phi = 2.0 * cos_phi * cos_kphi - cos_kminus_one_phi;
        cos_kminus_one_phi = cos_kphi;
        cos_kphi = cos_kplus_one_phi;
    }
    return value;
}

pub fn sample_fourier(ak: &[Float], recip: &[Float], u: Float) -> (Float, Float, Float) {
    let mut u = u;
    // Pick a side and declare bisection variables
    let flip = u >= 0.5;
    if flip {
        u = 1.0 - 2.0 * (u - 0.5);
    } else {
        u *= 2.0;
    }
    let mut a: f64 = 0.0;
    let mut b: f64 = PI as f64;
    let mut phi: f64 = 0.5 * PI as f64;
    let m = ak.len();

    let mut f;
    loop {
        // Evaluate $F(\phi)$ and its derivative $f(\phi)$

        // Initialize sine and cosine iterates
        let cos_phi = f64::cos(phi);
        let sin_phi = f64::sqrt(f64::max(0.0, 1.0 - cos_phi * cos_phi));
        let mut cos_phi_prev = cos_phi;
        let mut cos_phi_cur = 1.0;
        let mut sin_phi_prev = -sin_phi;
        let mut sin_phi_cur = 0.0;

        // Initialize _F_ and _f_ with the first series term
        let mut tf_ = ak[0] as f64 * phi; // F
        f = ak[0] as f64;
        for k in 1..m {
            // Compute next sine and cosine iterates
            let sin_phi_next = 2.0 * cos_phi * sin_phi_cur - sin_phi_prev;
            let cos_phi_next = 2.0 * cos_phi * cos_phi_cur - cos_phi_prev;
            sin_phi_prev = sin_phi_cur;
            sin_phi_cur = sin_phi_next;
            cos_phi_prev = cos_phi_cur;
            cos_phi_cur = cos_phi_next;

            // Add the next series term to _F_ and _f_
            tf_ += (ak[k] * recip[k]) as f64 * sin_phi_next;
            f += ak[k] as f64 * cos_phi_next;
        }
        tf_ -= (u * ak[0] * PI) as f64;

        // Update bisection bounds using updated $\phi$
        if tf_ > 0.0 {
            b = phi;
        } else {
            a = phi;
        }

        // Stop the Fourier bisection iteration if converged
        if f64::abs(tf_) < 1e-6 || b - a < 1e-6 {
            break;
        }

        // Perform a Newton step given $f(\phi)$ and $F(\phi)$
        phi -= tf_ / f;

        // Fall back to a bisection step when $\phi$ is out of bounds
        if !(phi > a && phi < b) {
            phi = 0.5 * (a + b);
        }
    }
    // Potentially flip $\phi$ and return the result
    if flip {
        phi = (2.0 * PI) as f64 - phi;
    }
    let pdf = INV_2_PI / ak[0] * f as Float;
    return (f as Float, pdf, phi as Float);
}
