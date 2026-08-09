// Imported from spectrum.cpp

use pbrt_r4::prelude::*;
use pbrt_r4::util::spectrum::utils::*;

fn near_equal(a: Float, b: Float, e: Float) -> bool {
    (a - b).abs() < e
}

#[test]
fn spectrum_blackbody() {
    // Relative error.
    let err = |a: f64, b: f64| (a - b).abs() / b;

    // Planck's law.
    // A few values via
    // http://www.spectralcalc.com/blackbody_calculator/blackbody.php
    // lambda, T, expected radiance
    let v = [
        (483, 6000, 3.1849e13),
        (600, 6000, 2.86772e13),
        (500, 3700, 1.59845e12),
        (600, 4500, 7.46497e12),
    ];
    for i in 0..v.len() {
        let (lambda, t, expected) = v[i];
        let le = blackbody::blackbody(&[lambda as f64], t as f64)[0];
        assert!(err(le, expected as f64) < 0.001);
    }

    // Use Wien's displacement law to compute maximum wavelength for a few
    // temperatures, then confirm that the value returned by Blackbody() is
    // consistent with this.
    for t in [2700, 3000, 4500, 5600, 6000] {
        let t = t as f64;
        let lambda_max = 2.8977721e-3 / t * 1e9;
        let lambda = [0.999 * lambda_max, lambda_max, 1.001 * lambda_max];
        let le = blackbody::blackbody(&lambda, t);
        assert!(le[0] < le[1]);
        assert!(le[1] > le[2]);
    }
}

#[test]
fn spectrum_linear_upsample_subset() {
    // Linear SPD where val(lambda) = 1 + 2 * lambda.
    let lambda = [0.0, 1.0, 2.0, 3.0, 4.0];
    let val = [1.0, 3.0, 5.0, 7.0, 9.0];
    assert_eq!(lambda.len(), val.len());

    // Resample at exactly the same rate, but over a subset of the input
    // samples; should give back exactly the original values for them.
    let mut new_val = [0.0; 3];
    resample_linear_spectrum(&lambda, &val, lambda[1], lambda[3], &mut new_val);
    for i in 0..3 {
        assert_eq!(val[i + 1], new_val[i]);
    }
}

#[test]
fn spectrum_linear_upsample_2x() {
    // Linear SPD where val(lambda) = 1 + 2 * lambda.
    let lambda = [0.0, 1.0, 2.0, 3.0, 4.0];
    let val = [1.0, 3.0, 5.0, 7.0, 9.0];
    assert_eq!(lambda.len(), val.len());

    // Resample at 2x the rate, but same endpoints. Should exactly
    // reproduce the linear function.
    let mut new_val = [0.0; 9];
    resample_linear_spectrum(&lambda, &val, lambda[0], lambda[4], &mut new_val);
    for i in 0..9 {
        assert_eq!((i + 1) as Float, new_val[i]);
    }
}

#[test]
fn spectrum_linear_upsample_higher() {
    // Linear SPD where val(lambda) = 1 + 2 * lambda.
    let lambda = [0.0, 1.0, 2.0, 3.0, 4.0];
    let val = [1.0, 3.0, 5.0, 7.0, 9.0];
    assert_eq!(lambda.len(), val.len());

    // Higher sampling rate, at subset of lambdas; should reproduce the
    // linear function (modulo floating-point roundoff error).
    let mut new_val = [0.0; 20];
    resample_linear_spectrum(&lambda, &val, lambda[1], lambda[3], &mut new_val);
    for i in 0..20 {
        let t = i as Float / (20 - 1) as Float;
        assert!(near_equal(lerp(t, val[1], val[3]), new_val[i], 1e-6));
    }
}

#[test]
fn spectrum_linear_upsampling() {
    // Linear SPD where val(lambda) = 1 + 2 * lambda.
    let lambda = [0.0, 1.0, 2.0, 3.0, 4.0];
    let val = [1.0, 3.0, 5.0, 7.0, 9.0];
    assert_eq!(lambda.len(), val.len());

    // Higher sampling rate, at subset of lambdas; should reproduce the
    // linear function (modulo floating-point roundoff error).
    let mut new_val = [0.0; 40];
    let lambda_min = 1.5;
    let lambda_max = 3.75;
    resample_linear_spectrum(&lambda, &val, lambda_min, lambda_max, &mut new_val);
    for i in 0..40 {
        let t = i as Float / (40 - 1) as Float;
        assert!(near_equal(
            lerp(t, 1.0 + 2.0 * lambda_min, 1.0 + 2.0 * lambda_max),
            new_val[i],
            1e-6
        ));
    }
}

#[test]
fn spectrum_linear_irregular_resample() {
    // Irregularly-sampled SPD, where f(lambda) = lambda^2.
    let lambda = [
        -1.5, -0.5, 0.01, 0.6, 1.0, 2.0, 2.1, 3.4, 4.6, 5.7, 7.0, 8.2, 9.0, 9.8, 11.11, 12.0, 13.0,
        14.7,
    ];
    let val = lambda.iter().map(|&l| l * l).collect::<Vec<Float>>();
    assert_eq!(lambda.len(), val.len());

    // Resample it over a subset of the wavelengths.
    let mut new_val = [0.0; 30];
    let lambda_min = -0.5;
    let lambda_max = 14.0;
    resample_linear_spectrum(&lambda, &val, lambda_min, lambda_max, &mut new_val);

    // The result should be generally close to lambda^2, though there will
    // be some differences due to the fact that in the places where we are
    // downsampling, we're averaging over a range of the quadratic
    // function.
    for i in 0..30 {
        let t = i as Float / (30 - 1) as Float;
        let lambda = lerp(t, lambda_min, lambda_max);
        assert!(lambda * lambda - new_val[i] < 0.75);
    }
}

#[test]
fn spectrum_linear_downsample_basic() {
    // Another linear SPD.
    let lambda = [0.0, 1.0, 2.0, 3.0, 4.0];
    let val = [1.0, 3.0, 5.0, 7.0, 9.0];
    assert_eq!(lambda.len(), val.len());

    // Resample it with the same endpoints but at a lower sampling rate.
    let mut new_val = [0.0; 3];
    let lambda_min = 0.0;
    let lambda_max = 4.0;
    resample_linear_spectrum(&lambda, &val, lambda_min, lambda_max, &mut new_val);

    // We expect the computed values to be the averages over lambda=[-1,1],
    // then [1,3], then [3,5]. For the endpoints, recall that we model the
    // SPD beyond the specified endpoints as constant.
    assert!(near_equal(1.5, new_val[0], 1e-6));
    assert!(near_equal(5.0, new_val[1], 1e-6));
    assert!(near_equal(8.5, new_val[2], 1e-6));
}

#[test]
fn spectrum_linear_downsample_offset() {
    // Another linear SPD.
    let lambda = [0.0, 1.0, 2.0, 3.0, 4.0];
    let val = [1.0, 3.0, 5.0, 7.0, 9.0];
    assert_eq!(lambda.len(), val.len());

    let mut new_val = [0.0; 4];
    let lambda_min = 0.5;
    let lambda_max = 3.5;
    resample_linear_spectrum(&lambda, &val, lambda_min, lambda_max, &mut new_val);

    // The spacing between samples in the destination SPD is 1, so we expect
    // to get back averages over [0,1], [1,2], [2,3], and [3,4].
    assert!(near_equal(2.0, new_val[0], 1e-6));
    assert!(near_equal(4.0, new_val[1], 1e-6));
    assert!(near_equal(6.0, new_val[2], 1e-6));
    assert!(near_equal(8.0, new_val[3], 1e-6));
}

#[test]
fn spectrum_linear_downsample_irreg() {
    // Generate a very irregular set of lambda values starting at -25, where
    // the SPD is f(lambda) = lambda^2.
    let mut rng = RNG::new();
    let mut lambda = vec![-25.0];
    for _ in 0..100 {
        lambda.push(lambda.last().unwrap() + rng.uniform_float());
    }
    let val = lambda.iter().map(|&l| l * l).collect::<Vec<Float>>();
    assert_eq!(lambda.len(), val.len());

    // Resample it over a subset of the wavelengths.
    let mut new_val = [0.0; 10];
    let lambda_min = -5.0;
    let lambda_max = 20.0;
    resample_linear_spectrum(&lambda, &val, lambda_min, lambda_max, &mut new_val);

    // As with the IrregularResample test, we need a enough of an error
    // tolerance so that we account for the averaging over ranges of the
    // quadratic function.
    for i in 0..10 {
        let t = i as Float / (10 - 1) as Float;
        let lambda = lerp(t, lambda_min, lambda_max);
        let val_delta = (lambda * lambda - new_val[i]).abs();
        assert!(val_delta < 0.8);
    }
}
