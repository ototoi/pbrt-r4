use crate::util::error::PbrtError;

use super::data::{
    DATASETS, LIMB_DARKENING_DATASETS, RADIANCE_DATASETS, SOLAR_DATASETS, SPECTRAL_WAVELENGTHS,
};

const PI: f64 = std::f64::consts::PI;
const SOLAR_RADIUS: f64 = (0.51 * PI / 180.0) / 2.0;
const SPECTRAL_BANDS: usize = 11;
const CONFIGURATION_COEFFICIENTS: usize = 9;
const SOLAR_ELEVATION_PIECES: usize = 45;
const SOLAR_ELEVATION_ORDER: usize = 4;

pub struct HosekSkyModel {
    configs: [[f64; CONFIGURATION_COEFFICIENTS]; SPECTRAL_BANDS],
    radiances: [f64; SPECTRAL_BANDS],
    emission_correction_factor_sun: [f64; SPECTRAL_BANDS],
    emission_correction_factor_sky: [f64; SPECTRAL_BANDS],
    turbidity: f64,
    solar_radius: f64,
}

impl HosekSkyModel {
    pub fn new(
        solar_elevation: f64,
        atmospheric_turbidity: f64,
        ground_albedo: f64,
    ) -> Result<Self, PbrtError> {
        if !(0.0..=PI / 2.0).contains(&solar_elevation) {
            return Err(PbrtError::error(
                "Hosek-Wilkie solar elevation must be between 0 and pi/2 radians",
            ));
        }
        if !(1.0..=10.0).contains(&atmospheric_turbidity) {
            return Err(PbrtError::error(
                "Hosek-Wilkie turbidity must be between 1 and 10",
            ));
        }
        if !(0.0..=1.0).contains(&ground_albedo) {
            return Err(PbrtError::error(
                "Hosek-Wilkie ground albedo must be between 0 and 1",
            ));
        }

        let mut model = Self {
            configs: [[0.0; CONFIGURATION_COEFFICIENTS]; SPECTRAL_BANDS],
            radiances: [0.0; SPECTRAL_BANDS],
            emission_correction_factor_sun: [1.0; SPECTRAL_BANDS],
            emission_correction_factor_sky: [1.0; SPECTRAL_BANDS],
            turbidity: atmospheric_turbidity,
            solar_radius: SOLAR_RADIUS,
        };

        for band in 0..SPECTRAL_BANDS {
            model.configs[band] = cook_configuration(
                DATASETS[band],
                atmospheric_turbidity,
                ground_albedo,
                solar_elevation,
            );
            model.radiances[band] = cook_radiance_configuration(
                RADIANCE_DATASETS[band],
                atmospheric_turbidity,
                ground_albedo,
                solar_elevation,
            );
        }

        Ok(model)
    }

    pub fn radiance(&self, theta: f64, gamma: f64, wavelength: f64) -> f64 {
        let low_wavelength = ((wavelength - 320.0) / 40.0) as isize;
        if !(0..SPECTRAL_BANDS as isize).contains(&low_wavelength) {
            return 0.0;
        }

        let low_wavelength = low_wavelength as usize;
        let interpolation = (wavelength - 320.0).rem_euclid(40.0) / 40.0;
        let value_low = self.radiance_at_band(theta, gamma, low_wavelength);
        if interpolation < 1e-6 || low_wavelength + 1 == SPECTRAL_BANDS {
            return value_low;
        }

        (1.0 - interpolation) * value_low
            + interpolation * self.radiance_at_band(theta, gamma, low_wavelength + 1)
    }

    pub fn solar_radiance(
        &self,
        theta: f64,
        gamma: f64,
        wavelength: f64,
    ) -> Result<f64, PbrtError> {
        if !(320.0..=720.0).contains(&wavelength) {
            return Err(PbrtError::error(
                "Hosek-Wilkie wavelength must be between 320 and 720 nm",
            ));
        }

        let direct = self.solar_radiance_internal(wavelength, PI / 2.0 - theta, gamma);
        Ok(direct + self.radiance(theta, gamma, wavelength))
    }

    pub fn spectral_wavelengths() -> &'static [f64; SPECTRAL_BANDS] {
        &SPECTRAL_WAVELENGTHS
    }

    fn radiance_at_band(&self, theta: f64, gamma: f64, band: usize) -> f64 {
        get_radiance_internal(&self.configs[band], theta, gamma)
            * self.radiances[band]
            * self.emission_correction_factor_sky[band]
    }

    fn solar_radiance_internal(&self, wavelength: f64, elevation: f64, gamma: f64) -> f64 {
        let solar_radius_sine = self.solar_radius.sin();
        let radius_squared_inverse = 1.0 / (solar_radius_sine * solar_radius_sine);
        let sine_gamma = gamma.sin();
        let mut sample_cosine_squared = 1.0 - radius_squared_inverse * sine_gamma * sine_gamma;
        if sample_cosine_squared < 0.0 {
            sample_cosine_squared = 0.0;
        }
        let sample_cosine = sample_cosine_squared.sqrt();
        if sample_cosine == 0.0 {
            return 0.0;
        }

        let mut turbidity_low = self.turbidity as usize - 1;
        let mut turbidity_fraction = self.turbidity - (turbidity_low + 1) as f64;
        if turbidity_low == 9 {
            turbidity_low = 8;
            turbidity_fraction = 1.0;
        }

        let mut wavelength_low = ((wavelength - 320.0) / 40.0) as usize;
        let mut wavelength_fraction = wavelength.rem_euclid(40.0) / 40.0;
        if wavelength_low == 10 {
            wavelength_low = 9;
            wavelength_fraction = 1.0;
        }

        let low_turbidity = (1.0 - wavelength_fraction)
            * solar_radiance_at_band(self, turbidity_low, wavelength_low, elevation)
            + wavelength_fraction
                * solar_radiance_at_band(self, turbidity_low, wavelength_low + 1, elevation);
        let high_turbidity = (1.0 - wavelength_fraction)
            * solar_radiance_at_band(self, turbidity_low + 1, wavelength_low, elevation)
            + wavelength_fraction
                * solar_radiance_at_band(self, turbidity_low + 1, wavelength_low + 1, elevation);
        let mut direct_radiance =
            (1.0 - turbidity_fraction) * low_turbidity + turbidity_fraction * high_turbidity;

        let low_limb = LIMB_DARKENING_DATASETS[wavelength_low];
        let high_limb = LIMB_DARKENING_DATASETS[wavelength_low + 1];
        let mut darkening_factor = 0.0;
        for coefficient in 0..6 {
            let limb = (1.0 - wavelength_fraction) * low_limb[coefficient]
                + wavelength_fraction * high_limb[coefficient];
            darkening_factor += limb * sample_cosine.powi(coefficient as i32);
        }
        direct_radiance *= darkening_factor;
        direct_radiance + 0.0
    }
}

fn cook_configuration(
    dataset: &[f64],
    turbidity: f64,
    albedo: f64,
    solar_elevation: f64,
) -> [f64; CONFIGURATION_COEFFICIENTS] {
    let turbidity_integer = turbidity as usize;
    let turbidity_fraction = turbidity - turbidity_integer as f64;
    let elevation = (solar_elevation / (PI / 2.0)).powf(1.0 / 3.0);
    let mut configuration = [0.0; CONFIGURATION_COEFFICIENTS];

    for (weight, offset, albedo_weight) in [
        (
            1.0 - turbidity_fraction,
            9 * 6 * (turbidity_integer - 1),
            1.0 - albedo,
        ),
        (
            1.0 - turbidity_fraction,
            9 * 6 * 10 + 9 * 6 * (turbidity_integer - 1),
            albedo,
        ),
        (turbidity_fraction, 9 * 6 * turbidity_integer, 1.0 - albedo),
        (
            turbidity_fraction,
            9 * 6 * 10 + 9 * 6 * turbidity_integer,
            albedo,
        ),
    ] {
        if weight == 0.0 {
            continue;
        }
        for coefficient in 0..CONFIGURATION_COEFFICIENTS {
            configuration[coefficient] += weight
                * albedo_weight
                * elevation_polynomial(dataset, offset + coefficient, elevation);
        }
    }
    configuration
}

fn elevation_polynomial(dataset: &[f64], offset: usize, elevation: f64) -> f64 {
    let one_minus_elevation = 1.0 - elevation;
    one_minus_elevation.powi(5) * dataset[offset]
        + 5.0 * one_minus_elevation.powi(4) * elevation * dataset[offset + 9]
        + 10.0 * one_minus_elevation.powi(3) * elevation.powi(2) * dataset[offset + 18]
        + 10.0 * one_minus_elevation.powi(2) * elevation.powi(3) * dataset[offset + 27]
        + 5.0 * one_minus_elevation * elevation.powi(4) * dataset[offset + 36]
        + elevation.powi(5) * dataset[offset + 45]
}

fn cook_radiance_configuration(
    dataset: &[f64],
    turbidity: f64,
    albedo: f64,
    solar_elevation: f64,
) -> f64 {
    let turbidity_integer = turbidity as usize;
    let turbidity_fraction = turbidity - turbidity_integer as f64;
    let elevation = (solar_elevation / (PI / 2.0)).powf(1.0 / 3.0);
    let mut result = 0.0;

    for (weight, offset, albedo_weight) in [
        (
            1.0 - turbidity_fraction,
            6 * (turbidity_integer - 1),
            1.0 - albedo,
        ),
        (
            1.0 - turbidity_fraction,
            6 * 10 + 6 * (turbidity_integer - 1),
            albedo,
        ),
        (turbidity_fraction, 6 * turbidity_integer, 1.0 - albedo),
        (turbidity_fraction, 6 * 10 + 6 * turbidity_integer, albedo),
    ] {
        if weight == 0.0 {
            continue;
        }
        result +=
            weight * albedo_weight * elevation_polynomial_radiance(dataset, offset, elevation);
    }
    result
}

fn elevation_polynomial_radiance(dataset: &[f64], offset: usize, elevation: f64) -> f64 {
    let one_minus_elevation = 1.0 - elevation;
    one_minus_elevation.powi(5) * dataset[offset]
        + 5.0 * one_minus_elevation.powi(4) * elevation * dataset[offset + 1]
        + 10.0 * one_minus_elevation.powi(3) * elevation.powi(2) * dataset[offset + 2]
        + 10.0 * one_minus_elevation.powi(2) * elevation.powi(3) * dataset[offset + 3]
        + 5.0 * one_minus_elevation * elevation.powi(4) * dataset[offset + 4]
        + elevation.powi(5) * dataset[offset + 5]
}

fn get_radiance_internal(
    configuration: &[f64; CONFIGURATION_COEFFICIENTS],
    theta: f64,
    gamma: f64,
) -> f64 {
    let cosine_gamma = gamma.cos();
    let exp_m = (configuration[4] * gamma).exp();
    let ray_m = cosine_gamma * cosine_gamma;
    let mie_m = (1.0 + ray_m)
        / (1.0 + configuration[8] * configuration[8] - 2.0 * configuration[8] * cosine_gamma)
            .powf(1.5);
    let zenith = theta.cos().sqrt();

    (1.0 + configuration[0] * (configuration[1] / (theta.cos() + 0.01)).exp())
        * (configuration[2]
            + configuration[3] * exp_m
            + configuration[5] * ray_m
            + configuration[6] * mie_m
            + configuration[7] * zenith)
}

fn solar_radiance_at_band(
    model: &HosekSkyModel,
    turbidity: usize,
    wavelength: usize,
    elevation: f64,
) -> f64 {
    let position =
        ((2.0 * elevation / PI).powf(1.0 / 3.0) * SOLAR_ELEVATION_PIECES as f64) as usize;
    let position = position.min(SOLAR_ELEVATION_PIECES - 1);
    let break_elevation = (position as f64 / SOLAR_ELEVATION_PIECES as f64).powi(3) * (PI * 0.5);
    let coefficient_end = SOLAR_ELEVATION_ORDER * SOLAR_ELEVATION_PIECES * turbidity
        + SOLAR_ELEVATION_ORDER * (position + 1)
        - 1;
    let coefficients = SOLAR_DATASETS[wavelength];
    let x = elevation - break_elevation;
    let mut result = 0.0;
    let mut x_power = 1.0;
    for index in 0..SOLAR_ELEVATION_ORDER {
        result += x_power * coefficients[coefficient_end - index];
        x_power *= x;
    }
    result * model.emission_correction_factor_sun[wavelength]
}
