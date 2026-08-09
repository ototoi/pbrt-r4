use crate::base::bxdf::*;
use crate::util::base::*;
use crate::util::geometry::*;
use crate::util::sampling::PiecewiseLinear2D;
use crate::util::scattering::{abs_cos_theta, cos_theta, reflect, same_hemisphere};
use crate::util::spectrum::*;
use crate::util::tensor::{TensorFile, TensorType};

use std::sync::Arc;

// --- coordinate maps shared with pbrt-v4 MeasuredBxDF ------------------
#[inline]
fn theta2u(theta: Float) -> Float {
    (theta * (2.0 / PI)).sqrt()
}
#[inline]
fn phi2u(phi: Float) -> Float {
    phi * (1.0 / (2.0 * PI)) + 0.5
}
#[inline]
fn u2theta(u: Float) -> Float {
    u * u * (PI / 2.0)
}
#[inline]
fn u2phi(u: Float) -> Float {
    (2.0 * u - 1.0) * PI
}

/// Tabulated measured-BRDF data loaded from a `.bsdf` tensor file.
#[derive(Debug, Clone)]
pub struct MeasuredBxDFData {
    pub filename: String,
    pub loaded: bool,
    pub isotropic: bool,
    pub wavelengths: Vec<f32>,
    pub interpolants: Option<MeasuredInterpolants>,
}

/// Bundle of the five v4-shape PiecewiseLinear2D interpolants the
/// measured BRDF math needs.
#[derive(Debug, Clone)]
pub struct MeasuredInterpolants {
    pub ndf: Arc<PiecewiseLinear2D<0>>,
    pub sigma: Arc<PiecewiseLinear2D<0>>,
    pub vndf: Arc<PiecewiseLinear2D<2>>,
    pub luminance: Arc<PiecewiseLinear2D<2>>,
    pub spectra: Arc<PiecewiseLinear2D<3>>,
}

impl MeasuredBxDFData {
    pub fn new(filename: String) -> Self {
        Self {
            filename,
            loaded: false,
            isotropic: true,
            wavelengths: Vec::new(),
            interpolants: None,
        }
    }

    pub fn from_file(filename: &str) -> Arc<Self> {
        if filename.is_empty() {
            return Arc::new(Self::new(String::new()));
        }
        match TensorFile::open(filename) {
            Ok(tf) => match Self::from_tensor(filename, tf) {
                Ok(data) => Arc::new(data),
                Err(e) => {
                    log::warn!(
                        "Material \"measured\": failed to interpret \"{}\": {} -- using empty data",
                        filename,
                        e
                    );
                    Arc::new(Self::new(filename.to_string()))
                }
            },
            Err(e) => {
                log::warn!(
                    "Material \"measured\": failed to load \"{}\": {} -- using empty data",
                    filename,
                    e
                );
                Arc::new(Self::new(filename.to_string()))
            }
        }
    }

    fn from_tensor(filename: &str, tf: TensorFile) -> Result<Self, String> {
        let f32_field = |name: &str| -> Result<&[f32], String> {
            let f = tf
                .field(name)
                .ok_or_else(|| format!("missing field `{}`", name))?;
            if f.dtype != TensorType::Float32 {
                return Err(format!("field `{}` must be Float32", name));
            }
            f.as_f32_slice()
                .ok_or_else(|| format!("field `{}` is not contiguous Float32", name))
        };
        let field = |name: &str| {
            tf.field(name)
                .ok_or_else(|| format!("missing field `{}`", name))
        };

        let theta_i = f32_field("theta_i")?;
        let phi_i = f32_field("phi_i")?;
        let wavelengths = f32_field("wavelengths")?.to_vec();

        let ndf_f = field("ndf")?;
        if ndf_f.shape.len() != 2 {
            return Err("`ndf` must be 2D".into());
        }
        let sigma_f = field("sigma")?;
        if sigma_f.shape.len() != 2 {
            return Err("`sigma` must be 2D".into());
        }
        let vndf_f = field("vndf")?;
        if vndf_f.shape.len() != 4
            || vndf_f.shape[0] != phi_i.len()
            || vndf_f.shape[1] != theta_i.len()
        {
            return Err("`vndf` shape must be [phi_i, theta_i, ny, nx]".into());
        }
        let luminance_f = field("luminance")?;
        if luminance_f.shape.len() != 4
            || luminance_f.shape[0] != phi_i.len()
            || luminance_f.shape[1] != theta_i.len()
            || luminance_f.shape[2] != luminance_f.shape[3]
        {
            return Err("`luminance` shape must be [phi_i, theta_i, n, n]".into());
        }
        let spectra_f = field("spectra")?;
        if spectra_f.shape.len() != 5
            || spectra_f.shape[0] != phi_i.len()
            || spectra_f.shape[1] != theta_i.len()
            || spectra_f.shape[2] != wavelengths.len()
            || spectra_f.shape[3] != spectra_f.shape[4]
        {
            return Err("`spectra` shape must be [phi_i, theta_i, wavelengths, n, n]".into());
        }
        if luminance_f.shape[2] != spectra_f.shape[3] || luminance_f.shape[3] != spectra_f.shape[4]
        {
            return Err("`luminance` and `spectra` must share inner (n, n) grid".into());
        }

        let isotropic = phi_i.len() <= 2;
        if !isotropic {
            let span = phi_i[phi_i.len() - 1] - phi_i[0];
            let reduction = (2.0 * std::f32::consts::PI / span).round() as i32;
            if reduction != 1 {
                return Err(format!(
                    "phi_i reduction {} (!= 1) not supported",
                    reduction
                ));
            }
        }

        let ndf_data = ndf_f.as_f32_slice().ok_or("ndf raw bytes not float32")?;
        let sigma_data = sigma_f
            .as_f32_slice()
            .ok_or("sigma raw bytes not float32")?;
        let vndf_data = vndf_f.as_f32_slice().ok_or("vndf raw bytes not float32")?;
        let luminance_data = luminance_f
            .as_f32_slice()
            .ok_or("luminance raw bytes not float32")?;
        let spectra_data = spectra_f
            .as_f32_slice()
            .ok_or("spectra raw bytes not float32")?;

        let ndf = PiecewiseLinear2D::<0>::new(
            ndf_data,
            ndf_f.shape[1],
            ndf_f.shape[0],
            [],
            [],
            false,
            false,
        );
        let sigma = PiecewiseLinear2D::<0>::new(
            sigma_data,
            sigma_f.shape[1],
            sigma_f.shape[0],
            [],
            [],
            false,
            false,
        );
        let vndf = PiecewiseLinear2D::<2>::new(
            vndf_data,
            vndf_f.shape[3],
            vndf_f.shape[2],
            [phi_i.len(), theta_i.len()],
            [phi_i, theta_i],
            true,
            true,
        );
        let luminance = PiecewiseLinear2D::<2>::new(
            luminance_data,
            luminance_f.shape[3],
            luminance_f.shape[2],
            [phi_i.len(), theta_i.len()],
            [phi_i, theta_i],
            true,
            true,
        );
        let spectra = PiecewiseLinear2D::<3>::new(
            spectra_data,
            spectra_f.shape[4],
            spectra_f.shape[3],
            [phi_i.len(), theta_i.len(), wavelengths.len()],
            [phi_i, theta_i, &wavelengths],
            false,
            false,
        );

        Ok(Self {
            filename: filename.to_string(),
            loaded: true,
            isotropic,
            wavelengths,
            interpolants: Some(MeasuredInterpolants {
                ndf: Arc::new(ndf),
                sigma: Arc::new(sigma),
                vndf: Arc::new(vndf),
                luminance: Arc::new(luminance),
                spectra: Arc::new(spectra),
            }),
        })
    }
}

/// Direct translation of pbrt-v4 `MeasuredBxDF` (`bxdfs.h:1021`).
#[derive(Debug, Clone)]
pub struct MeasuredBxDF {
    brdf: Arc<MeasuredBxDFData>,
    lambda: SampledWavelengths,
}

impl MeasuredBxDF {
    pub fn new(brdf: Arc<MeasuredBxDFData>, lambda: &SampledWavelengths) -> Self {
        Self {
            brdf,
            lambda: *lambda,
        }
    }

    pub fn data(&self) -> &MeasuredBxDFData {
        &self.brdf
    }

    /// pbrt-v4 `MeasuredBxDF::f`.
    pub fn f(&self, wo: &Vector3f, wi: &Vector3f, _mode: TransportMode) -> SampledSpectrum {
        let Some(interp) = self.brdf.interpolants.as_ref() else {
            return SampledSpectrum::zero();
        };
        if !same_hemisphere(wo, wi) {
            return SampledSpectrum::zero();
        }
        let (mut wo, mut wi) = (*wo, *wi);
        if wo.z < 0.0 {
            wo = -wo;
            wi = -wi;
        }

        let mut wm = wi + wo;
        if wm.length_squared() == 0.0 {
            return SampledSpectrum::zero();
        }
        wm = wm.normalize();

        let theta_o = spherical_theta(&wo);
        let phi_o = wo.y.atan2(wo.x);
        let theta_m = spherical_theta(&wm);
        let phi_m = wm.y.atan2(wm.x);

        let u_wo = Point2f::new(theta2u(theta_o), phi2u(phi_o));
        let mut u_wm = Point2f::new(
            theta2u(theta_m),
            phi2u(if self.brdf.isotropic {
                phi_m - phi_o
            } else {
                phi_m
            }),
        );
        u_wm.y -= u_wm.y.floor();

        let ui = interp.vndf.invert(u_wm, &[phi_o, theta_o]).p;

        let mut fr = SampledSpectrum::zero();
        for i in 0..SampledSpectrum::N_SAMPLES {
            fr[i] = interp
                .spectra
                .evaluate(ui, &[phi_o, theta_o, self.lambda[i] as Float])
                .max(0.0);
        }

        let ndf_val = interp.ndf.evaluate(u_wm, &[]);
        let sigma_val = interp.sigma.evaluate(u_wo, &[]).max(1e-6);
        let denom = 4.0 * sigma_val * cos_theta(&wi);
        if denom == 0.0 {
            return SampledSpectrum::zero();
        }
        fr * (ndf_val / denom)
    }

    /// pbrt-v4 `MeasuredBxDF::Sample_f`.
    pub fn sample_f(
        &self,
        wo: &Vector3f,
        _uc: Float,
        u: &Point2f,
        _mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Option<BSDFSample> {
        if sample_flags & BXDF_REFL_TRANS_REFLECTION == 0 {
            return None;
        }
        let interp = self.brdf.interpolants.as_ref()?;

        let mut wo = *wo;
        let mut flip_wi = false;
        if wo.z <= 0.0 {
            wo = -wo;
            flip_wi = true;
        }

        let theta_o = spherical_theta(&wo);
        let phi_o = wo.y.atan2(wo.x);

        let lum = interp.luminance.sample(*u, &[phi_o, theta_o]);
        let warped_u = lum.p;
        let lum_pdf = lum.pdf;

        let vndf = interp.vndf.sample(warped_u, &[phi_o, theta_o]);
        let u_wm = vndf.p;
        let pdf = vndf.pdf;

        let mut phi_m = u2phi(u_wm.y);
        let theta_m = u2theta(u_wm.x);
        if self.brdf.isotropic {
            phi_m += phi_o;
        }
        let sin_theta_m = theta_m.sin();
        let cos_theta_m = theta_m.cos();
        let wm = spherical_direction(sin_theta_m, cos_theta_m, phi_m);
        let wi = reflect(&wo, &wm);
        if wi.z <= 0.0 {
            return None;
        }

        let mut fr = SampledSpectrum::zero();
        for i in 0..SampledSpectrum::N_SAMPLES {
            fr[i] = interp
                .spectra
                .evaluate(warped_u, &[phi_o, theta_o, self.lambda[i] as Float])
                .max(0.0);
        }

        let u_wo = Point2f::new(theta2u(theta_o), phi2u(phi_o));
        let ndf_val = interp.ndf.evaluate(u_wm, &[]);
        let sigma_val = interp.sigma.evaluate(u_wo, &[]).max(1e-6);
        let abs_cos_wi = abs_cos_theta(&wi).max(1e-6);
        let fr_packet = fr * (ndf_val / (4.0 * sigma_val * abs_cos_wi));

        let dot_wo_wm = Vector3f::dot(&wo, &wm).abs().max(1e-6);
        let jacobian = 4.0 * dot_wo_wm * (2.0 * PI * PI * u_wm.x * sin_theta_m).max(1e-6);

        let wi_final = if flip_wi { -wi } else { wi };
        Some(BSDFSample::new(
            fr_packet,
            wi_final,
            pdf * lum_pdf / jacobian,
            BXDF_REFLECTION | BXDF_GLOSSY,
            1.0,
            false,
        ))
    }

    /// pbrt-v4 `MeasuredBxDF::PDF`.
    pub fn pdf(
        &self,
        wo: &Vector3f,
        wi: &Vector3f,
        _mode: TransportMode,
        sample_flags: BxDFReflTransFlags,
    ) -> Float {
        if sample_flags & BXDF_REFL_TRANS_REFLECTION == 0 {
            return 0.0;
        }
        let Some(interp) = self.brdf.interpolants.as_ref() else {
            return 0.0;
        };
        if !same_hemisphere(wo, wi) {
            return 0.0;
        }
        let (mut wo, mut wi) = (*wo, *wi);
        if wo.z < 0.0 {
            wo = -wo;
            wi = -wi;
        }

        let mut wm = wi + wo;
        if wm.length_squared() == 0.0 {
            return 0.0;
        }
        wm = wm.normalize();

        let theta_o = spherical_theta(&wo);
        let phi_o = wo.y.atan2(wo.x);
        let theta_m = spherical_theta(&wm);
        let phi_m = wm.y.atan2(wm.x);

        let mut u_wm = Point2f::new(
            theta2u(theta_m),
            phi2u(if self.brdf.isotropic {
                phi_m - phi_o
            } else {
                phi_m
            }),
        );
        u_wm.y -= u_wm.y.floor();

        let ui = interp.vndf.invert(u_wm, &[phi_o, theta_o]);
        let sample = ui.p;
        let vndf_pdf = ui.pdf;
        let lum_pdf = interp.luminance.evaluate(sample, &[phi_o, theta_o]);
        let sin_theta_m = (wm.x * wm.x + wm.y * wm.y).sqrt();
        let dot_wo_wm = Vector3f::dot(&wo, &wm).abs().max(1e-6);
        let jacobian = 4.0 * dot_wo_wm * (2.0 * PI * PI * u_wm.x * sin_theta_m).max(1e-6);
        vndf_pdf * lum_pdf / jacobian
    }

    pub fn flags(&self) -> BxDFFlags {
        BXDF_REFLECTION | BXDF_GLOSSY
    }

    pub fn regularize(&mut self) {}
}
