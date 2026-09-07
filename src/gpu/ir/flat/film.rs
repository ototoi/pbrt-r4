use super::SpectrumId;

#[derive(Clone, Debug, PartialEq)]
pub struct Film {
    pub sensor_response: [SpectrumId; 3],
    pub output_rgb_from_sensor_rgb: [[f32; 3]; 3],
    pub imaging_ratio: f32,
    pub scale: f32,
    pub max_sample_luminance: f32,
}
