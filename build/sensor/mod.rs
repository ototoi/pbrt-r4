mod build_pixel_sensor;
mod cie_s_data;
mod illuminant;
mod math;
mod sensor_data;
mod swatch_data;

pub fn build() {
    build_pixel_sensor::build();
}
