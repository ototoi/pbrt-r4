mod build_cie;
mod build_config;
mod build_rgb_refl;
mod build_rgb_to_spectrum;
mod build_utils;
mod build_xyz;
pub mod cie_data;
pub mod config;
mod rgb_data;
mod spectrum_config;
mod to_string;
mod utils;

pub fn build() {
    build_config::build();
    build_utils::build();
    build_cie::build();
    build_xyz::build();
    build_rgb_refl::build();
    build_rgb_to_spectrum::build();
}
