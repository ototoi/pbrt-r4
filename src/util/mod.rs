// Utility functions and data structures for pbrt-r4
// Corresponds to pbrt-v4's src/pbrt/util/

// Base types and constants
pub mod base;

// Math and geometry
pub mod efloat;
pub mod geometry;
pub mod interpolation;
pub mod quaternion;
pub mod transform;
pub mod vecmath;

// Sampling and random numbers
pub mod distribution;
pub mod lowdiscrepancy;
pub mod rng;
pub mod sampling;
pub mod scattering; // BSDF/scattering utility functions

// Color and spectrum
pub mod spectrum;

// Image processing
pub mod image;
pub mod imageio;
pub mod math;
pub mod mesh;

// Memory and performance
pub mod memory;
pub mod profile;
pub mod stats;

// Error handling and utilities
pub mod error;
pub mod misc;
pub mod tensor;

// Re-exports for convenience
pub use base::*;
pub use distribution::*;
pub use efloat::*;
pub use error::*;
pub use geometry::*;
pub use imageio::*;
pub use interpolation::*;
pub use lowdiscrepancy::*;
pub use math::*;
pub use memory::*;
pub use mesh::*;
pub use misc::*;
pub use profile::*;
pub use quaternion::*;
pub use sampling::*;
pub use spectrum::*;
pub use stats::*;
pub use transform::*;
pub use vecmath::*;
