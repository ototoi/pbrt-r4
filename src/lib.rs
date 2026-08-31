// pbrt-r4 library structure
// Organized to match pbrt-v4's architecture

// Prelude for external consumers. **Library-internal modules must not
// import `crate::prelude::*`; use explicit paths.**
pub mod prelude;

// New pbrt-v4 style structure
pub mod base; // Base trait definitions
pub mod bsdf; // BSDF wrapper around BxDF
pub mod bssrdf; // BSSRDF for subsurface scattering
pub mod cpu;
pub mod util; // Utility functions and data structures // CPU rendering implementation

// Top-level modules (pbrt-v4 style)
pub mod film; // Film and pixel handling
pub mod interaction; // Ray-surface interactions
pub mod options; // Rendering options
pub mod paramdict; // Parameter dictionary (paramdict)
pub mod parser; // Scene file parser
pub mod scene; // Scene representation

// Implementation modules (cameras, materials, lights, etc.)
pub mod bxdfs;
pub mod cameras;
pub mod displays;
pub mod ext;
pub mod filters;
pub mod lights;
pub mod materials;
pub mod media;
pub mod samplers;
pub mod shapes;
pub mod textures;

// GPU rendering implementation
pub mod gpu;
