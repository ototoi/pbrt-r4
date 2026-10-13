@group(0) @binding(0) var tlas: acceleration_structure;
@group(0) @binding(1) var<storage, read> vertices: array<Vertex>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> geometries: array<Geometry>;
@group(0) @binding(4) var<storage, read> instances: array<Instance>;
@group(0) @binding(5) var<storage, read> bssrdf_tables: array<BSSRDFTableRecord>;
@group(0) @binding(6) var<storage, read> bssrdf_values: array<f32>;
@group(0) @binding(7) var<storage, read_write> bssrdf_work: BSSRDFProbeQueue;
@group(0) @binding(8) var<storage, read_write> bssrdf_results: array<BSSRDFProbeResult>;
@group(0) @binding(9) var<storage, read_write> bssrdf_error: atomic<u32>;
fn set_render_error() { atomicStore(&bssrdf_error, 1u); }
