fn random01(pixel_index: u32, dimension: u32, depth: u32) -> f32 {
    return random01_stream(pixel_index, dimension, depth, 0u);
}

fn random01_stream(pixel_index: u32, dimension: u32, depth: u32, stream: u32) -> f32 {
    // pixel_index is tile-local; hash the absolute image pixel instead so
    // the sequence doesn't repeat across tiles.
    let pixel = sampler_pixel_from_index(pixel_index);
    let absolute_index = pixel.y * viewport.full_width + pixel.x;
    let value = viewport.seed
        ^ (absolute_index * 0x9e3779b9u)
        ^ (viewport.sample_index * 0x85ebca6bu)
        ^ ((dimension + depth * 8u) * 0xc2b2ae35u)
        ^ (stream * 0x27d4eb2du);
    return f32(hash_u32(value) & 0x00ffffffu) / 16777216.0;
}
