fn hash_u32(value: u32) -> u32 {
    var h = value;
    h = (h ^ (h >> 16u)) * 0x7feb352du;
    h = (h ^ (h >> 15u)) * 0x846ca68bu;
    return h ^ (h >> 16u);
}

fn random01(pixel_index: u32, dimension: u32, depth: u32) -> f32 {
    // pixel_index is tile-local; hash the absolute image pixel instead so
    // the sequence doesn't repeat across tiles.
    let pixel = sampler_pixel_from_index(pixel_index);
    let absolute_index = pixel.y * viewport.full_width + pixel.x;
    let value = viewport.seed
        ^ (absolute_index * 0x9e3779b9u)
        ^ (viewport.sample_index * 0x85ebca6bu)
        ^ ((dimension + depth * 8u) * 0xc2b2ae35u);
    return f32(hash_u32(value) & 0x00ffffffu) / 16777216.0;
}
