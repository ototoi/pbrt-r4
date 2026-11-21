struct SparseGridLeafLookup {
    leaf_base: u32,
    value: f32,
}

fn sparse_grid_tile(value: f32) -> SparseGridLeafLookup {
    return SparseGridLeafLookup(0xffffffffu, value);
}

// Finds the leaf holding the coordinate, or the constant tile value covering it.
fn sparse_grid_find_leaf(grid: SparseScalarGridRecord, coordinate: vec3<i32>) -> SparseGridLeafLookup {
    if (grid.ranges[1].z != 1u) {
        set_render_error();
        return sparse_grid_tile(0.0);
    }
    let roots_begin = grid.ranges[0].x;
    let roots_count = grid.ranges[0].y;
    if (roots_begin > arrayLength(&sparse_grid_roots)
        || roots_count > arrayLength(&sparse_grid_roots) - roots_begin) {
        set_render_error();
        return sparse_grid_tile(grid.background.x);
    }
    let entries_begin = grid.ranges[0].z;
    let entries_count = grid.ranges[0].w;
    if (entries_begin > arrayLength(&sparse_grid_entries)
        || entries_count > arrayLength(&sparse_grid_entries) - entries_begin) {
        set_render_error();
        return sparse_grid_tile(grid.background.x);
    }
    let entries_end = entries_begin + entries_count;
    let leaf_begin = grid.ranges[1].x;
    let leaf_count = grid.ranges[1].y;
    if (leaf_begin > arrayLength(&volume_data)
        || leaf_count > arrayLength(&volume_data) - leaf_begin) {
        set_render_error();
        return sparse_grid_tile(grid.background.x);
    }
    let leaf_end = leaf_begin + leaf_count;
    let block = coordinate >> vec3<u32>(12);
    for (var i = 0u; i < roots_count; i += 1u) {
        let root = sparse_grid_roots[roots_begin + i];
        if (all(root.block == block)) {
            if (root.child_offset == 0xffffffffu) { return sparse_grid_tile(root.value); }
            let upper_base = root.child_offset;
            if (upper_base < entries_begin || upper_base > arrayLength(&sparse_grid_entries)
                || 32768u > arrayLength(&sparse_grid_entries) - upper_base
                || upper_base + 32768u > entries_end) {
                set_render_error();
                return sparse_grid_tile(grid.background.x);
            }
            let upper_local = coordinate >> vec3<u32>(7) & vec3<i32>(31);
            let upper_index = u32(upper_local.x * 1024 + upper_local.y * 32 + upper_local.z);
            let upper = sparse_grid_entries[upper_base + upper_index];
            if (upper.child_offset == 0xffffffffu) { return sparse_grid_tile(upper.value); }
            let lower_base = upper.child_offset;
            if (lower_base < entries_begin || lower_base > arrayLength(&sparse_grid_entries)
                || 4096u > arrayLength(&sparse_grid_entries) - lower_base
                || lower_base + 4096u > entries_end) {
                set_render_error();
                return sparse_grid_tile(grid.background.x);
            }
            let lower_local = coordinate >> vec3<u32>(3) & vec3<i32>(15);
            let lower_index = u32(lower_local.x * 256 + lower_local.y * 16 + lower_local.z);
            let lower = sparse_grid_entries[lower_base + lower_index];
            if (lower.child_offset == 0xffffffffu) { return sparse_grid_tile(lower.value); }
            let leaf_base = lower.child_offset;
            if (leaf_base < leaf_begin || leaf_base > arrayLength(&volume_data)
                || 512u > arrayLength(&volume_data) - leaf_base
                || leaf_base + 512u > leaf_end) {
                set_render_error();
                return sparse_grid_tile(grid.background.x);
            }
            return SparseGridLeafLookup(leaf_base, 0.0);
        }
    }
    return sparse_grid_tile(grid.background.x);
}

fn sparse_grid_leaf_value(lookup: SparseGridLeafLookup, coordinate: vec3<i32>) -> f32 {
    if (lookup.leaf_base == 0xffffffffu) { return lookup.value; }
    let leaf_local = coordinate & vec3<i32>(7);
    return volume_data[lookup.leaf_base + u32(leaf_local.x * 64 + leaf_local.y * 8 + leaf_local.z)];
}

fn sparse_grid_get_value(grid: SparseScalarGridRecord, coordinate: vec3<i32>) -> f32 {
    return sparse_grid_leaf_value(sparse_grid_find_leaf(grid, coordinate), coordinate);
}
