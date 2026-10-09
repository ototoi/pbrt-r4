use super::super::abi::DispatchIndirectArgs;

pub(super) fn dispatch(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bind_groups: &[wgpu::BindGroup; 2],
    workgroups_x: u32,
    workgroups_y: u32,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind_groups[0], &[]);
    pass.set_bind_group(1, &bind_groups[1], &[]);
    pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
}

pub(super) fn dispatch_count(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bind_groups: &[wgpu::BindGroup; 2],
    count: u32,
) {
    let groups = count.div_ceil(64);
    let x = groups.min(65_535);
    let y = groups.div_ceil(65_535);
    dispatch(encoder, pipeline, bind_groups, x, y);
}

pub(super) fn dispatch_indirect(
    encoder: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bind_groups: &[wgpu::BindGroup; 2],
    indirect_buffer: &wgpu::Buffer,
    slot: u64,
) {
    let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
        label: None,
        timestamp_writes: None,
    });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, &bind_groups[0], &[]);
    pass.set_bind_group(1, &bind_groups[1], &[]);
    let offset = slot * std::mem::size_of::<DispatchIndirectArgs>() as u64;
    pass.dispatch_workgroups_indirect(indirect_buffer, offset);
}
