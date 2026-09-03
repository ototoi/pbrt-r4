use crate::gpu::ir::flat;
use crate::util::error::PbrtError;

use super::abi::{row_major_to_tlas_transform, validate_affine, Geometry, Instance, Vertex};

pub struct Acceleration {
    pub blases: Vec<wgpu::Blas>,
    pub tlas: wgpu::Tlas,
}

pub fn build(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    vertex_buffer: &wgpu::Buffer,
    index_buffer: &wgpu::Buffer,
    geometries: &[Geometry],
    instances: &[Instance],
    flat_instances: &[flat::Instance],
) -> Result<Acceleration, PbrtError> {
    if geometries.is_empty() || instances.is_empty() {
        return Err(PbrtError::error(
            "WebGPU primary-ray rendering requires geometry and instance data.",
        ));
    }
    if instances.len() > 0x00ff_ffff {
        return Err(PbrtError::error(
            "WebGPU instance count exceeds the TLAS custom-data range.",
        ));
    }
    let sizes: Vec<_> = geometries
        .iter()
        .map(|geometry| wgpu::BlasTriangleGeometrySizeDescriptor {
            vertex_format: wgpu::VertexFormat::Float32x3,
            vertex_count: geometry.vertex_count,
            index_format: Some(wgpu::IndexFormat::Uint32),
            index_count: Some(geometry.index_count),
            flags: wgpu::AccelerationStructureGeometryFlags::OPAQUE,
        })
        .collect();
    let blases: Vec<_> = sizes
        .iter()
        .enumerate()
        .map(|(index, size)| {
            if size.vertex_count == 0 || size.index_count.unwrap_or(0) == 0 {
                return Err(PbrtError::error(&format!(
                    "WebGPU geometry {index} is empty."
                )));
            }
            Ok(device.create_blas(
                &wgpu::CreateBlasDescriptor {
                    label: Some("pbrt-r4 geometry BLAS"),
                    flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
                    update_mode: wgpu::AccelerationStructureUpdateMode::Build,
                },
                wgpu::BlasGeometrySizeDescriptors::Triangles {
                    descriptors: vec![size.clone()],
                },
            ))
        })
        .collect::<Result<_, _>>()?;

    if flat_instances.len() != instances.len() {
        return Err(PbrtError::error(
            "WebGPU backend instance data does not match Flat IR instance data.",
        ));
    }
    let mut tlas = device.create_tlas(&wgpu::CreateTlasDescriptor {
        label: Some("pbrt-r4 scene TLAS"),
        max_instances: instances.len() as u32,
        flags: wgpu::AccelerationStructureFlags::PREFER_FAST_TRACE,
        update_mode: wgpu::AccelerationStructureUpdateMode::Build,
    });
    for (index, (instance, flat_instance)) in instances.iter().zip(flat_instances).enumerate() {
        validate_affine(flat_instance.transform, &format!("Flat instance {index}"))?;
        let transform = row_major_to_tlas_transform(flat_instance.transform);
        tlas[index] = Some(wgpu::TlasInstance::new(
            &blases[instance.geometry as usize],
            transform,
            index as u32,
            0xff,
        ));
    }

    let entries: Vec<_> = geometries
        .iter()
        .zip(sizes.iter())
        .zip(blases.iter())
        .map(|((geometry, size), blas)| wgpu::BlasBuildEntry {
            blas,
            geometry: wgpu::BlasGeometries::TriangleGeometries(vec![wgpu::BlasTriangleGeometry {
                size,
                vertex_buffer,
                first_vertex: geometry.vertex_offset,
                vertex_stride: std::mem::size_of::<Vertex>() as u64,
                index_buffer: Some(index_buffer),
                first_index: Some(geometry.index_offset),
                transform_buffer: None,
                transform_buffer_offset: None,
            }]),
        })
        .collect();
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("pbrt-r4 acceleration structure encoder"),
    });
    encoder.build_acceleration_structures(entries.iter(), std::iter::once(&tlas));
    queue.submit(Some(encoder.finish()));
    device
        .poll(wgpu::PollType::wait_indefinitely())
        .map_err(|error| {
            PbrtError::error(&format!(
                "WebGPU acceleration structure build failed: {error}"
            ))
        })?;
    Ok(Acceleration { blases, tlas })
}
