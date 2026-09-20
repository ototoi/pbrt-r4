use super::{FlatBuilder, Geometry, Vertex};
use crate::gpu::node::TriangleMeshShape;
use crate::util::error::PbrtError;

pub fn geometry_index(
    node_key: usize,
    component_index: usize,
    node_name: &str,
    shape: &TriangleMeshShape,
    builder: &mut FlatBuilder,
) -> Result<u32, PbrtError> {
    let key = (node_key, component_index);
    if let Some(&geometry) = builder.geometries_by_shape.get(&key) {
        return Ok(geometry);
    }

    let vertex_count = u32::try_from(shape.positions.len()).map_err(|_| {
        PbrtError::error(&format!(
            "Too many vertices in shape node \"{}\".",
            node_name
        ))
    })?;
    let index_count = u32::try_from(shape.indices.len()).map_err(|_| {
        PbrtError::error(&format!(
            "Too many indices in shape node \"{}\".",
            node_name
        ))
    })?;
    if shape.indices.len() % 3 != 0 {
        return Err(PbrtError::error(&format!(
            "Shape node \"{}\" has an index count that is not divisible by three.",
            node_name
        )));
    }

    validate_attribute_len(
        node_name,
        shape.normals.as_deref(),
        shape.positions.len(),
        "normal",
    )?;
    validate_attribute_len(
        node_name,
        shape.tangents.as_deref(),
        shape.positions.len(),
        "tangent",
    )?;
    validate_attribute_len(node_name, shape.uvs.as_deref(), shape.positions.len(), "UV")?;

    let first_vertex = u32::try_from(builder.vertices.len()).map_err(|_| {
        PbrtError::error("The flattened GPU vertex buffer exceeds the u32 index range.")
    })?;
    let first_index = u32::try_from(builder.indices.len()).map_err(|_| {
        PbrtError::error("The flattened GPU index buffer exceeds the u32 index range.")
    })?;
    for (index, position) in shape.positions.iter().enumerate() {
        builder.vertices.push(Vertex {
            position: position.0,
            normal: shape
                .normals
                .as_ref()
                .map(|normals| normals[index].0)
                .unwrap_or([0.0; 3]),
            tangent: shape
                .tangents
                .as_ref()
                .map(|tangents| tangents[index].0)
                .unwrap_or([0.0; 3]),
            uv: shape
                .uvs
                .as_ref()
                .map(|uvs| uvs[index].0)
                .unwrap_or([0.0; 2]),
        });
    }
    for &index in &shape.indices {
        if index >= vertex_count {
            return Err(PbrtError::error(&format!(
                "Shape node \"{}\" contains an out-of-range vertex index.",
                node_name
            )));
        }
        let flattened_index = first_vertex.checked_add(index).ok_or_else(|| {
            PbrtError::error("The flattened GPU vertex buffer exceeds the u32 index range.")
        })?;
        builder.indices.push(flattened_index);
    }

    let geometry = u32::try_from(builder.geometries.len()).map_err(|_| {
        PbrtError::error("The flattened GPU geometry table exceeds the u32 index range.")
    })?;
    builder.geometries.push(Geometry {
        first_vertex,
        vertex_count,
        first_index,
        index_count,
    });
    builder.geometries_by_shape.insert(key, geometry);
    Ok(geometry)
}

pub fn validate_attribute_len<T>(
    node_name: &str,
    attribute: Option<&[T]>,
    vertex_count: usize,
    attribute_name: &str,
) -> Result<(), PbrtError> {
    if let Some(attribute) = attribute {
        if attribute.len() != vertex_count {
            return Err(PbrtError::error(&format!(
                "Shape node \"{}\" has {} {} values for {} vertices.",
                node_name,
                attribute.len(),
                attribute_name,
                vertex_count
            )));
        }
    }
    Ok(())
}
