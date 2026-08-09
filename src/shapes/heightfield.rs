use super::triangle::create_triangle_mesh;
use crate::base::shape::Shape;
use crate::paramdict::*;

use crate::shapes::*;
use crate::util::base::*;
use crate::util::error::*;
// Includes cos_theta, abs_cos_theta, same_hemisphere, etc.

use std::sync::Arc;

pub struct HeightField;

impl HeightField {
    pub fn create(
        o2w: &Transform,
        w2o: &Transform,
        reverse_orientation: bool,
        params: &ParameterDictionary,
    ) -> Result<Vec<Arc<Shape>>, PbrtError> {
        create_heightfield(o2w, w2o, reverse_orientation, params)
    }
}

pub fn create_heightfield(
    o2w: &Transform,
    w2o: &Transform,
    reverse_orientation: bool,
    params: &ParameterDictionary,
) -> Result<Vec<Arc<Shape>>, PbrtError> {
    let nx = params.get_one_int("nu", -1);
    let ny = params.get_one_int("nv", -1);
    if nx == -1 || ny == -1 {
        return Err(PbrtError::error(
            "Must provide \"nu\" and \"nv\" parameters to heightfield shape.",
        ));
    }
    let nx = nx as usize;
    let ny = ny as usize;
    if let Some(z) = params.get_floats_ref("Pz") {
        let nitems = z.len();
        if nitems != nx * ny {
            return Err(PbrtError::error(
                "Number of \"Pz\" values doesn't match resolution.",
            ));
        }

        let ntris = 2 * (nx - 1) * (ny - 1);
        let mut indices = vec![0; 3 * ntris];
        let mut p = vec![Point3f::default(); nx * ny];
        let mut uvs = vec![Point2f::default(); nx * ny];

        // Compute heightfield vertex positions
        for y in 0..ny {
            for x in 0..nx {
                // Compute height for heightfield vertex
                let pos = nx * y + x;

                let xx = x as Float / (nx - 1) as Float;
                let yy = y as Float / (ny - 1) as Float;
                let zz = z[pos];

                p[pos] = Point3f::new(xx, yy, zz);
                uvs[pos] = Point2f::new(xx, yy);
            }
        }

        // Fill in heightfield vertex offset array
        let vert = |x, y| (x + y * nx) as u32;
        for y in 0..ny - 1 {
            for x in 0..nx - 1 {
                let i = (x + y * (nx - 1)) * 6;
                indices[i] = vert(x, y);
                indices[i + 1] = vert(x + 1, y);
                indices[i + 2] = vert(x + 1, y + 1);
                indices[i + 3] = vert(x, y);
                indices[i + 4] = vert(x + 1, y + 1);
                indices[i + 5] = vert(x, y + 1);
            }
        }

        let nparams = ParameterDictionary::new();
        let mesh = create_triangle_mesh(
            o2w,
            w2o,
            reverse_orientation,
            indices,
            p,
            Vec::new(),
            Vec::new(),
            uvs,
            &nparams,
        )?;
        let mesh: Vec<Arc<Shape>> = mesh
            .into_iter()
            .map(|tri| Arc::new(Shape::Triangle(tri)))
            .collect();
        return Ok(mesh);
    } else {
        return Err(PbrtError::error(
            "No vertex positions provided for heightfield shape.",
        ));
    }
}
