use super::lightdistrib::*;
use crate::base::{Light, LightSampleContext};
use crate::cpu::integrators::IntegratorBase;
use crate::interaction::*;
use crate::media::*;
use crate::util::base::*;
use crate::util::geometry::*;
use crate::util::lowdiscrepancy::*;
use crate::util::sampling::*;
use crate::util::spectrum::SampledWavelengths;
//use std::collections::HashMap;
use std::sync::Arc;
use std::sync::Mutex;

#[derive(Debug, Clone)]
struct HashEntry {
    packed_pos: u64,
    distribution: Arc<Distribution1D>,
}

// A spatially-varying light distribution that adjusts the probability of
// sampling a light source based on an estimate of its contribution to a
// region of space.  A fixed voxel grid is imposed over the scene bounds
// and a sampling distribution is computed as needed for each voxel.
pub struct SpatialLightDistribution {
    world_bound: Bounds3f,
    lights: Vec<Arc<Light>>,
    voxels: [u32; 3],
    hash_table: Vec<Mutex<Option<HashEntry>>>,
}

impl SpatialLightDistribution {
    pub fn new(base: &IntegratorBase, max_voxels: u32) -> Self {
        let mut voxels = [1, 1, 1];

        let b = base.aggregate.bounds();
        let diag = b.diagonal();
        let bmax = diag[b.maximum_extent()];
        for i in 0..3 {
            voxels[i] =
                (Float::ceil(diag[i] / bmax * max_voxels as Float) as u32).clamp(1, max_voxels);

            assert!(voxels[i] < (1 << 20));
        }

        let hash_table_size = (4 * voxels[0] * voxels[1] * voxels[2]) as usize;
        let mut hash_table = Vec::new();
        for _ in 0..hash_table_size {
            hash_table.push(Mutex::new(None));
        }
        SpatialLightDistribution {
            world_bound: b,
            lights: base.lights.clone(),
            voxels,
            hash_table,
        }
    }

    pub fn make_hash(&self, packed_pos: u64) -> u64 {
        let hash_table_size = self.hash_table.len() as u64;

        // Compute a hash value from the packed voxel coordinates.  We could
        // just take packedPos mod the hash table size, but since packedPos
        // isn't necessarily well distributed on its own, it's worthwhile to do
        // a little work to make sure that its bits values are individually
        // fairly random. For details of and motivation for the following, see:
        // http://zimbry.blogspot.ch/2011/09/better-bit-mixing-improving-on.html
        let mut hash: u64 = packed_pos;
        hash ^= hash.wrapping_shr(31); // hash >> 31
        hash = hash.wrapping_mul(0x7fb5d329728ea185); //hash *= 0x7fb5d329728ea185;
        hash ^= hash.wrapping_shr(27); //hash >> 27)
        hash = hash.wrapping_mul(0x81dadef4bc2dd44d); //hash *= 0x81dadef4bc2dd44d;
        hash ^= hash.wrapping_shr(33); //(hash >> 33);
        hash %= hash_table_size;
        return hash;
    }

    pub fn get_hash_key(&self, p: &Point3f) -> (u64, u64, [u32; 3]) {
        // First, compute integer voxel coordinates for the given point |p|
        // with respect to the overall voxel grid.
        let bounds = self.world_bound.clone();
        let offset = bounds.offset(p);
        let mut offset = [offset[0], offset[1], offset[2]];
        for i in 0..3 {
            offset[i] = offset[i].clamp(0.0, 1.0);
        }
        let mut pi: [u32; 3] = [0; 3];
        for i in 0..3 {
            // The clamp should almost never be necessary, but is there to be
            // robust to computed intersection points being slightly outside
            // the scene bounds due to floating-point roundoff error.
            let voxels_i = self.voxels[i];
            pi[i] = u32::clamp(
                (offset[i] * voxels_i as Float) as u32,
                0,
                (voxels_i - 1) as u32,
            );
        }
        // Pack the 3D integer voxel coordinates into a single 64-bit value.
        let packed_pos = ((pi[0] as u64).wrapping_shl(40)) as u64
            | ((pi[1] as u64).wrapping_shl(20)) as u64
            | (pi[2]) as u64;
        return (packed_pos, self.make_hash(packed_pos), pi);
    }

    fn compute_distribution(&self, pi: &[u32; 3]) -> Arc<Distribution1D> {
        // Compute the world-space bounding box of the voxel corresponding to
        // |pi|.
        let world_bound = self.world_bound.clone();
        let voxels = self.voxels.as_ref();
        let p0 = Point3f::new(
            pi[0] as Float / voxels[0] as Float,
            pi[1] as Float / voxels[1] as Float,
            pi[2] as Float / voxels[2] as Float,
        );
        let p1 = Point3f::new(
            (pi[0] + 1) as Float / voxels[0] as Float,
            (pi[1] + 1) as Float / voxels[1] as Float,
            (pi[2] + 1) as Float / voxels[2] as Float,
        );
        let pp0 = world_bound.lerp(&p0);
        let pp1 = world_bound.lerp(&p1);
        let voxel_bounds = Bounds3f::new(&pp0, &pp1);

        // Compute the sampling distribution. Sample a number of points inside
        // voxelBounds using a 3D Halton sequence; at each one, sample each
        // light source and compute a weight based on Li/pdf for the light's
        // sample (ignoring visibility between the point in the voxel and the
        // point on the light source) as an approximation to how much the light
        // is likely to contribute to illumination in the voxel.
        let lights = &self.lights;
        let n_samples: usize = 128;
        let lsz = lights.len();
        let mut light_contrib = vec![0.0; lsz];
        for i in 0..n_samples {
            let t = Point3f::new(
                radical_inverse(0, i as u64),
                radical_inverse(1, i as u64),
                radical_inverse(2, i as u64),
            );
            let po = voxel_bounds.lerp(&t);

            //p, n, perr, wo, time, mi
            let intr = Interaction::from(BaseInteraction {
                p: po, //0
                time: 0.0,
                p_error: Vector3f::zero(), //2
                n: Normal3f::zero(),       //1
                uv: Point2f::zero(),
                wo: Vector3f::new(1.0, 0.0, 0.0), //3
                medium_interface: MediumInterface::new(),
            });

            // Use the next two Halton dimensions to sample a point on the
            // light source.
            let u = Point2f::new(radical_inverse(3, i as u64), radical_inverse(4, i as u64));
            for j in 0..lsz {
                let light = lights[j].as_ref();
                let lambda = SampledWavelengths::sample_visible(0.5);
                let ctx = LightSampleContext::from(&intr);
                if let Some(s) = light.sample_li(&ctx, u, &lambda, false) {
                    let (li, pdf) = (s.l, s.pdf);
                    if pdf > 0.0 {
                        light_contrib[j] += li.y(&lambda) / pdf;
                    }
                }
            }
        }

        // We don't want to leave any lights with a zero probability; it's
        // possible that a light contributes to points in the voxel even though
        // we didn't find such a point when sampling above.  Therefore, compute
        // a minimum (small) weight and ensure that all lights are given at
        // least the corresponding probability.
        let sum_contrib: Float = light_contrib.iter().sum();
        let avg_contrib = sum_contrib / ((n_samples * lsz) as Float);
        let min_contrib = if avg_contrib > 0.0 {
            0.001 * avg_contrib
        } else {
            1.0
        };
        for i in 0..lsz {
            light_contrib[i] = Float::max(min_contrib, light_contrib[i]);
        }
        return Arc::new(Distribution1D::new(&light_contrib));
    }
}

impl LightDistribution for SpatialLightDistribution {
    fn lookup(&self, p: &Point3f) -> Arc<Distribution1D> {
        let (packed_pos, mut hash, pi) = self.get_hash_key(p);
        assert!((hash as usize) < self.hash_table.len());

        // Now, see if the hash table already has an entry for the voxel. We'll
        // use quadratic probing when the hash table entry is already used for
        // another value; step stores the square root of the probe step.
        let mut step = 1;
        loop {
            let mut hash_entry = self.hash_table[hash as usize].lock().unwrap();
            // Does the hash table entry at offset |hash| match the current point?
            if let Some(entry) = hash_entry.as_ref() {
                let entry_packed_pos = entry.packed_pos;
                if entry_packed_pos == packed_pos {
                    // Yes! Most of the time, there should already by a light
                    // sampling distribution available.
                    return entry.distribution.clone();
                } else {
                    // The hash table entry we're checking has already been
                    // allocated for another voxel. Advance to the next entry with
                    // quadratic probing.
                    hash += step * step;
                    let hash_table_size = self.hash_table.len() as u64;
                    if hash >= hash_table_size {
                        hash %= hash_table_size;
                    }
                    step += 1;
                }
            } else {
                // We have found an invalid entry. (Though this may have
                // changed since the load into entryPackedPos above.)  Use an
                // atomic compare/exchange to try to claim this entry for the
                // current position.
                let distrib = self.compute_distribution(&pi);
                *hash_entry = Some(HashEntry {
                    packed_pos,
                    distribution: distrib.clone(),
                });

                return distrib;
            }
        }
    }
}

unsafe impl Sync for SpatialLightDistribution {}
unsafe impl Send for SpatialLightDistribution {}
