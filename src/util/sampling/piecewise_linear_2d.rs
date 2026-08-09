//! Conditional 2D piecewise-linear distribution used by the measured BRDF
//! data loader. Mirrors pbrt-v4's `PiecewiseLinear2D<Dimension>` class
//! template (`src/pbrt/util/sampling.h`) with the template parameter
//! expressed as `const N: usize` and the recursive `lookup<Dim>` helper
//! flattened into an iterative routine.
//!
//! The structure stores a 2D bilinear density grid (size_x * size_y)
//! optionally indexed by `N` external conditioning axes. CDFs are
//! pre-built so `sample` / `invert` are O(log(size)).

use crate::util::base::*;

/// Result of a `sample` / `invert` call: the (warped) point in the unit
/// square and the associated density.
#[derive(Debug, Clone, Copy)]
pub struct PLSample {
    pub p: Point2f,
    pub pdf: Float,
}

/// 2D piecewise-linear distribution conditioned on `N` outer axes.
///
/// `m_data` / `m_marginal_cdf` / `m_conditional_cdf` are flat buffers
/// indexed as `slice_index * (slice_size) + intra_slice_offset`. The
/// `slice_index` is determined from the conditioning parameters via the
/// `param_strides` table (a parameter that has `param_size == 1`
/// contributes a stride of zero so we just sit on the single slice).
#[derive(Debug, Clone)]
pub struct PiecewiseLinear2D<const N: usize> {
    /// Resolution of the discretized density (x, y).
    size: Vector2i,
    /// Patch size in the unit square (= 1 / (size - 1)).
    patch_size: Vector2f,
    /// Reciprocal of `patch_size` (= size - 1).
    inv_patch_size: Vector2f,
    /// Resolution of each conditioning axis.
    param_size: [usize; N],
    /// Stride per conditioning axis in slice units.
    param_strides: [usize; N],
    /// Sample locations along each conditioning axis.
    param_values: [Vec<f32>; N],
    /// Flat density values, one slice per conditioning combination.
    data: Vec<f32>,
    /// Marginal CDF, length `slices * size.y` per slice.
    marginal_cdf: Vec<f32>,
    /// Conditional CDF, length `slices * size.x * size.y` per slice.
    conditional_cdf: Vec<f32>,
}

impl<const N: usize> PiecewiseLinear2D<N> {
    /// Build the distribution from a flat density grid.
    ///
    /// `data` is laid out as `slices * size_y * size_x` floats. With no
    /// conditioning (`N == 0`) the input is a single 2D grid.
    ///
    /// When `build_cdf` is `true` (matching `normalize = true`), the
    /// constructor computes the marginal/conditional CDFs needed for
    /// `sample` / `invert`. Setting `build_cdf = false` skips that step
    /// to save memory when only `evaluate` is needed.
    pub fn new(
        data: &[f32],
        x_size: usize,
        y_size: usize,
        param_res: [usize; N],
        param_values_in: [&[f32]; N],
        normalize: bool,
        build_cdf: bool,
    ) -> Self {
        assert!(
            !(build_cdf && !normalize),
            "PiecewiseLinear2D: build_cdf implies normalize=true"
        );
        assert!(x_size >= 2 && y_size >= 2, "size must be >= 2 per axis");

        let size = Vector2i::new(x_size as i32, y_size as i32);
        let patch_size = Vector2f::new(1.0 / (x_size - 1) as Float, 1.0 / (y_size - 1) as Float);
        let inv_patch_size = Vector2f::new((x_size - 1) as Float, (y_size - 1) as Float);

        // Walk the conditioning axes in reverse so the lowest-index axis
        // strides the slowest -- exactly v4's layout.
        let mut param_size = [1usize; N];
        let mut param_strides = [0usize; N];
        let mut param_values: [Vec<f32>; N] = std::array::from_fn(|_| Vec::new());
        let mut slices = 1usize;
        for i in (0..N).rev() {
            assert!(
                param_res[i] >= 1,
                "PiecewiseLinear2D: parameter resolution must be >= 1"
            );
            param_size[i] = param_res[i];
            param_values[i] = param_values_in[i][..param_res[i]].to_vec();
            param_strides[i] = if param_res[i] > 1 { slices } else { 0 };
            slices *= param_size[i];
        }

        let n_values = x_size * y_size;
        let mut out_data = vec![0.0f32; slices * n_values];
        let mut marginal_cdf = if build_cdf {
            vec![0.0f32; slices * y_size]
        } else {
            Vec::new()
        };
        let mut conditional_cdf = if build_cdf {
            vec![0.0f32; slices * n_values]
        } else {
            Vec::new()
        };

        for slice in 0..slices {
            let slice_data = &data[slice * n_values..(slice + 1) * n_values];

            if build_cdf {
                let m_off = slice * y_size;
                let c_off = slice * n_values;

                // Conditional CDF along x: each row gets its prefix sum
                // using the trapezoid rule (matches v4).
                for y in 0..y_size {
                    let i = y * x_size;
                    conditional_cdf[c_off + i] = 0.0;
                    let mut sum: f64 = 0.0;
                    for x in 0..(x_size - 1) {
                        sum += 0.5 * (slice_data[i + x] as f64 + slice_data[i + x + 1] as f64);
                        conditional_cdf[c_off + i + x + 1] = sum as f32;
                    }
                }

                // Marginal CDF over y: trapezoid sum of each row's total
                // (which lives at the end of every conditional cdf row).
                marginal_cdf[m_off] = 0.0;
                let mut sum: f64 = 0.0;
                for y in 0..(y_size - 1) {
                    let row_a = conditional_cdf[c_off + (y + 1) * x_size - 1] as f64;
                    let row_b = conditional_cdf[c_off + (y + 2) * x_size - 1] as f64;
                    sum += 0.5 * (row_a + row_b);
                    marginal_cdf[m_off + y + 1] = sum as f32;
                }

                // Normalize. If the slice is identically zero, leave it
                // alone so downstream code can still evaluate to zero.
                let total = marginal_cdf[m_off + y_size - 1];
                if total > 0.0 {
                    let norm = 1.0 / total;
                    for i in 0..n_values {
                        conditional_cdf[c_off + i] *= norm;
                        out_data[slice * n_values + i] = slice_data[i] * norm;
                    }
                    for i in 0..y_size {
                        marginal_cdf[m_off + i] *= norm;
                    }
                } else {
                    for i in 0..n_values {
                        out_data[slice * n_values + i] = slice_data[i];
                    }
                }
            } else {
                // pbrt-v4 default: `1 / HProd(m_inv_patch_size)` (= the
                // unit patch area). evaluate() multiplies back by
                // `HProd(m_inv_patch_size)`, so the round-trip leaves
                // raw data values untouched. When `normalize=true` the
                // scale instead bakes a `1 / integral(data)` so the
                // round-tripped `evaluate` integrates to 1.
                let mut norm = 1.0 as Float / (inv_patch_size.x * inv_patch_size.y);
                if normalize {
                    let mut sum: f64 = 0.0;
                    for y in 0..(y_size - 1) {
                        let i = y * x_size;
                        for x in 0..(x_size - 1) {
                            let v00 = slice_data[i + x] as f64;
                            let v10 = slice_data[i + x + 1] as f64;
                            let v01 = slice_data[i + x + x_size] as f64;
                            let v11 = slice_data[i + x + x_size + 1] as f64;
                            sum += 0.25 * (v00 + v10 + v01 + v11);
                        }
                    }
                    if sum > 0.0 {
                        norm = (1.0 / sum) as Float;
                    }
                }
                for k in 0..n_values {
                    out_data[slice * n_values + k] = slice_data[k] * norm as f32;
                }
            }
        }

        Self {
            size,
            patch_size,
            inv_patch_size,
            param_size,
            param_strides,
            param_values,
            data: out_data,
            marginal_cdf,
            conditional_cdf,
        }
    }

    /// Compute per-axis parameter index + interpolation weight pair.
    /// Output buffer `weights` has 2*N entries: `[w0_0, w1_0, w0_1, w1_1, ...]`.
    fn param_lookup(&self, params: &[Float; N], weights: &mut [Float]) -> usize {
        let mut slice_offset = 0usize;
        for dim in 0..N {
            if self.param_size[dim] == 1 {
                weights[2 * dim] = 1.0;
                weights[2 * dim + 1] = 0.0;
                continue;
            }
            let vals = &self.param_values[dim];
            // Find largest index `pi` with vals[pi] <= params[dim].
            // The v4 FindInterval returns clamp(first - 1, 0, n - 2).
            let p = params[dim] as f32;
            let mut first = 0usize;
            let mut len = vals.len();
            while len > 0 {
                let half = len / 2;
                let mid = first + half;
                if vals[mid] <= p {
                    first = mid + 1;
                    len -= half + 1;
                } else {
                    len = half;
                }
            }
            let pi = first.saturating_sub(1).min(vals.len().saturating_sub(2));
            let p0 = vals[pi] as Float;
            let p1 = vals[pi + 1] as Float;
            let denom = (p1 - p0).max(Float::MIN_POSITIVE);
            let w1 = ((params[dim] - p0) / denom).clamp(0.0, 1.0);
            weights[2 * dim + 1] = w1;
            weights[2 * dim] = 1.0 - w1;
            slice_offset += self.param_strides[dim] * pi;
        }
        slice_offset
    }

    /// Recursive bilinear blend in v4 is `Sum_{corners} (w_0 * w_1 * ...) * data[corner_offset]`,
    /// which fans out to `2^N` data reads. We do that iteratively here.
    fn lookup(&self, data: &[f32], index: usize, slice_size: usize, weights: &[Float]) -> Float {
        let mut sum = 0.0 as Float;
        // Iterate over the 2^N combinations of (low/high) along each
        // conditioning axis; each combination contributes the product of
        // the chosen `param_weight` entries multiplied by `data[offset]`.
        let n_combos = 1usize << N;
        for combo in 0..n_combos {
            let mut offset = index;
            let mut w: Float = 1.0;
            for dim in 0..N {
                let pick_hi = (combo >> dim) & 1 == 1;
                if pick_hi {
                    offset += self.param_strides[dim] * slice_size;
                    w *= weights[2 * dim + 1];
                } else {
                    w *= weights[2 * dim];
                }
            }
            sum += w * data[offset] as Float;
        }
        sum
    }

    fn h_prod_i(v: Vector2i) -> usize {
        (v.x as usize) * (v.y as usize)
    }

    fn h_prod_f(v: Vector2f) -> Float {
        v.x * v.y
    }

    /// Evaluate the (unnormalized after `normalize=false`, normalized
    /// otherwise) density at `pos`. Matches pbrt-v4 `Evaluate`.
    pub fn evaluate(&self, mut pos: Point2f, params: &[Float; N]) -> Float {
        let mut weights = vec![0.0 as Float; 2 * N.max(1)];
        let slice_offset = self.param_lookup(params, &mut weights);

        pos.x *= self.inv_patch_size.x;
        pos.y *= self.inv_patch_size.y;
        let off_x = (pos.x as i32).min(self.size.x - 2).max(0);
        let off_y = (pos.y as i32).min(self.size.y - 2).max(0);
        let w1x = pos.x - off_x as Float;
        let w1y = pos.y - off_y as Float;
        let w0x = 1.0 - w1x;
        let w0y = 1.0 - w1y;

        let slice_size = Self::h_prod_i(self.size);
        let x_size = self.size.x as usize;
        let base = off_x as usize + off_y as usize * x_size + slice_offset * slice_size;

        let v00 = self.lookup(&self.data, base, slice_size, &weights);
        let v10 = self.lookup(&self.data, base + 1, slice_size, &weights);
        let v01 = self.lookup(&self.data, base + x_size, slice_size, &weights);
        let v11 = self.lookup(&self.data, base + x_size + 1, slice_size, &weights);

        (w0y * (w0x * v00 + w1x * v10) + w1y * (w0x * v01 + w1x * v11))
            * Self::h_prod_f(self.inv_patch_size)
    }

    /// Warp a uniform sample to one drawn from the distribution. Matches
    /// pbrt-v4 `Sample`.
    pub fn sample(&self, mut sample: Point2f, params: &[Float; N]) -> PLSample {
        let one_minus_eps = 1.0 - Float::EPSILON;
        sample.x = sample.x.clamp(1.0 - one_minus_eps, one_minus_eps);
        sample.y = sample.y.clamp(1.0 - one_minus_eps, one_minus_eps);

        let mut weights = vec![0.0 as Float; 2 * N.max(1)];
        let slice_offset = self.param_lookup(params, &mut weights);

        let x_size = self.size.x as usize;
        let y_size = self.size.y as usize;
        let slice_size = x_size * y_size;

        // Sample the row using the marginal CDF.
        let mut offset = if N != 0 { slice_offset * y_size } else { 0 };
        let fetch_marginal = |idx: usize, weights: &[Float], offset: usize| -> Float {
            self.lookup(&self.marginal_cdf, offset + idx, y_size, weights)
        };
        let mut first = 0usize;
        let mut len = y_size;
        while len > 0 {
            let half = len / 2;
            let mid = first + half;
            if fetch_marginal(mid, &weights, offset) < sample.y {
                first = mid + 1;
                len -= half + 1;
            } else {
                len = half;
            }
        }
        let row = first.saturating_sub(1).min(y_size.saturating_sub(2));

        sample.y -= fetch_marginal(row, &weights, offset);

        // Sample the column.
        offset = row * x_size;
        if N != 0 {
            offset += slice_offset * slice_size;
        }

        let r0 = self.lookup(
            &self.conditional_cdf,
            offset + x_size - 1,
            slice_size,
            &weights,
        );
        let r1 = self.lookup(
            &self.conditional_cdf,
            offset + x_size * 2 - 1,
            slice_size,
            &weights,
        );
        let is_const = (r0 - r1).abs() < 1e-4 * (r0 + r1).abs().max(1e-12);
        sample.y = if is_const {
            2.0 * sample.y
        } else {
            let disc = (r0 * r0 - 2.0 * sample.y * (r0 - r1)).max(0.0);
            r0 - disc.sqrt()
        };
        sample.y /= if is_const { r0 + r1 } else { r0 - r1 };

        sample.x *= (1.0 - sample.y) * r0 + sample.y * r1;

        let fetch_conditional = |idx: usize, weights: &[Float], offset: usize| -> Float {
            let v0 = self.lookup(&self.conditional_cdf, offset + idx, slice_size, weights);
            let v1 = self.lookup(
                &self.conditional_cdf,
                offset + idx + x_size,
                slice_size,
                weights,
            );
            (1.0 - sample.y) * v0 + sample.y * v1
        };
        first = 0;
        len = x_size;
        while len > 0 {
            let half = len / 2;
            let mid = first + half;
            if fetch_conditional(mid, &weights, offset) < sample.x {
                first = mid + 1;
                len -= half + 1;
            } else {
                len = half;
            }
        }
        let col = first.saturating_sub(1).min(x_size.saturating_sub(2));

        sample.x -= fetch_conditional(col, &weights, offset);

        offset += col;
        let v00 = self.lookup(&self.data, offset, slice_size, &weights);
        let v10 = self.lookup(&self.data, offset + 1, slice_size, &weights);
        let v01 = self.lookup(&self.data, offset + x_size, slice_size, &weights);
        let v11 = self.lookup(&self.data, offset + x_size + 1, slice_size, &weights);
        let c0 = (1.0 - sample.y) * v00 + sample.y * v01;
        let c1 = (1.0 - sample.y) * v10 + sample.y * v11;
        let is_const = (c0 - c1).abs() < 1e-4 * (c0 + c1).abs().max(1e-12);
        sample.x = if is_const {
            2.0 * sample.x
        } else {
            let disc = (c0 * c0 - 2.0 * sample.x * (c0 - c1)).max(0.0);
            c0 - disc.sqrt()
        };
        sample.x /= if is_const { c0 + c1 } else { c0 - c1 };

        PLSample {
            p: Point2f::new(
                (col as Float + sample.x) * self.patch_size.x,
                (row as Float + sample.y) * self.patch_size.y,
            ),
            pdf: ((1.0 - sample.x) * c0 + sample.x * c1) * Self::h_prod_f(self.inv_patch_size),
        }
    }

    /// Inverse of `sample` -- map a point in the unit square back to the
    /// uniform sample that would have produced it. Matches pbrt-v4
    /// `Invert`.
    pub fn invert(&self, mut sample: Point2f, params: &[Float; N]) -> PLSample {
        let mut weights = vec![0.0 as Float; 2 * N.max(1)];
        let slice_offset = self.param_lookup(params, &mut weights);

        let x_size = self.size.x as usize;
        let y_size = self.size.y as usize;
        let slice_size = x_size * y_size;

        sample.x *= self.inv_patch_size.x;
        sample.y *= self.inv_patch_size.y;
        let pos_x = (sample.x as i32).min(self.size.x - 2).max(0) as usize;
        let pos_y = (sample.y as i32).min(self.size.y - 2).max(0) as usize;
        sample.x -= pos_x as Float;
        sample.y -= pos_y as Float;

        let mut offset = pos_x + pos_y * x_size;
        if N != 0 {
            offset += slice_offset * slice_size;
        }

        let v00 = self.lookup(&self.data, offset, slice_size, &weights);
        let v10 = self.lookup(&self.data, offset + 1, slice_size, &weights);
        let v01 = self.lookup(&self.data, offset + x_size, slice_size, &weights);
        let v11 = self.lookup(&self.data, offset + x_size + 1, slice_size, &weights);

        let w1x = sample.x;
        let w0x = 1.0 - w1x;
        let w1y = sample.y;
        let w0y = 1.0 - w1y;

        let c0 = w0y * v00 + w1y * v01;
        let c1 = w0y * v10 + w1y * v11;
        let pdf = w0x * c0 + w1x * c1;

        sample.x *= c0 + 0.5 * sample.x * (c1 - c0);

        let v0_cond = self.lookup(&self.conditional_cdf, offset, slice_size, &weights);
        let v1_cond = self.lookup(&self.conditional_cdf, offset + x_size, slice_size, &weights);
        sample.x += (1.0 - sample.y) * v0_cond + sample.y * v1_cond;

        let row_offset = if N != 0 {
            pos_y * x_size + slice_offset * slice_size
        } else {
            pos_y * x_size
        };
        let r0 = self.lookup(
            &self.conditional_cdf,
            row_offset + x_size - 1,
            slice_size,
            &weights,
        );
        let r1 = self.lookup(
            &self.conditional_cdf,
            row_offset + x_size * 2 - 1,
            slice_size,
            &weights,
        );

        sample.x /= (1.0 - sample.y) * r0 + sample.y * r1;

        sample.y *= r0 + 0.5 * sample.y * (r1 - r0);

        let m_off = if N != 0 {
            slice_offset * y_size + pos_y
        } else {
            pos_y
        };
        sample.y += self.lookup(&self.marginal_cdf, m_off, y_size, &weights);

        PLSample {
            p: sample,
            pdf: pdf * Self::h_prod_f(self.inv_patch_size),
        }
    }

    pub fn size(&self) -> (usize, usize) {
        (self.size.x as usize, self.size.y as usize)
    }

    pub fn bytes_used(&self) -> usize {
        4 * (self.data.capacity() + self.marginal_cdf.capacity() + self.conditional_cdf.capacity())
            + self
                .param_values
                .iter()
                .map(|v| 4 * v.capacity())
                .sum::<usize>()
    }
}
