use crate::util::error::PbrtError;
use std::collections::HashSet;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Bounds3 {
    pub min: [f32; 3],
    pub max: [f32; 3],
}

impl Bounds3 {
    pub fn new(min: [f32; 3], max: [f32; 3]) -> Result<Self, PbrtError> {
        if !min.into_iter().chain(max).all(f32::is_finite) {
            return Err(PbrtError::error(
                "Light bounds contain a non-finite coordinate.",
            ));
        }
        if min.into_iter().zip(max).any(|(min, max)| min > max) {
            return Err(PbrtError::error("Light bounds have min greater than max."));
        }
        Ok(Self { min, max })
    }

    pub fn union(self, other: Self) -> Self {
        Self {
            min: [
                self.min[0].min(other.min[0]),
                self.min[1].min(other.min[1]),
                self.min[2].min(other.min[2]),
            ],
            max: [
                self.max[0].max(other.max[0]),
                self.max[1].max(other.max[1]),
                self.max[2].max(other.max[2]),
            ],
        }
    }

    pub fn diagonal(self) -> [f32; 3] {
        [
            self.max[0] - self.min[0],
            self.max[1] - self.min[1],
            self.max[2] - self.min[2],
        ]
    }

    pub fn centroid(self) -> [f32; 3] {
        [
            (self.min[0] + self.max[0]) * 0.5,
            (self.min[1] + self.max[1]) * 0.5,
            (self.min[2] + self.max[2]) * 0.5,
        ]
    }

    pub fn surface_area(self) -> f32 {
        let d = self.diagonal();
        2.0 * (d[0] * d[1] + d[1] * d[2] + d[2] * d[0])
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LightBounds {
    pub bounds: Bounds3,
    pub direction: [f32; 3],
    pub phi: f32,
    pub cos_theta_o: f32,
    pub cos_theta_e: f32,
    pub two_sided: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub enum LightBoundInput {
    Point {
        handle: u32,
        world_position: [f32; 3],
        intensity_max: f32,
        scale: f32,
    },
    AreaGroup {
        handle: u32,
        triangles: Vec<AreaTriangleInput>,
        emission_max: f32,
        scale: f32,
        two_sided: bool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AreaTriangleInput {
    pub world_positions: [[f32; 3]; 3],
    pub area: f32,
    pub geometric_normal: [f32; 3],
}

impl LightBounds {
    pub fn validate(self) -> Result<Self, PbrtError> {
        Bounds3::new(self.bounds.min, self.bounds.max)?;
        if !self
            .direction
            .into_iter()
            .chain([self.phi, self.cos_theta_o, self.cos_theta_e])
            .all(f32::is_finite)
        {
            return Err(PbrtError::error("Light bounds contain a non-finite value."));
        }
        let length_squared = self.direction.into_iter().map(|v| v * v).sum::<f32>();
        if length_squared == 0.0 {
            return Err(PbrtError::error("Light bounds direction must be non-zero."));
        }
        if self.phi < 0.0
            || !(-1.0..=1.0).contains(&self.cos_theta_o)
            || !(-1.0..=1.0).contains(&self.cos_theta_e)
        {
            return Err(PbrtError::error("Light bounds contain an invalid range."));
        }
        Ok(self)
    }

    pub fn union(self, other: Self) -> Result<Self, PbrtError> {
        self.validate()?;
        other.validate()?;
        if self.phi == 0.0 {
            return Ok(other);
        }
        if other.phi == 0.0 {
            return Ok(self);
        }

        let (direction, cos_theta_o) = union_direction_cones(
            self.direction,
            self.cos_theta_o,
            other.direction,
            other.cos_theta_o,
        );
        Ok(Self {
            bounds: self.bounds.union(other.bounds),
            direction,
            phi: self.phi + other.phi,
            cos_theta_o,
            cos_theta_e: self.cos_theta_e.min(other.cos_theta_e),
            two_sided: self.two_sided || other.two_sided,
        })
    }

    pub fn importance(self, p: [f32; 3], n: [f32; 3]) -> Result<f32, PbrtError> {
        self.validate()?;
        if !p.into_iter().chain(n).all(f32::is_finite) {
            return Err(PbrtError::error(
                "Importance input contains a non-finite value.",
            ));
        }
        let center = self.bounds.centroid();
        let diagonal = self.bounds.diagonal();
        let delta = [p[0] - center[0], p[1] - center[1], p[2] - center[2]];
        let mut d2 = dot(delta, delta);
        let diagonal_length = dot(diagonal, diagonal).sqrt();
        d2 = d2.max(diagonal_length * 0.5);
        if d2 <= 0.0 {
            return Ok(0.0);
        }

        // Match pbrt-v4 LightBounds::importance: wi points from the light
        // bound's center toward the reference point.
        let wi = normalize([p[0] - center[0], p[1] - center[1], p[2] - center[2]])?;
        let mut cos_theta_w = dot(self.direction, wi);
        if self.two_sided {
            cos_theta_w = cos_theta_w.abs();
        }
        let sin_theta_w = safe_sqrt(1.0 - cos_theta_w * cos_theta_w);

        let cos_theta_b = bound_subtended_directions(self.bounds, p)?;
        let sin_theta_b = safe_sqrt(1.0 - cos_theta_b * cos_theta_b);
        let sin_theta_o = safe_sqrt(1.0 - self.cos_theta_o * self.cos_theta_o);
        let cos_theta_x = cos_sub_clamped(sin_theta_w, cos_theta_w, sin_theta_o, self.cos_theta_o);
        let sin_theta_x = sin_sub_clamped(sin_theta_w, cos_theta_w, sin_theta_o, self.cos_theta_o);
        let cos_theta_p = cos_sub_clamped(sin_theta_x, cos_theta_x, sin_theta_b, cos_theta_b);
        if cos_theta_p <= self.cos_theta_e {
            return Ok(0.0);
        }

        let mut importance = self.phi * cos_theta_p / d2;
        if dot(n, n) != 0.0 {
            let cos_theta_i = dot(wi, normalize(n)?).abs();
            let sin_theta_i = safe_sqrt(1.0 - cos_theta_i * cos_theta_i);
            importance *= cos_sub_clamped(sin_theta_i, cos_theta_i, sin_theta_b, cos_theta_b);
        }
        Ok(importance.max(0.0))
    }
}

pub fn build_light_bounds(inputs: &[LightBoundInput]) -> Result<Vec<LightBounds>, PbrtError> {
    let mut handles = HashSet::with_capacity(inputs.len());
    for (index, input) in inputs.iter().enumerate() {
        let handle = match input {
            LightBoundInput::Point { handle, .. } | LightBoundInput::AreaGroup { handle, .. } => {
                *handle
            }
        };
        let expected_handle = u32::try_from(index)
            .map_err(|_| PbrtError::error("Light handle exceeds the u32 range."))?;
        if handle != expected_handle {
            return Err(PbrtError::error(
                "Light handles must match the global LightRecord order.",
            ));
        }
        if !handles.insert(handle) {
            return Err(PbrtError::error("Light handles must be unique."));
        }
    }
    inputs.iter().map(light_bounds_for_input).collect()
}

fn light_bounds_for_input(input: &LightBoundInput) -> Result<LightBounds, PbrtError> {
    match input {
        LightBoundInput::Point {
            handle: _,
            world_position,
            intensity_max,
            scale,
        } => {
            validate_emission(*intensity_max, *scale)?;
            let bounds = Bounds3::new(*world_position, *world_position)?;
            LightBounds {
                bounds,
                direction: [0.0, 0.0, 1.0],
                phi: 4.0 * std::f32::consts::PI * scale * intensity_max,
                cos_theta_o: -1.0,
                cos_theta_e: 0.0,
                two_sided: false,
            }
            .validate()
        }
        LightBoundInput::AreaGroup {
            handle: _,
            triangles,
            emission_max,
            scale,
            two_sided,
        } => {
            validate_emission(*emission_max, *scale)?;
            let mut result: Option<LightBounds> = None;
            for triangle in triangles {
                let bounds = triangle_bounds(triangle.world_positions)?;
                let area = triangle.area;
                if !area.is_finite() || area <= 0.0 {
                    return Err(PbrtError::error("Area light triangle has zero area."));
                }
                let direction = normalize(triangle.geometric_normal)?;
                if !direction.iter().all(|value| value.is_finite()) {
                    return Err(PbrtError::error(
                        "Area light geometric normal must be finite and non-zero.",
                    ));
                }
                let current = LightBounds {
                    bounds,
                    direction,
                    phi: emission_max * scale * area * std::f32::consts::PI,
                    cos_theta_o: 1.0,
                    cos_theta_e: 0.0,
                    two_sided: *two_sided,
                }
                .validate()?;
                result = Some(match result {
                    Some(existing) => existing.union(current)?,
                    None => current,
                });
            }
            result.ok_or_else(|| PbrtError::error("Area light group has no triangles."))
        }
    }
}

fn triangle_bounds(world_positions: [[f32; 3]; 3]) -> Result<Bounds3, PbrtError> {
    Bounds3::new(
        [
            world_positions[0][0]
                .min(world_positions[1][0])
                .min(world_positions[2][0]),
            world_positions[0][1]
                .min(world_positions[1][1])
                .min(world_positions[2][1]),
            world_positions[0][2]
                .min(world_positions[1][2])
                .min(world_positions[2][2]),
        ],
        [
            world_positions[0][0]
                .max(world_positions[1][0])
                .max(world_positions[2][0]),
            world_positions[0][1]
                .max(world_positions[1][1])
                .max(world_positions[2][1]),
            world_positions[0][2]
                .max(world_positions[1][2])
                .max(world_positions[2][2]),
        ],
    )
}

fn validate_emission(value: f32, scale: f32) -> Result<(), PbrtError> {
    if !value.is_finite() || !scale.is_finite() || value < 0.0 || scale < 0.0 {
        return Err(PbrtError::error("Light emission or scale is invalid."));
    }
    Ok(())
}

fn bound_subtended_directions(bounds: Bounds3, p: [f32; 3]) -> Result<f32, PbrtError> {
    let center = bounds.centroid();
    let radius = 0.5 * dot(bounds.diagonal(), bounds.diagonal()).sqrt();
    let to_center = sub(center, p);
    let d2 = dot(to_center, to_center);
    if d2 < radius * radius || d2 == 0.0 {
        return Ok(-1.0);
    }
    Ok(safe_sqrt(1.0 - radius * radius / d2))
}

fn union_direction_cones(a: [f32; 3], cos_a: f32, b: [f32; 3], cos_b: f32) -> ([f32; 3], f32) {
    let theta_a = cos_a.clamp(-1.0, 1.0).acos();
    let theta_b = cos_b.clamp(-1.0, 1.0).acos();
    let theta_d = angle_between(a, b);
    if (theta_d + theta_b).min(std::f32::consts::PI) <= theta_a {
        return (a, cos_a);
    }
    if (theta_d + theta_a).min(std::f32::consts::PI) <= theta_b {
        return (b, cos_b);
    }
    let theta_o = (theta_a + theta_d + theta_b) * 0.5;
    if theta_o >= std::f32::consts::PI {
        return ([0.0, 0.0, 1.0], -1.0);
    }
    let axis = cross(a, b);
    if dot(axis, axis) == 0.0 {
        return ([0.0, 0.0, 1.0], -1.0);
    }
    let axis = normalize(axis).unwrap_or([0.0, 0.0, 1.0]);
    let theta_r = theta_o - theta_a;
    (rotate(a, axis, theta_r), theta_o.cos())
}

fn angle_between(a: [f32; 3], b: [f32; 3]) -> f32 {
    let half_chord = |v: [f32; 3]| (dot(v, v).sqrt() * 0.5).min(1.0);
    if dot(a, b) < 0.0 {
        std::f32::consts::PI - 2.0 * half_chord(add(a, b)).asin()
    } else {
        2.0 * half_chord(sub(b, a)).asin()
    }
}

fn rotate(v: [f32; 3], axis: [f32; 3], angle: f32) -> [f32; 3] {
    let (s, c) = angle.sin_cos();
    let cross = cross(axis, v);
    add(
        add(scale_vector(v, c), scale_vector(cross, s)),
        scale_vector(axis, dot(axis, v) * (1.0 - c)),
    )
}

fn cos_sub_clamped(sin_a: f32, cos_a: f32, sin_b: f32, cos_b: f32) -> f32 {
    if cos_a > cos_b {
        1.0
    } else {
        cos_a * cos_b + sin_a * sin_b
    }
}

fn sin_sub_clamped(sin_a: f32, cos_a: f32, sin_b: f32, cos_b: f32) -> f32 {
    if cos_a > cos_b {
        0.0
    } else {
        sin_a * cos_b - cos_a * sin_b
    }
}

fn safe_sqrt(value: f32) -> f32 {
    value.max(0.0).sqrt()
}

fn normalize(v: [f32; 3]) -> Result<[f32; 3], PbrtError> {
    let length = dot(v, v).sqrt();
    if !length.is_finite() || length == 0.0 {
        return Err(PbrtError::error(
            "Light direction must be finite and non-zero.",
        ));
    }
    Ok(scale_vector(v, 1.0 / length))
}

fn dot(a: [f32; 3], b: [f32; 3]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn add(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn scale_vector(v: [f32; 3], scale: f32) -> [f32; 3] {
    [v[0] * scale, v[1] * scale, v[2] * scale]
}
