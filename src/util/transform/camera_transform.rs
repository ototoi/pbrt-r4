// pbrt-v4 verbatim translation of `class CameraTransform`
// (cameras.h:27-98, cameras.cpp:27-57). Owns both the
// `renderFromCamera` animated transform and the (static)
// `worldFromRender` transform; together they cover all the queries
// the camera and the scene plumbing need.

use super::animated_transform::AnimatedTransform;
use super::transform::Transform;
use crate::options::pbrt_options::RenderingCoordinateSystem;
use crate::util::base::*;
use crate::util::geometry::*;

/// pbrt-v4 `class CameraTransform` (cameras.h:27).
///
/// The ctor takes the `worldFromCamera` animated transform (the
/// CTM at `Camera` time, inverted) plus the `RenderingCoordinateSystem`
/// option and computes:
///
/// * `world_from_render` — static `Transform`, used to map render-
///   space coordinates back to world coordinates for, e.g., scene
///   build plumbing.
/// * `render_from_camera` — `AnimatedTransform`, given to the camera
///   subclass so it can emit rays in render space.
#[derive(Debug, Clone)]
pub struct CameraTransform {
    render_from_camera: AnimatedTransform,
    world_from_render: Transform,
}

impl CameraTransform {
    /// pbrt-v4 `CameraTransform::CameraTransform(const
    /// AnimatedTransform &worldFromCamera)` (cameras.cpp:27). Picks
    /// `world_from_render` according to the rendering coordinate
    /// system and composes `render_from_camera = inv(world_from_render)
    /// * world_from_camera`.
    pub fn new(
        world_from_camera: &AnimatedTransform,
        space: RenderingCoordinateSystem,
    ) -> Option<Self> {
        let world_from_render = match space {
            RenderingCoordinateSystem::Camera => {
                let t_mid = 0.5 * (world_from_camera.times[0] + world_from_camera.times[1]);
                world_from_camera.interpolate(t_mid)
            }
            RenderingCoordinateSystem::CameraWorld => {
                // pbrt-v4 (cameras.cpp:35-40): translate render space
                // so that the camera origin at mid shutter lands at
                // the world origin.
                let t_mid = 0.5 * (world_from_camera.times[0] + world_from_camera.times[1]);
                let p_camera = world_from_camera
                    .interpolate(t_mid)
                    .transform_point(&Point3f::new(0.0, 0.0, 0.0));
                Transform::translate(p_camera.x, p_camera.y, p_camera.z)
            }
            RenderingCoordinateSystem::World => Transform::identity(),
        };
        let render_from_world = world_from_render.inverse();
        let rfc_start = render_from_world * world_from_camera.transforms[0];
        let rfc_end = render_from_world * world_from_camera.transforms[1];
        let render_from_camera = AnimatedTransform::new(
            &rfc_start,
            world_from_camera.times[0],
            &rfc_end,
            world_from_camera.times[1],
        )?;
        Some(CameraTransform {
            render_from_camera,
            world_from_render,
        })
    }

    /// pbrt-v4 `CameraTransform::RenderFromCamera() const`
    /// (cameras.h:87). Used by the Camera subclass to map rays from
    /// camera space to render space.
    pub fn render_from_camera(&self) -> &AnimatedTransform {
        &self.render_from_camera
    }

    /// pbrt-v4 `CameraTransform::WorldFromRender() const`
    /// (cameras.h:90). Used by the scene plumbing to pre-multiply
    /// every entity's `render_from_object` transform.
    pub fn world_from_render(&self) -> &Transform {
        &self.world_from_render
    }

    /// pbrt-v4 `CameraTransform::RenderFromWorld() const`
    /// (cameras.h:45). Convenience accessor — equals
    /// `Inverse(WorldFromRender)`. Computed lazily; for repeated use
    /// inside scene-build inner loops cache the result yourself.
    pub fn render_from_world(&self) -> Transform {
        self.world_from_render.inverse()
    }

    /// pbrt-v4 `CameraTransform::CameraFromWorld(Float time)`
    /// (cameras.h:51). Equivalent to
    /// `inv(world_from_render * render_from_camera.Interpolate(time))`.
    pub fn camera_from_world(&self, time: Float) -> Transform {
        (self.world_from_render * self.render_from_camera.interpolate(time)).inverse()
    }
}
