use crate::base::light::*; // Import the Light trait
use crate::cpu::primitive::*; // Import the Primitive trait
use crate::interaction::*; // Import the SurfaceInteraction type
use crate::util::base::*;
use crate::util::geometry::*; // Import the Bounds3f type

use std::sync::Arc;

pub struct Scene {
    pub lights: Vec<Arc<Light>>,
    pub infinite_lights: Vec<Arc<Light>>,
    pub aggregate: Arc<Primitive>,
    pub world_bound: Bounds3f,
}

impl Scene {
    pub fn new(aggregate: &Arc<Primitive>, lights: &[Arc<Light>]) -> Self {
        let world_bound = aggregate.bounds();
        let infinite_lights: Vec<Arc<Light>> =
            lights.iter().filter(|l| l.is_infinite()).cloned().collect();
        let scene = Scene {
            lights: lights.to_vec(),
            infinite_lights,
            aggregate: aggregate.clone(),
            world_bound,
        };
        for light in lights.iter() {
            let l = light.as_ref();
            l.preprocess(&scene.world_bound);
        }

        return scene;
    }

    pub fn world_bound(&self) -> Bounds3f {
        return self.world_bound;
    }

    pub fn intersect(&self, ray: &Ray, t_max: Float) -> Option<ShapeIntersection> {
        let aggregate = self.aggregate.as_ref();
        return aggregate.intersect(ray, t_max);
    }

    pub fn intersect_p(&self, ray: &Ray, t_max: Float) -> bool {
        let aggregate = self.aggregate.as_ref();
        return aggregate.intersect_p(ray, t_max);
    }
}

unsafe impl Sync for Scene {}
