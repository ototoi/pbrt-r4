use super::lightdistrib::*;
use crate::cpu::integrators::IntegratorBase;
use crate::util::base::*;
use crate::util::sampling::*;

use std::sync::Arc;

pub struct UniformLightDistribution {
    distrib: Arc<Distribution1D>,
}

impl UniformLightDistribution {
    pub fn new(base: &IntegratorBase) -> Self {
        let prob = vec![1.0; base.lights.len()];
        UniformLightDistribution {
            distrib: Arc::new(Distribution1D::new(&prob)),
        }
    }
}

impl LightDistribution for UniformLightDistribution {
    fn lookup(&self, _p: &Point3f) -> Arc<Distribution1D> {
        return self.distrib.clone();
    }
}
