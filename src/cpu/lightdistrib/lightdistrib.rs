use crate::util::base::*;
use crate::util::sampling::*;

use std::sync::Arc;

// LightDistribution defines a general interface for classes that provide
// probability distributions for sampling light sources at a given point in
// space.
pub trait LightDistribution: Sync + Send {
    fn lookup(&self, p: &Point3f) -> Arc<Distribution1D>;
}
