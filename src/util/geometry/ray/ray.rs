use crate::media::*;
use crate::util::base::*;
use std::sync::Arc;

#[derive(Debug, Default, Clone)]
pub struct Ray {
    pub o: Point3f,
    pub d: Vector3f,

    pub time: Float,
    pub medium: Option<Arc<Medium>>,
}

impl Ray {
    pub fn new(o: &Point3f, d: &Vector3f, _t_max: Float, time: Float) -> Self {
        let o = *o;
        let d = *d;
        Ray {
            o,
            d,
            time,
            medium: None,
        }
    }

    pub fn zero() -> Self {
        Ray {
            o: Vector3f::new(0.0, 0.0, 0.0),
            d: Vector3f::new(0.0, 0.0, 0.0),
            time: 0.0,
            medium: None,
        }
    }

    pub fn position(&self, t: Float) -> Point3f {
        return self.o + self.d * t;
    }
}

impl From<(&Point3f, &Vector3f, Float, Float, &Option<Arc<Medium>>)> for Ray {
    fn from(value: (&Point3f, &Vector3f, Float, Float, &Option<Arc<Medium>>)) -> Self {
        Ray {
            o: *value.0,
            d: *value.1,
            time: value.3,
            medium: value.4.clone(),
        }
    }
}
