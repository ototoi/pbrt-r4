use crate::util::base::*;

pub trait MicrofacetDistribution {
    fn d(&self, wh: &Vector3f) -> Float;
    fn lambda(&self, w: &Vector3f) -> Float;
    fn g1(&self, w: &Vector3f) -> Float {
        return 1.0 / (1.0 + self.lambda(w));
    }
    fn g(&self, wo: &Vector3f, wi: &Vector3f) -> Float {
        return 1.0 / (1.0 + self.lambda(wo) + self.lambda(wi));
    }
    fn sample_wh(&self, wo: &Vector3f, u: &Vector2f) -> Vector3f;
    fn pdf(&self, wo: &Vector3f, wh: &Vector3f) -> Float;
}
