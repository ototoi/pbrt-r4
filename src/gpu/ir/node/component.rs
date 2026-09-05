use super::accelerator::Accelerator;
use super::area_light::AreaLightComponent;
use super::camera::Camera;
use super::film::Film;
use super::filter::Filter;
use super::instance::Instance;
use super::integrator::Integrator;
use super::light::Light;
use super::material::Material;
use super::medium::Medium;
use super::output::Output;
use super::sampler::Sampler;
use super::scene::Scene;
use super::shape::Shape;

use std::sync::Arc;

#[derive(Clone)]
pub struct SceneComponent {
    pub scene: Scene,
}

#[derive(Clone)]
pub struct SamplerComponent {
    pub sampler: Sampler,
}

#[derive(Clone)]
pub struct IntegratorComponent {
    pub integrator: Integrator,
}

#[derive(Clone)]
pub struct AcceleratorComponent {
    pub accelerator: Accelerator,
}

#[derive(Clone)]
pub struct FilterComponent {
    pub filter: Filter,
}

#[derive(Clone)]
pub struct FilmComponent {
    pub film: Film,
}

#[derive(Clone)]
pub struct OutputComponent {
    pub output: Output,
}

#[derive(Clone)]
pub struct CameraComponent {
    pub camera: Camera,
}

#[derive(Clone)]
pub struct ShapeComponent {
    pub shape: Shape,
    pub reverse_orientation: bool,
}

#[derive(Clone)]
pub struct MaterialComponent {
    pub material: Arc<Material>,
}

#[derive(Clone)]
pub struct LightComponent {
    pub light: Light,
}

#[derive(Clone)]
pub struct MediumComponent {
    pub medium: Medium,
}

#[derive(Clone)]
pub struct InstanceComponent {
    pub instance: Instance,
}

pub enum Component {
    Scene(SceneComponent),
    Sampler(SamplerComponent),
    Integrator(IntegratorComponent),
    Accelerator(AcceleratorComponent),
    AreaLight(AreaLightComponent),
    Filter(FilterComponent),
    Camera(CameraComponent),
    Film(FilmComponent),
    Output(OutputComponent),
    Shape(ShapeComponent),
    Material(MaterialComponent),
    Light(LightComponent),
    Medium(MediumComponent),
    Instance(InstanceComponent),
}
