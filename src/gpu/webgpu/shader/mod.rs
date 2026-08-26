mod fragment;
mod recipe;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ShaderStageId {
    PrepareCameraRays,
    GenerateCameraRays,
    IntersectClosest,
    HandleEscapedRays,
    EvaluateSurfaceInteraction,
    EvaluateMaterial,
    SampleDirectLighting,
    IntersectShadow,
    UpdateFilm,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShaderStage {
    pub id: ShaderStageId,
    pub entry_point: &'static str,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ShaderSet {
    pub source: String,
    pub label: &'static str,
    pub stages: Vec<ShaderStage>,
}

impl ShaderSet {
    pub fn stage(&self, id: ShaderStageId) -> Option<&ShaderStage> {
        self.stages.iter().find(|stage| stage.id == id)
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ShaderBuildError {
    Fragment(String),
}

impl std::fmt::Display for ShaderBuildError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Fragment(error) => {
                write!(formatter, "shader fragment composition failed: {error}")
            }
        }
    }
}

impl std::error::Error for ShaderBuildError {}

pub fn build_wavefront_shader_set() -> Result<ShaderSet, ShaderBuildError> {
    let recipe = recipe::wavefront::build_wavefront();
    let source = fragment::compose(&recipe.fragments, &recipe.roots)
        .map_err(|error| ShaderBuildError::Fragment(format!("{error:?}")))?;
    Ok(ShaderSet {
        source,
        label: recipe.label,
        stages: recipe.stages,
    })
}
