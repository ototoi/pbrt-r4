use super::medium::Medium;

use std::sync::Arc;
//use std::sync::Weak;

#[derive(Clone, Default, Debug)]
pub struct MediumInterface {
    pub inside: Option<Arc<Medium>>,
    pub outside: Option<Arc<Medium>>,
}

impl MediumInterface {
    pub fn new() -> Self {
        MediumInterface {
            inside: None,
            outside: None,
        }
    }

    pub fn set_inside(&mut self, medium: &Arc<Medium>) {
        self.inside = Some(Arc::clone(medium));
    }

    pub fn set_outside(&mut self, medium: &Arc<Medium>) {
        self.outside = Some(Arc::clone(medium));
    }

    pub fn get_inside(&self) -> Option<Arc<Medium>> {
        match &self.inside {
            Some(inside) => Some(inside.clone()),
            None => None,
        }
    }

    pub fn get_outside(&self) -> Option<Arc<Medium>> {
        match &self.outside {
            Some(outside) => Some(outside.clone()),
            None => None,
        }
    }

    pub fn is_medium_transition(&self) -> bool {
        match (self.inside.as_ref(), self.outside.as_ref()) {
            (Some(inside), Some(outside)) => {
                return !std::ptr::eq(inside, outside);
            }
            (Some(_), None) => true,
            (None, Some(_)) => true,
            (None, None) => false,
        }
    }
}

impl From<&Option<Arc<Medium>>> for MediumInterface {
    fn from(medium: &Option<Arc<Medium>>) -> Self {
        match medium {
            Some(medium) => MediumInterface::from(medium),
            None => MediumInterface::new(),
        }
    }
}

impl From<&Arc<Medium>> for MediumInterface {
    fn from(medium: &Arc<Medium>) -> Self {
        let inside = Arc::clone(medium);
        let outside = Arc::clone(medium);
        MediumInterface {
            inside: Some(inside),
            outside: Some(outside),
        }
    }
}
