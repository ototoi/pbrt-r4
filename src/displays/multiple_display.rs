use super::display::{Display, DisplayTile};
use crate::util::error::*;
use std::sync::Arc;
use std::sync::RwLock;

/// Fan-out wrapper that lets a `Film` hand a single `update` call to
/// every registered `Display`. All methods take `&self` so callers can
/// push tiles while only holding a `Film` read lock — no outer write
/// lock is required.
pub struct MultipleDisplay {
    displays: RwLock<Vec<Arc<RwLock<dyn Display>>>>,
}

impl MultipleDisplay {
    pub fn new() -> Self {
        MultipleDisplay {
            displays: RwLock::new(Vec::new()),
        }
    }

    pub fn add_display(&self, display: &Arc<RwLock<dyn Display>>) {
        self.displays.write().unwrap().push(display.clone());
    }

    pub fn len(&self) -> usize {
        self.displays.read().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.displays.read().unwrap().is_empty()
    }

    pub fn start(
        &self,
        title: &str,
        resolution: &[usize; 2],
        channel_names: &[&str],
    ) -> Result<(), PbrtError> {
        let displays = self.displays.read().unwrap();
        for d in displays.iter() {
            let mut display = d.write().unwrap();
            display.start(title, resolution, channel_names)?;
        }
        Ok(())
    }

    pub fn update(&self, tile: &DisplayTile) -> Result<(), PbrtError> {
        let displays = self.displays.read().unwrap();
        for d in displays.iter() {
            let mut display = d.write().unwrap();
            display.update(tile)?;
        }
        Ok(())
    }

    pub fn end(&self) -> Result<(), PbrtError> {
        let displays = self.displays.read().unwrap();
        for d in displays.iter() {
            let mut display = d.write().unwrap();
            display.end()?;
        }
        Ok(())
    }
}

unsafe impl Sync for MultipleDisplay {}
