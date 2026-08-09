use super::display::*;
use crate::displays::*;

use crate::util::error::*;
// Includes cos_theta, abs_cos_theta, same_hemisphere, etc.

#[derive(Default)]
pub struct TevDisplay {
    display_item: Option<Box<DisplayItem>>,
    chan: Option<Box<IPCChannel>>,
}

impl TevDisplay {
    pub fn new() -> Self {
        return TevDisplay::default();
        //TevDisplay {
        //    display_item: None,
        //    chan: None,
    }

    pub fn connect(&mut self, hostname: &str) -> Result<(), PbrtError> {
        let mut chan = IPCChannel::new(hostname)?;
        chan.connect()?;
        self.chan = Some(Box::new(chan));
        return Ok(());
    }
}

impl Display for TevDisplay {
    fn start(
        &mut self,
        title: &str,
        resolution: &[usize; 2],
        channel_names: &[&str],
    ) -> Result<(), PbrtError> {
        if let Some(chan) = &mut self.chan {
            chan.connect()?;
            let display_item = Box::new(DisplayItem::new(title, resolution, channel_names));
            display_item.create_image(chan.as_mut())?;
            self.display_item = Some(display_item);
            return Ok(());
        } else {
            return Err(PbrtError::error("No channel"));
        }
    }
    fn update(&mut self, tile: &DisplayTile) -> Result<(), PbrtError> {
        if let Some(chan) = &mut self.chan {
            if let Some(item) = &self.display_item {
                let x = tile.x as u32;
                let y = tile.y as u32;
                let width = tile.width as u32;
                let height = tile.height as u32;
                return item.update_image(chan, x, y, width, height, &tile.buffer);
            } else {
                return Err(PbrtError::error("No image"));
            }
        } else {
            return Err(PbrtError::error("No channel"));
        }
    }

    fn end(&mut self) -> Result<(), PbrtError> {
        return Ok(());
    }
}
