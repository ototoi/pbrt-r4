use crate::util::error::*;

use log::*;
use std::{
    io::{BufWriter, Write},
    net::{SocketAddr, TcpStream, ToSocketAddrs},
};

enum DisplayDirective {
    _OpenImage = 7,
    _ReloadImage = 1,
    _CloseImage = 2,
    CreateImage = 4,
    UpdateImage = 6,
    _VectorGraphics = 8,
}

pub struct IPCChannel {
    addr: SocketAddr,
    stream: Option<TcpStream>,
}

impl IPCChannel {
    pub fn new(hostname: &str) -> Result<Self, PbrtError> {
        let split: Vec<_> = hostname.split(':').collect();
        if split.len() >= 2 {
            let address = split[0..split.len() - 1].join(":");
            let port = String::from(split[split.len() - 1]);
            let host_and_port = format!("{}:{}", address, port);
            let mut addrs = host_and_port.to_socket_addrs().unwrap();
            if let Some(addr) = addrs.find(|x| (*x).is_ipv4()) {
                let s = TcpStream::connect(addr);
                match s {
                    Ok(stream) => {
                        let channel = IPCChannel {
                            addr,
                            stream: Some(stream),
                        };
                        return Ok(channel);
                    }
                    Err(e) => {
                        return Err(PbrtError::from(e));
                    }
                }
            } else {
                return Err(PbrtError::error("2"));
            }
        } else {
            let msg = format!(
                "Expected \"host:port\" for display server address. Given \"{}\".",
                hostname
            );
            return Err(PbrtError::error(&msg));
        }
    }

    pub fn send(&mut self, message: &[u8]) -> Result<(), PbrtError> {
        if self.stream.is_none() {
            self.connect()?;
        }
        if let Some(stream) = self.stream.as_ref() {
            let mut writer = BufWriter::new(stream);
            match writer.write_all(message) {
                Ok(_) => {
                    return Ok(());
                }
                Err(e) => {
                    return Err(PbrtError::from(e));
                }
            }
        } else {
            let msg = format!("send to display server failed: {}", "");
            return Err(PbrtError::from(msg));
        }
    }

    pub fn connected(&mut self) -> Result<(), PbrtError> {
        if let Some(stream) = self.stream.as_ref() {
            match stream.take_error() {
                Ok(_) => {
                    return Ok(());
                }
                Err(e) => {
                    return Err(PbrtError::from(e));
                }
            }
        } else {
            let msg = String::from("not connected.");
            return Err(PbrtError::from(msg));
        }
    }

    pub fn connect(&mut self) -> Result<(), PbrtError> {
        if self.stream.is_none() {
            match TcpStream::connect(self.addr) {
                Ok(stream) => {
                    self.stream = Some(stream);
                    return Ok(());
                }
                Err(e) => {
                    return Err(PbrtError::from(e));
                }
            }
        } else {
            return Ok(());
        }
    }

    pub fn disconnect(&mut self) -> Result<(), PbrtError> {
        if let Some(stream) = self.stream.as_ref() {
            match stream.shutdown(std::net::Shutdown::Both) {
                Ok(_) => {
                    self.stream = None;
                    return Ok(());
                }
                Err(e) => {
                    return Err(PbrtError::from(e));
                }
            }
        } else {
            return Ok(());
        }
    }
}

impl Drop for IPCChannel {
    fn drop(&mut self) {
        match self.disconnect() {
            Ok(_) => {}
            Err(e) => {
                error!("{}", e);
            }
        }
    }
}

pub struct IPCGen {}

impl IPCGen {
    pub fn create_image(name: &str, width: u32, height: u32, channel_names: &[&str]) -> Vec<u8> {
        let grab_focus: u8 = 1;

        let mut buffer = Vec::new();
        buffer.write_all(&0_i32.to_le_bytes()).unwrap();
        buffer
            .write_all(&(DisplayDirective::CreateImage as u8).to_le_bytes())
            .unwrap(); //5 +1

        buffer.write_all(&grab_focus.to_le_bytes()).unwrap();
        buffer.write_all(name.as_bytes()).unwrap();
        buffer.write_all(&0_u8.to_le_bytes()).unwrap();

        let n_channels = channel_names.len() as u32;
        buffer.write_all(&width.to_le_bytes()).unwrap();
        buffer.write_all(&height.to_le_bytes()).unwrap();
        buffer.write_all(&n_channels.to_le_bytes()).unwrap();

        for c in 0..channel_names.len() {
            buffer.write_all(channel_names[c].as_bytes()).unwrap();
            buffer.write_all(&0_u8.to_le_bytes()).unwrap();
        }

        let length = (buffer.len() as u32).to_le_bytes();
        buffer[0..4].copy_from_slice(&length[0..4]);
        return buffer;
    }

    pub fn update_image(
        name: &str,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        image: &[f32],
        channel_names: &[&str],
    ) -> Vec<u8> {
        let grab_focus: u8 = 0;

        let mut buffer = Vec::new();
        buffer.write_all(&0_i32.to_le_bytes()).unwrap(); //4 +4
        buffer
            .write_all(&(DisplayDirective::UpdateImage as u8).to_le_bytes())
            .unwrap(); //5 +1
        buffer.write_all(&grab_focus.to_le_bytes()).unwrap(); //6 +1
        buffer.write_all(name.as_bytes()).unwrap(); //6+tsz +tsz
        buffer.write_all(&0_u8.to_le_bytes()).unwrap(); //7+tsz +1

        let n_channels = channel_names.len();
        buffer
            .write_all(&(n_channels as u32).to_le_bytes())
            .unwrap();

        for i in 0..n_channels {
            let channel_name = channel_names[i];
            buffer.write_all(channel_name.as_bytes()).unwrap(); //7+tsz+csz +csz
            buffer.write_all(&[0_u8]).unwrap(); //8+tsz+csz +1
        }

        buffer.write_all(&x.to_le_bytes()).unwrap();
        buffer.write_all(&y.to_le_bytes()).unwrap();
        buffer.write_all(&width.to_le_bytes()).unwrap();
        buffer.write_all(&height.to_le_bytes()).unwrap();

        for i in 0..n_channels {
            buffer.write_all(&(i as i64).to_le_bytes()).unwrap();
        }

        for _ in 0..n_channels {
            buffer
                .write_all(&(n_channels as i64).to_le_bytes())
                .unwrap();
        }

        let mut image_buffer: Vec<u8> = Vec::new();
        for y in 0..height {
            for x in 0..width {
                let index = (y * width + x) as usize;
                for c in 0..n_channels {
                    let f = image[n_channels * index + c];
                    image_buffer.write_all(&f.to_le_bytes()).unwrap();
                }
            }
        }
        buffer.append(&mut image_buffer);

        let length = (buffer.len() as u32).to_le_bytes();
        buffer[0..4].copy_from_slice(&length[0..4]);
        return buffer;
    }
}

pub const TILE_SIZE: usize = 128;

struct Tile {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
    pub buffer: Vec<f32>,
}

pub struct DisplayItem {
    pub title: String,
    pub resolution: [usize; 2],
    pub channel_names: Vec<String>,
}

impl DisplayItem {
    pub fn new(title: &str, resolution: &[usize; 2], channel_names: &[&str]) -> Self {
        let mut v_channel_names = Vec::new();
        for name in channel_names {
            v_channel_names.push(String::from(*name));
        }
        let item = DisplayItem {
            title: String::from(title),
            resolution: *resolution,
            channel_names: v_channel_names,
        };
        return item;
    }

    fn gen_tiles(
        &self,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        image: &[f32],
    ) -> Result<Vec<Tile>, PbrtError> {
        assert!(width * height * self.channel_names.len() as u32 == image.len() as u32);

        let mut xranges = Vec::new();
        let mut yranges = Vec::new();

        let tile_size = TILE_SIZE as u32;
        if width <= tile_size {
            xranges.push((0, width));
        } else {
            let xx1 = width + tile_size - 1;
            for i in (0..xx1).step_by(tile_size as usize) {
                let t0 = i;
                let t1 = u32::min(t0 + tile_size, width);
                if t0 < t1 {
                    xranges.push((t0, t1));
                }
            }
        }
        if height <= tile_size {
            yranges.push((0, height));
        } else {
            let yy1 = height + tile_size - 1;
            for i in (0..yy1).step_by(tile_size as usize) {
                let t0 = i;
                let t1 = u32::min(t0 + tile_size, height);
                if t0 < t1 {
                    yranges.push((t0, t1));
                }
            }
        }

        let mut tiles = Vec::with_capacity(xranges.len() * yranges.len());
        let channel_count = self.channel_names.len();
        for (yt0, yt1) in yranges.iter() {
            for (xt0, xt1) in xranges.iter() {
                let nw = xt1 - xt0;
                let nh = yt1 - yt0;

                let mut buffer: Vec<f32> = vec![0.0; (nw * nh) as usize * channel_count];
                for j in 0..nh {
                    for i in 0..nw {
                        let dst_index = (j * nw + i) as usize;
                        let src_index = ((yt0 + j) * width + (xt0 + i)) as usize;
                        for c in 0..channel_count {
                            buffer[channel_count * dst_index + c] =
                                image[channel_count * src_index + c];
                        }
                    }
                }

                let tile = Tile {
                    x: x + xt0,
                    y: y + yt0,
                    width: nw,
                    height: nh,
                    buffer,
                };
                tiles.push(tile);
            }
        }
        return Ok(tiles);
    }

    pub fn update_image(
        &self,
        chan: &mut IPCChannel,
        x: u32,
        y: u32,
        width: u32,
        height: u32,
        image: &[f32],
    ) -> Result<(), PbrtError> {
        let name = &self.title;
        let mut channel_names: Vec<&str> = Vec::new();
        for name in &self.channel_names {
            channel_names.push(name);
        }
        chan.connect()?;
        let tiles = self.gen_tiles(x, y, width, height, image)?;
        for tile in tiles {
            let buffer = IPCGen::update_image(
                name,
                tile.x,
                tile.y,
                tile.width,
                tile.height,
                &tile.buffer,
                &channel_names,
            );
            chan.send(&buffer)?;
        }
        return Ok(());
    }

    pub fn create_image(&self, chan: &mut IPCChannel) -> Result<(), PbrtError> {
        let name = &self.title;
        let width = self.resolution[0] as u32;
        let height = self.resolution[1] as u32;
        let mut channel_names: Vec<&str> = Vec::new();
        for name in &self.channel_names {
            channel_names.push(name);
        }
        let buffer = IPCGen::create_image(name, width, height, &channel_names);
        return chan.send(&buffer);
    }
}
