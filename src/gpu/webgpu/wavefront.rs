//! Host-side layout for the GPU wavefront work queues.

const ALIGNMENT: u32 = 16;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum QueueKind {
    Camera,
    Hit,
    Miss,
    Shadow,
    Film,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct QueueRegion {
    pub offset: u32,
    pub stride: u32,
    pub capacity: u32,
}

impl QueueRegion {
    pub fn byte_len(self) -> u32 {
        self.stride.saturating_mul(self.capacity)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WavefrontLayout {
    pub camera: QueueRegion,
    pub hit: QueueRegion,
    pub miss: QueueRegion,
    pub shadow: QueueRegion,
    pub film: QueueRegion,
    pub byte_len: u32,
}

impl WavefrontLayout {
    pub fn for_pixel_count(pixel_count: u32) -> Option<Self> {
        let mut offset = 0;
        let camera = allocate(&mut offset, 32, pixel_count)?;
        let hit = allocate(&mut offset, 48, pixel_count)?;
        let miss = allocate(&mut offset, 16, pixel_count)?;
        let shadow = allocate(&mut offset, 32, pixel_count)?;
        let film = allocate(&mut offset, 16, pixel_count)?;
        Some(Self {
            camera,
            hit,
            miss,
            shadow,
            film,
            byte_len: offset,
        })
    }

    pub fn region(self, kind: QueueKind) -> QueueRegion {
        match kind {
            QueueKind::Camera => self.camera,
            QueueKind::Hit => self.hit,
            QueueKind::Miss => self.miss,
            QueueKind::Shadow => self.shadow,
            QueueKind::Film => self.film,
        }
    }
}

fn allocate(offset: &mut u32, stride: u32, capacity: u32) -> Option<QueueRegion> {
    *offset = align(*offset, ALIGNMENT)?;
    let region = QueueRegion {
        offset: *offset,
        stride,
        capacity,
    };
    *offset = (*offset).checked_add(region.byte_len())?;
    Some(region)
}

fn align(value: u32, alignment: u32) -> Option<u32> {
    value
        .checked_add(alignment - 1)
        .map(|value| value / alignment * alignment)
}
