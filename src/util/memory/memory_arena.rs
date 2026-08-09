pub struct MemoryArena {}

impl MemoryArena {
    pub fn new() -> Self {
        MemoryArena {}
    }

    pub fn reset(&mut self) {}
}

unsafe impl Send for MemoryArena {}
unsafe impl Sync for MemoryArena {}
