#[derive(Clone, Debug, Default, PartialEq)]
pub struct Geometry {
    pub first_vertex: u32,
    pub vertex_count: u32,
    pub first_index: u32,
    pub index_count: u32,
}
