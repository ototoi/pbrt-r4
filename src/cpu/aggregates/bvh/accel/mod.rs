pub mod lbvh;
pub mod qbvh;

#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
//#[cfg(any(target_arch = "x86_64"))]
pub type BVHAccel = qbvh::QBVHAccel;
#[cfg(any(target_arch = "x86_64", target_arch = "aarch64"))]
pub type BVHTree = qbvh::QBVHTree;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
//#[cfg(not(any(target_arch = "x86_64")))]
pub type BVHAccel = lbvh::LBVHAccel;
#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub type BVHTree = lbvh::LBVHTree;
