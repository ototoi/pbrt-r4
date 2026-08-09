#[cfg(target_arch = "x86_64")]
pub mod qbvh_x86;
#[cfg(target_arch = "x86_64")]
pub use qbvh_x86::QBVHAccel;

#[cfg(target_arch = "aarch64")]
pub mod qbvh_arm;
#[cfg(target_arch = "aarch64")]
pub use qbvh_arm::QBVHAccel;
