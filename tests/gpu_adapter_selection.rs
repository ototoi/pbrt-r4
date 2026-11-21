#![cfg(feature = "webgpu")]

use pbrt_r4::gpu::webgpu::context::{preferred_adapter, BASE_REQUIRED_FEATURES};
use wgpu::DeviceType::{Cpu, DiscreteGpu, IntegratedGpu};

#[test]
fn prefers_a_capable_adapter_then_a_discrete_one() {
    let none = wgpu::Features::empty();
    let all = BASE_REQUIRED_FEATURES;
    assert_eq!(preferred_adapter(&[]), None);
    assert_eq!(preferred_adapter(&[(none, DiscreteGpu)]), None);
    assert_eq!(
        preferred_adapter(&[(all, IntegratedGpu), (none, DiscreteGpu)]),
        Some(0)
    );
    assert_eq!(
        preferred_adapter(&[(all, Cpu), (all, IntegratedGpu), (all, DiscreteGpu)]),
        Some(2)
    );
    assert_eq!(
        preferred_adapter(&[(all, DiscreteGpu), (all, DiscreteGpu)]),
        Some(0)
    );
}
