use pbrt_r4::util::base::Point2f;
use pbrt_r4::util::geometry::Bounds2f;
use pbrt_r4::util::sampling::summed_area_table::SummedAreaTable;

#[test]
fn integral_recovers_uniform_field() {
    let values = vec![1.0; 16];
    let sat = SummedAreaTable::new(&values, 4, 4);
    let b = Bounds2f::new(&Point2f::new(0.0, 0.0), &Point2f::new(1.0, 1.0));
    assert!((sat.integral(&b) - 1.0).abs() < 1e-6);
}

#[test]
fn integral_zero_outside_function() {
    let values = vec![1.0; 16];
    let sat = SummedAreaTable::new(&values, 4, 4);
    let b = Bounds2f::new(&Point2f::new(0.0, 0.0), &Point2f::new(0.5, 0.5));
    assert!((sat.integral(&b) - 0.25).abs() < 1e-6);
}
