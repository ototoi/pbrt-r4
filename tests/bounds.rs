// Imported from bounds.cpp

use pbrt_r4::prelude::*;

#[test]
fn bounds2_iterator_basic() {
    let b = Bounds2i::from(((0, 1), (2, 3)));
    let e = [
        Point2i::new(0, 1),
        Point2i::new(1, 1),
        Point2i::new(0, 2),
        Point2i::new(1, 2),
    ];
    for (offset, p) in b.into_iter().enumerate() {
        println!("{}: {:?}", offset, p);
        assert!(offset < e.len());
        assert_eq!(e[offset], p);
    }
}

#[test]
fn bounds2_iterator_degenerate() {
    {
        let b = Bounds2i::from(((0, 0), (0, 10)));
        for p in b.into_iter() {
            // This loop should never run.
            assert!(false, "p = {:?}", p);
        }
    }
    {
        let b = Bounds2i::from(((0, 0), (4, 0)));
        for p in b.into_iter() {
            // This loop should never run.
            assert!(false, "p = {:?}", p);
        }
    }
    {
        let b = Bounds2i::default();
        for p in b.into_iter() {
            // This loop should never run.
            assert!(false, "p = {:?}", p);
        }
    }
}

#[test]
fn bounds3_point_distance() {
    {
        let b = Bounds3f::from(((0.0, 0.0, 0.0), (1.0, 1.0, 1.0)));

        // Points inside the bounding box or on faces
        assert_eq!(0.0, b.distance(&Point3f::new(0.5, 0.5, 0.5)));
        assert_eq!(0.0, b.distance(&Point3f::new(0.0, 1.0, 1.0)));
        assert_eq!(0.0, b.distance(&Point3f::new(0.25, 0.8, 1.0)));
        assert_eq!(0.0, b.distance(&Point3f::new(0.0, 0.25, 0.8)));
        assert_eq!(0.0, b.distance(&Point3f::new(0.7, 0.0, 0.8)));

        // Aligned with the plane of one of the faces
        assert_eq!(5.0, b.distance(&Point3f::new(6.0, 1.0, 1.0)));
        assert_eq!(10.0, b.distance(&Point3f::new(0.0, -10.0, 1.0)));

        // 2 of the dimensions inside the box's extent
        assert_eq!(2.0, b.distance(&Point3f::new(0.5, 0.5, 3.0)));
        assert_eq!(3.0, b.distance(&Point3f::new(0.5, 0.5, -3.0)));
        assert_eq!(2.0, b.distance(&Point3f::new(0.5, 3.0, 0.5)));
        assert_eq!(3.0, b.distance(&Point3f::new(0.5, -3.0, 0.5)));
        assert_eq!(2.0, b.distance(&Point3f::new(3.0, 0.5, 0.5)));
        assert_eq!(3.0, b.distance(&Point3f::new(-3.0, 0.5, 0.5)));

        // General points
        assert_eq!(
            (3 * 3 + 7 * 7 + 10 * 10) as Float,
            b.distance_squared(&Point3f::new(4.0, 8.0, -10.0))
        );
        assert_eq!(
            (6 * 6 + 10 * 10 + 7 * 7) as Float,
            b.distance_squared(&Point3f::new(-6.0, -10.0, 8.0))
        );
    }
    {
        // A few with a more irregular box, just to be sure
        let b = Bounds3f::from(((-1.0, -3.0, 5.0), (2.0, -2.0, 18.0)));
        assert_eq!(0.0, b.distance(&Point3f::new(-0.99, -2.0, 5.0)));
        assert_eq!(
            (2 * 2 + 6 * 6 + 4 * 4) as Float,
            b.distance_squared(&Point3f::new(-3.0, -9.0, 22.0))
        );
    }
}

#[test]
fn bounds2_union() {
    let a = Bounds2f::from(((-10.0, -10.0), (0.0, 20.0)));
    let b = Bounds2f::default();
    let c = a.union(&b);
    assert_eq!(a, c);

    assert_eq!(b, b.union(&b));

    let d = Bounds2f::from((-15.0, 10.0));
    let e = a.union(&d);
    assert_eq!(Bounds2f::from(((-15.0, -10.0), (0.0, 20.0))), e);
}

#[test]
fn bounds3_union() {
    let a = Bounds3f::from(((-10.0, -10.0, 5.0), (0.0, 20.0, 10.0)));
    let b = Bounds3f::default();
    let c = a.union(&b);
    assert_eq!(a, c);

    assert_eq!(b, b.union(&b));

    let d = Bounds3f::from((-15.0, 10.0, 30.0));
    let e = a.union(&d);
    assert_eq!(Bounds3f::from(((-15.0, -10.0, 5.0), (0.0, 20.0, 30.0))), e);
}
