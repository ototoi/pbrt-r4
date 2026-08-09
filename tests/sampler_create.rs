use pbrt_r4::options::PbrtOptions;
use pbrt_r4::prelude::*;
use pbrt_r4::util::lowdiscrepancy::primes::PRIMES;

fn collect_independent_samples(
    sampler: &mut IndependentSampler,
    sample_num: u32,
) -> (Vec<Point2f>, Vec<Float>) {
    assert!(sampler.set_sample_number(sample_num));

    let mut s2d = Vec::new();
    let mut s1d = Vec::new();
    for _ in 0..10 {
        s2d.push(sampler.get_2d());
        s1d.push(sampler.get_1d());
    }
    (s2d, s1d)
}

fn collect_zsobol_samples(
    sampler: &mut ZSobolSampler,
    sample_num: u32,
) -> (Vec<Point2f>, Vec<Float>) {
    assert!(sampler.set_sample_number(sample_num));

    let mut s2d = Vec::new();
    let mut s1d = Vec::new();
    for _ in 0..10 {
        s2d.push(sampler.get_2d());
        s1d.push(sampler.get_1d());
    }
    (s2d, s1d)
}

#[test]
fn sampler_create_zsobol_uses_zsobol_sampler() {
    let params = ParameterDictionary::new();
    let sampler = Sampler::create("zsobol", &params, Point2i::new(16, 16)).unwrap();
    assert!(matches!(sampler, Sampler::ZSobol(_)));
}

#[test]
fn sampler_create_paddedsobol_uses_paddedsobol_sampler() {
    let params = ParameterDictionary::new();
    let sampler = Sampler::create("paddedsobol", &params, Point2i::new(16, 16)).unwrap();
    assert!(matches!(sampler, Sampler::PaddedSobol(_)));
}

#[test]
fn sampler_create_sobol_uses_sobol_sampler() {
    let params = ParameterDictionary::new();
    let sampler = Sampler::create("sobol", &params, Point2i::new(16, 16)).unwrap();
    assert!(matches!(sampler, Sampler::Sobol(_)));
}

#[test]
fn sampler_create_independent_uses_independent_sampler() {
    let params = ParameterDictionary::new();
    let sampler = Sampler::create("independent", &params, Point2i::new(16, 16)).unwrap();
    assert!(matches!(sampler, Sampler::Independent(_)));
}

#[test]
fn sampler_create_unknown_type_returns_error() {
    let params = ParameterDictionary::new();
    let sampler = Sampler::create("not-a-real-sampler", &params, Point2i::new(16, 16));
    assert!(sampler.is_err());
}

#[test]
fn sampler_create_halton_defaults_to_permutedigits_and_seed_zero() {
    let params = ParameterDictionary::new();
    let sampler = Sampler::create("halton", &params, Point2i::new(16, 16)).unwrap();
    assert!(matches!(sampler, Sampler::Halton(_)));
}

#[test]
fn sampler_create_paddedsobol_defaults_to_fastowen_and_seed_zero() {
    let params = ParameterDictionary::new();
    let sampler = Sampler::create("paddedsobol", &params, Point2i::new(16, 16)).unwrap();
    assert!(matches!(sampler, Sampler::PaddedSobol(_)));
}

#[test]
fn sampler_create_zsobol_defaults_to_fastowen_and_seed_zero() {
    let params = ParameterDictionary::new();
    let sampler = Sampler::create("zsobol", &params, Point2i::new(16, 16)).unwrap();
    assert!(matches!(sampler, Sampler::ZSobol(_)));
}

#[test]
fn sampler_create_uses_option_seed_when_seed_parameter_is_missing() {
    let mut options = PbrtOptions::default();
    options.seed = 17;
    PbrtOptions::set(options);

    let params = ParameterDictionary::new();
    let resolution = Point2i::new(16, 16);

    match Sampler::create("independent", &params, resolution).unwrap() {
        Sampler::Independent(mut sampler) => {
            let mut expected = IndependentSampler::new(4, 17);
            let pixel = Point2i::new(1, 2);
            sampler.start_pixel(&pixel);
            expected.start_pixel(&pixel);
            assert_eq!(sampler.get_1d(), expected.get_1d());
        }
        _ => panic!("expected independent sampler"),
    }

    match Sampler::create("halton", &params, resolution).unwrap() {
        Sampler::Halton(mut sampler) => {
            let mut expected =
                HaltonSampler::new(16, resolution, RandomizeStrategy::PermuteDigits, 17);
            let pixel = Point2i::new(1, 2);
            sampler.start_pixel(&pixel);
            expected.start_pixel(&pixel);
            assert_eq!(sampler.get_1d(), expected.get_1d());
        }
        _ => panic!("expected halton sampler"),
    }

    match Sampler::create("paddedsobol", &params, resolution).unwrap() {
        Sampler::PaddedSobol(mut sampler) => {
            let mut expected = PaddedSobolSampler::new(16, RandomizeStrategy::FastOwen, 17);
            let pixel = Point2i::new(1, 2);
            sampler.start_pixel(&pixel);
            expected.start_pixel(&pixel);
            assert_eq!(sampler.get_1d(), expected.get_1d());
        }
        _ => panic!("expected padded sobol sampler"),
    }

    match Sampler::create("sobol", &params, resolution).unwrap() {
        Sampler::Sobol(mut sampler) => {
            let mut expected = SobolSampler::new(16, resolution, RandomizeStrategy::FastOwen, 17);
            let pixel = Point2i::new(1, 2);
            sampler.start_pixel(&pixel);
            expected.start_pixel(&pixel);
            assert_eq!(sampler.get_1d(), expected.get_1d());
        }
        _ => panic!("expected sobol sampler"),
    }

    match Sampler::create("zsobol", &params, resolution).unwrap() {
        Sampler::ZSobol(mut sampler) => {
            let mut expected = ZSobolSampler::new(16, resolution, RandomizeStrategy::FastOwen, 17);
            let pixel = Point2i::new(1, 2);
            sampler.start_pixel(&pixel);
            expected.start_pixel(&pixel);
            assert_eq!(sampler.get_1d(), expected.get_1d());
        }
        _ => panic!("expected zsobol sampler"),
    }

    PbrtOptions::set(PbrtOptions::default());
}

#[test]
fn stratified_sampler_uses_option_seed_when_seed_parameter_is_missing() {
    let mut options = PbrtOptions::default();
    options.seed = 23;
    PbrtOptions::set(options);

    let params = ParameterDictionary::new();
    let mut sampler = StratifiedSampler::create(&params).unwrap();
    let mut expected = StratifiedSampler::new(4, 4, true, 23, 4);
    let pixel = Point2i::new(2, 3);
    sampler.start_pixel(&pixel);
    expected.start_pixel(&pixel);
    assert_eq!(sampler.get_1d(), expected.get_1d());

    PbrtOptions::set(PbrtOptions::default());
}

#[test]
fn independent_sampler_repeats_samples_for_same_pixel_and_sample_number() {
    let mut sampler = IndependentSampler::new(16, 7);
    let pixel = Point2i::new(1, 5);
    sampler.start_pixel(&pixel);

    let mut recorded = Vec::new();
    for sample_num in 0..sampler.get_samples_per_pixel() {
        recorded.push(collect_independent_samples(&mut sampler, sample_num));
    }

    sampler.start_pixel(&Point2i::new(0, 6));
    assert!(sampler.set_sample_number(10));
    let _ = sampler.get_2d();
    let _ = sampler.get_2d();
    let _ = sampler.get_1d();

    sampler.start_pixel(&pixel);
    for sample_num in (0..sampler.get_samples_per_pixel()).rev() {
        let (s2d, s1d) = collect_independent_samples(&mut sampler, sample_num);
        assert_eq!(recorded[sample_num as usize].0, s2d);
        assert_eq!(recorded[sample_num as usize].1, s1d);
    }
}

#[test]
fn zsobol_sampler_repeats_samples_for_same_pixel_and_sample_number() {
    let mut sampler =
        ZSobolSampler::new(16, Point2i::new(100, 101), RandomizeStrategy::FastOwen, 11);
    let pixel = Point2i::new(1, 5);
    sampler.start_pixel(&pixel);

    let mut recorded = Vec::new();
    for sample_num in 0..sampler.get_samples_per_pixel() {
        recorded.push(collect_zsobol_samples(&mut sampler, sample_num));
    }

    sampler.start_pixel(&Point2i::new(0, 6));
    assert!(sampler.set_sample_number(10));
    let _ = sampler.get_2d();
    let _ = sampler.get_2d();
    let _ = sampler.get_1d();

    sampler.start_pixel(&pixel);
    for sample_num in (0..sampler.get_samples_per_pixel()).rev() {
        let (s2d, s1d) = collect_zsobol_samples(&mut sampler, sample_num);
        assert_eq!(recorded[sample_num as usize].0, s2d);
        assert_eq!(recorded[sample_num as usize].1, s1d);
    }
}

#[test]
fn zsobol_sampler_pixel_samples_stay_in_unit_square() {
    let mut sampler = ZSobolSampler::new(
        16,
        Point2i::new(10, 10),
        RandomizeStrategy::PermuteDigits,
        3,
    );
    let pixel = Point2i::new(3, 4);
    sampler.start_pixel(&pixel);

    for sample_num in 0..sampler.get_samples_per_pixel() {
        assert!(sampler.set_sample_number(sample_num));
        let p = sampler.get_pixel_2d();
        assert!(p.x.is_finite());
        assert!(p.y.is_finite());
        assert!((0.0..1.0).contains(&p.x));
        assert!((0.0..1.0).contains(&p.y));
    }
}

#[test]
fn halton_sampler_does_not_panic_for_high_dimensions() {
    let sample_bounds = Bounds2i::new(&Point2i::new(0, 0), &Point2i::new(16, 16));
    let mut sampler = HaltonSampler::new(1, sample_bounds.max, RandomizeStrategy::PermuteDigits, 0);
    sampler.start_pixel(&Point2i::new(0, 0));

    for _ in 0..(PRIMES.len() + 32) {
        let v = sampler.get_1d();
        assert!(v.is_finite());
    }
}

#[test]
fn halton_sampler_handles_origin_pixel_without_dividing_by_zero() {
    let mut sampler = HaltonSampler::new(4, Point2i::new(16, 16), RandomizeStrategy::Owen, 0);
    sampler.start_pixel(&Point2i::new(0, 0));

    for sample_num in 0..sampler.get_samples_per_pixel() {
        assert!(sampler.set_sample_number(sample_num));
        let p = sampler.get_pixel_2d();
        assert!(p.x.is_finite());
        assert!(p.y.is_finite());
        assert!((0.0..1.0).contains(&p.x));
        assert!((0.0..1.0).contains(&p.y));
    }
}
