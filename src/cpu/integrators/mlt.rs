// MLT uses three v4-compatible sample streams on top of the BDPT subpath
// helpers. Debug sampler, display server, and intermediate pixel statistics
// remain outside this CPU implementation.

use crate::base::camera::{Camera, CameraSample};
use crate::base::lightsampler::LightSampler;
use crate::base::sampler::Sampler;
use crate::cpu::integrators::bdpt::{
    connect_bdpt, generate_camera_subpath, generate_light_subpath, Vertex,
};
use crate::cpu::integrators::*;
use crate::film::Film;
use crate::options::PbrtOptions;
use crate::paramdict::*;
use crate::samplers::MLTSampler;
use crate::scene::*;
use crate::util::base::*;
use crate::util::error::*;
use crate::util::geometry::*;
use crate::util::memory::*;
use crate::util::rng::RNG;
use crate::util::sampling::Distribution1D;
use crate::util::spectrum::*;

use rayon::prelude::*;
use std::sync::{Arc, RwLock};

const CAMERA_STREAM_INDEX: u64 = 0;
const LIGHT_STREAM_INDEX: u64 = 1;
const CONNECTION_STREAM_INDEX: u64 = 2;
const N_SAMPLE_STREAMS: u64 = 3;

pub struct MLTIntegrator {
    base: IntegratorBase,
    camera: Arc<Camera>,
    max_depth: i32,
    n_bootstrap: i32,
    n_chains: i32,
    mutations_per_pixel: i32,
    sigma: Float,
    large_step_probability: Float,
    regularize: bool,
    light_sample_strategy: String,
    light_sampler: Option<LightSampler>,
}

impl MLTIntegrator {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        scene: &Scene,
        camera: Arc<Camera>,
        max_depth: i32,
        n_bootstrap: i32,
        n_chains: i32,
        mutations_per_pixel: i32,
        sigma: Float,
        large_step_probability: Float,
        regularize: bool,
        light_sample_strategy: &str,
    ) -> Self {
        Self {
            base: IntegratorBase::from_scene(scene),
            camera,
            max_depth,
            n_bootstrap,
            n_chains,
            mutations_per_pixel,
            sigma,
            large_step_probability,
            regularize,
            light_sample_strategy: light_sample_strategy.to_string(),
            light_sampler: None,
        }
    }

    fn get_film(&self) -> Arc<RwLock<Film>> {
        self.camera.as_ref().get_film()
    }

    /// pbrt-v4 `MLTIntegrator::c(L, lambda) = L.y(lambda)`.
    fn c(l: &SampledSpectrum, lambda: &SampledWavelengths) -> Float {
        l.y(lambda)
    }

    /// pbrt-v4 `MLTIntegrator::L(scratchBuffer, sampler, depth, &pRaster, &lambda)`
    /// (integrators.cpp:2479-2543). Generates a depth-`k` BDPT path
    /// through primary-sample-space stream switching.
    fn l(
        &self,
        light_sampler: &LightSampler,
        scratch_buffer: &mut MemoryArena,
        sampler_wrapper: &mut Sampler,
        depth: i32,
    ) -> (SampledSpectrum, Point2f, SampledWavelengths) {
        if self.base.lights.is_empty() {
            return (
                SampledSpectrum::zero(),
                Point2f::default(),
                SampledWavelengths::sample_visible(0.5),
            );
        }

        sampler_wrapper.start_stream(CAMERA_STREAM_INDEX);

        // Determine s, t for this depth (v4 lines 2486-2495).
        let (s, t, n_strategies) = if depth == 0 {
            (0i32, 2i32, 1i32)
        } else {
            let n = depth + 2;
            let s = ((sampler_wrapper.get_1d() * n as Float) as i32).min(n - 1);
            (s, n - s, n)
        };

        // Wavelengths
        let film = self.get_film();
        let u_lambda = if PbrtOptions::get().disable_wavelength_jitter {
            0.5
        } else {
            sampler_wrapper.get_1d()
        };
        let mut lambda = film.read().unwrap().sample_wavelengths(u_lambda);

        // Camera sample
        let sample_bounds = film.read().unwrap().sample_bounds();
        let bmin = Point2f::new(sample_bounds.min.x as Float, sample_bounds.min.y as Float);
        let bmax = Point2f::new(sample_bounds.max.x as Float, sample_bounds.max.y as Float);
        let pixel_2d = sampler_wrapper.get_pixel_2d();
        let p_raster = Point2f::new(
            bmin.x + (bmax.x - bmin.x) * pixel_2d.x,
            bmin.y + (bmax.y - bmin.y) * pixel_2d.y,
        );
        let mut cs = CameraSample::default();
        cs.p_film = p_raster;
        cs.time = sampler_wrapper.get_1d();
        cs.p_lens = sampler_wrapper.get_2d();
        cs.filter_weight = 1.0;

        let crd = match self.camera.as_ref().generate_ray_differential(&cs, &lambda) {
            Some(c) => c,
            None => return (SampledSpectrum::zero(), p_raster, lambda),
        };
        if crd.weight == 0.0 {
            return (SampledSpectrum::zero(), p_raster, lambda);
        }
        let mpp = self.mutations_per_pixel as Float;
        let ray_diff_scale = Float::max(0.125, 1.0 / Float::sqrt(mpp));
        let mut ray = crd.ray;
        ray.scale_differentials(ray_diff_scale);

        // Camera subpath with exactly t vertices
        let mut camera_vertices: Vec<Vertex> = Vec::with_capacity(t as usize);
        let n_camera = generate_camera_subpath(
            &self.base,
            &self.camera,
            &ray,
            &mut lambda,
            sampler_wrapper,
            scratch_buffer,
            t as usize,
            &mut camera_vertices,
            self.regularize,
        );
        if n_camera != t as usize {
            return (SampledSpectrum::zero(), p_raster, lambda);
        }

        // Light subpath
        sampler_wrapper.start_stream(LIGHT_STREAM_INDEX);
        let mut light_vertices: Vec<Vertex> = Vec::with_capacity(s as usize);
        let time = if !camera_vertices.is_empty() {
            camera_vertices[0].time()
        } else {
            0.0
        };
        let n_light = generate_light_subpath(
            &self.base,
            &self.camera,
            &mut lambda,
            sampler_wrapper,
            scratch_buffer,
            s as usize,
            time,
            light_sampler,
            &mut light_vertices,
            self.regularize,
        );
        if n_light != s as usize {
            return (SampledSpectrum::zero(), p_raster, lambda);
        }

        // Connection
        sampler_wrapper.start_stream(CONNECTION_STREAM_INDEX);
        let (l_path, p_raster_new) = connect_bdpt(
            &self.base,
            &self.camera,
            &mut lambda,
            &mut light_vertices,
            &mut camera_vertices,
            s,
            t,
            light_sampler,
            sampler_wrapper,
        );

        let l_scaled = l_path * (n_strategies as Float);
        let p_final = p_raster_new.unwrap_or(p_raster);
        (l_scaled, p_final, lambda)
    }
}

impl Integrator for MLTIntegrator {
    fn render(&mut self) {
        let film = self.get_film();
        film.read().unwrap().render_start();

        // Build light sampler
        let light_sampler = match LightSampler::create(&self.light_sample_strategy, &self.base) {
            Ok(ls) => ls,
            Err(e) => {
                log::warn!("MLTIntegrator: {:?}", e);
                let f = film.read().unwrap();
                f.render_end();
                f.write_image();
                return;
            }
        };
        self.light_sampler = Some(light_sampler.clone());

        // ----- Bootstrap pass (parallel) -----
        // pbrt-v4 (integrators.cpp:2588-2616) wraps this in
        // `ParallelFor(0, nBootstrap, ...)`. We flatten the
        // (i, depth) double loop to a single `rng_index` ∈
        // [0, n_bootstrap*(maxDepth+1)) so `par_iter_mut` writes each
        // weight at a unique index; per-thread scratch is provided by
        // `for_each_init`.
        let max_depth = self.max_depth;
        let n_bootstrap_samples = self.n_bootstrap * (max_depth + 1);
        let mut bootstrap_weights: Vec<Float> = vec![0.0; n_bootstrap_samples as usize];
        let mpp = self.mutations_per_pixel;
        let sigma = self.sigma;
        let lsp = self.large_step_probability;
        let depth_span = (max_depth + 1) as u64;
        bootstrap_weights.par_iter_mut().enumerate().for_each_init(
            || MemoryArena::new(),
            |scratch, (rng_index, weight_slot)| {
                let depth = (rng_index as u64 % depth_span) as i32;
                let mlt =
                    MLTSampler::new(mpp as u32, rng_index as u64, sigma, lsp, N_SAMPLE_STREAMS);
                let mut sampler_wrapper = Sampler::MLT(mlt);
                let (l_pkt, _p, lambda) =
                    self.l(&light_sampler, scratch, &mut sampler_wrapper, depth);
                *weight_slot = MLTIntegrator::c(&l_pkt, &lambda);
                scratch.reset();
            },
        );

        // Build the bootstrap discrete distribution and `b`.
        let sum_w: Float = bootstrap_weights.iter().sum();
        if sum_w <= 0.0 {
            log::warn!("MLTIntegrator: bootstrap weights all zero -- rendering black image");
            let f = film.read().unwrap();
            f.render_end();
            f.write_image();
            return;
        }
        let bootstrap_distrib = Distribution1D::new(&bootstrap_weights);
        let b = (self.max_depth + 1) as Float / bootstrap_weights.len() as Float * sum_w;

        // ----- Markov chain pass -----
        // pbrt-v4 (integrators.cpp:2643-2644) bases the total mutation count
        // on `Film::SampleBounds().Area()` (the pixel bounds inflated by the
        // filter radius), not the raw pixel bounds. Using sample_bounds keeps
        // variance comparable to v4 even though the expected value is the
        // same.
        let sample_bounds = film.read().unwrap().sample_bounds();
        let n_total_mutations = sample_bounds.area() as i64 * self.mutations_per_pixel as i64;

        // ----- Markov chain pass (parallel over chains, broken into
        //       rounds for live-display updates) -----
        // pbrt-v4 (integrators.cpp:2647-2708) wraps the chain loop in
        // `ParallelFor(0, nChains, ...)`. r4 splits that into
        // `N_ROUNDS` sequential rounds; each round runs
        // `n_chains / N_ROUNDS` chains in parallel, and the round
        // boundary is where we flush `splat_tiles` into `splat_pixels`
        // and ping the registered displays. Since
        // `merge_splat_tiles` clears each tile after copying it to
        // `splat_pixels` with the given scale, calling
        // `merge_splats(b/mpp)` multiple times with the same scale
        // yields exactly the same `splat_pixels` as a single
        // end-of-render call.
        let pixel_bounds = film.read().unwrap().pixel_bounds();
        let splat_scale = b / self.mutations_per_pixel as Float;

        let n_chains = self.n_chains;
        const N_ROUNDS_TARGET: i32 = 10;
        let n_rounds = N_ROUNDS_TARGET.min(n_chains.max(1));

        let max_depth = self.max_depth;
        let mpp = self.mutations_per_pixel;
        let sigma = self.sigma;
        let lsp = self.large_step_probability;
        let bootstrap_distrib = &bootstrap_distrib;
        let film = &film;

        for round in 0..n_rounds {
            let chain_lo = ((round as i64) * n_chains as i64 / n_rounds as i64) as i32;
            let chain_hi = (((round + 1) as i64) * n_chains as i64 / n_rounds as i64) as i32;
            (chain_lo..chain_hi).into_par_iter().for_each_init(
                || MemoryArena::new(),
                |chain_scratch, i| {
                    let n_chain_mutations = ((i + 1) as i64 * n_total_mutations / n_chains as i64)
                        .min(n_total_mutations)
                        - (i as i64 * n_total_mutations / n_chains as i64);

                    // pbrt-v4 (integrators.cpp:2655): `RNG rng(i)` →
                    // `SetSequence(i, MixBits(i))`.
                    let mut rng = RNG::new();
                    rng.set_sequence(i as u64);
                    let u_bootstrap = rng.uniform_float();
                    let (bootstrap_index, _pdf, _) = bootstrap_distrib.sample_discrete(u_bootstrap);
                    let depth = (bootstrap_index as i32) % (max_depth + 1);

                    let mlt = MLTSampler::new(
                        mpp as u32,
                        bootstrap_index as u64,
                        sigma,
                        lsp,
                        N_SAMPLE_STREAMS,
                    );
                    let mut sampler_wrapper = Sampler::MLT(mlt);
                    let (mut l_current, mut p_current, mut lambda_current) =
                        self.l(&light_sampler, chain_scratch, &mut sampler_wrapper, depth);

                    // Hold one read lock for the whole chain; the splat
                    // path uses interior mutability per-tile so
                    // concurrent chains can splat in parallel.
                    let film_read = film.read().unwrap();
                    for _ in 0..n_chain_mutations {
                        sampler_wrapper.start_iteration();
                        let (l_proposed, p_proposed, lambda_proposed) =
                            self.l(&light_sampler, chain_scratch, &mut sampler_wrapper, depth);
                        let c_proposed = MLTIntegrator::c(&l_proposed, &lambda_proposed);
                        let c_current = MLTIntegrator::c(&l_current, &lambda_current);
                        // pbrt-v4 (integrators.cpp:2683):
                        // `accept = std::min<Float>(1, cProposed / cCurrent)`.
                        // `c_current == 0` yields `inf`, `min(1, inf) == 1`.
                        let accept = Float::min(1.0, c_proposed / c_current);

                        // Splat both proposals (pbrt-v4 integrators.cpp:2685-2687).
                        if accept > 0.0 {
                            let scaled = l_proposed * (accept / c_proposed);
                            film_read.add_splat_packet(
                                &Vector2f::new(p_proposed.x, p_proposed.y),
                                &scaled,
                                &lambda_proposed,
                            );
                        }
                        {
                            let scaled = l_current * ((1.0 - accept) / c_current);
                            film_read.add_splat_packet(
                                &Vector2f::new(p_current.x, p_current.y),
                                &scaled,
                                &lambda_current,
                            );
                        }

                        if rng.uniform_float() < accept {
                            p_current = p_proposed;
                            l_current = l_proposed;
                            lambda_current = lambda_proposed;
                            sampler_wrapper.accept();
                        } else {
                            sampler_wrapper.reject();
                        }
                        chain_scratch.reset();
                    }
                    drop(film_read);
                },
            );

            // All chains in this round have completed, so no read lock
            // is held on `film`. Flush in-flight splats and push a
            // preview to the displays.
            let f = film.read().unwrap();
            f.merge_splats(splat_scale);
            f.update_display(&pixel_bounds);
        }

        let f = film.read().unwrap();
        f.merge_splats(splat_scale);
        f.render_end();
        f.write_image();
    }

    fn get_camera(&self) -> Arc<Camera> {
        self.camera.clone()
    }
}

unsafe impl Sync for MLTIntegrator {}

pub fn create_mlt_integrator(
    params: &ParameterDictionary,
    _sampler: &Arc<RwLock<Sampler>>,
    camera: &Arc<Camera>,
    scene: &Scene,
) -> Result<Arc<RwLock<dyn Integrator>>, PbrtError> {
    let max_depth = params.get_one_int("maxdepth", 5);
    let n_bootstrap = params.get_one_int("bootstrapsamples", 100000);
    let n_chains = params.get_one_int("chains", 1000);
    let mutations_per_pixel = params.get_one_int("mutationsperpixel", 100);
    let large_step_probability = params.get_one_float("largestepprobability", 0.3);
    let sigma = params.get_one_float("sigma", 0.01);
    let regularize = params.get_one_bool("regularize", false);
    let light_strategy = {
        let strategy = params.get_one_string("lightsampler", "");
        if strategy.is_empty() {
            params.get_one_string("lightsamplestrategy", "power")
        } else {
            strategy
        }
    };
    Ok(Arc::new(RwLock::new(MLTIntegrator::new(
        scene,
        camera.clone(),
        max_depth,
        n_bootstrap,
        n_chains,
        mutations_per_pixel,
        sigma,
        large_step_probability,
        regularize,
        &light_strategy,
    ))))
}
