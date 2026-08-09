#[cfg(feature = "stats")]
mod _impl {
    use crate::util::base::{Float, Point2i};
    use crate::util::geometry::Bounds2i;
    use crate::util::imageio::write_image;

    use std::cell::RefCell;
    use std::path::Path;
    use std::sync::{Arc, Mutex, OnceLock};
    use std::time::Instant;

    struct PixelStatsState {
        bounds: Bounds2i,
        base_name: String,
        elapsed_ms: Vec<Float>,
        counters: Vec<(String, Vec<Float>)>,
        ratios: Vec<(String, Vec<Float>, Vec<Float>)>,
    }

    static STATE: OnceLock<Mutex<Option<PixelStatsState>>> = OnceLock::new();
    type PixelCallback = Arc<dyn Fn(Point2i, usize) + Send + Sync>;
    static CALLBACKS: OnceLock<Mutex<Vec<PixelCallback>>> = OnceLock::new();

    thread_local! {
        static PIXEL_START: RefCell<Option<(Point2i, Instant)>> = const { RefCell::new(None) };
    }

    fn state() -> &'static Mutex<Option<PixelStatsState>> {
        STATE.get_or_init(|| Mutex::new(None))
    }

    fn callbacks() -> &'static Mutex<Vec<PixelCallback>> {
        CALLBACKS.get_or_init(|| Mutex::new(Vec::new()))
    }

    /// A v4-style per-pixel counter. Values accumulated between
    /// `report_pixel_start` and `report_pixel_end` are written to the registered
    /// counter image when the pixel ends.
    pub struct PixelCounter {
        index: usize,
        name: String,
        value: Mutex<i64>,
    }

    impl PixelCounter {
        pub fn add(&self, value: i64) {
            *self.value.lock().unwrap() += value;
        }

        pub fn inc(&self) {
            self.add(1);
        }
    }

    pub fn register_pixel_counter(name: &str) -> Arc<PixelCounter> {
        let mut registered = callbacks().lock().unwrap();
        let index = registered.len();
        let counter = Arc::new(PixelCounter {
            index,
            name: name.to_string(),
            value: Mutex::new(0),
        });
        let callback_counter = Arc::clone(&counter);
        registered.push(Arc::new(move |pixel, _| {
            let value = {
                let mut value = callback_counter.value.lock().unwrap();
                let result = *value;
                *value = 0;
                result
            };
            report_counter(pixel, callback_counter.index, &callback_counter.name, value);
        }));
        counter
    }

    /// A v4-style per-pixel ratio. Numerator and denominator values accumulated
    /// during a pixel are flushed together when that pixel ends.
    pub struct PixelRatio {
        index: usize,
        name: String,
        values: Mutex<(i64, i64)>,
    }

    impl PixelRatio {
        pub fn add_num(&self, value: i64) {
            self.values.lock().unwrap().0 += value;
        }

        pub fn add_denom(&self, value: i64) {
            self.values.lock().unwrap().1 += value;
        }
    }

    pub fn register_pixel_ratio(name: &str) -> Arc<PixelRatio> {
        let mut registered = callbacks().lock().unwrap();
        let index = registered.len();
        let ratio = Arc::new(PixelRatio {
            index,
            name: name.to_string(),
            values: Mutex::new((0, 0)),
        });
        let callback_ratio = Arc::clone(&ratio);
        registered.push(Arc::new(move |pixel, _| {
            let (numerator, denominator) = {
                let mut values = callback_ratio.values.lock().unwrap();
                let result = *values;
                *values = (0, 0);
                result
            };
            report_ratio(
                pixel,
                callback_ratio.index,
                &callback_ratio.name,
                numerator,
                denominator,
            );
        }));
        ratio
    }

    pub fn enabled() -> bool {
        state().lock().unwrap().is_some()
    }

    pub fn enable(bounds: Bounds2i, film_name: &str) {
        let area = bounds.area().max(0) as usize;
        let base_name = Path::new(film_name)
            .with_extension("")
            .to_str()
            .filter(|name| !name.is_empty())
            .unwrap_or("pbrt")
            .to_string();
        *state().lock().unwrap() = Some(PixelStatsState {
            bounds,
            base_name,
            elapsed_ms: vec![0.0; area],
            counters: Vec::new(),
            ratios: Vec::new(),
        });
    }

    fn pixel_index(stats: &PixelStatsState, pixel: Point2i) -> Option<usize> {
        let x = pixel.x - stats.bounds.min.x;
        let y = pixel.y - stats.bounds.min.y;
        let width = stats.bounds.max.x - stats.bounds.min.x;
        let height = stats.bounds.max.y - stats.bounds.min.y;
        if x < 0 || y < 0 || x >= width || y >= height {
            None
        } else {
            Some((y * width + x) as usize)
        }
    }

    pub fn report_counter(pixel: Point2i, stat_index: usize, name: &str, value: i64) {
        let mut guard = state().lock().unwrap();
        let Some(stats) = guard.as_mut() else { return };
        let Some(index) = pixel_index(stats, pixel) else {
            return;
        };
        let area = stats.elapsed_ms.len();
        while stats.counters.len() <= stat_index {
            stats.counters.push((String::new(), vec![0.0; area]));
        }
        let (stored_name, values) = &mut stats.counters[stat_index];
        if stored_name.is_empty() {
            *stored_name = name.to_string();
        }
        values[index] += value as Float;
    }

    pub fn report_ratio(
        pixel: Point2i,
        stat_index: usize,
        name: &str,
        numerator: i64,
        denominator: i64,
    ) {
        let mut guard = state().lock().unwrap();
        let Some(stats) = guard.as_mut() else { return };
        let Some(index) = pixel_index(stats, pixel) else {
            return;
        };
        let area = stats.elapsed_ms.len();
        while stats.ratios.len() <= stat_index {
            stats
                .ratios
                .push((String::new(), vec![0.0; area], vec![0.0; area]));
        }
        let (stored_name, numerators, denominators) = &mut stats.ratios[stat_index];
        if stored_name.is_empty() {
            *stored_name = name.to_string();
        }
        numerators[index] += numerator as Float;
        denominators[index] += denominator as Float;
    }

    pub fn report_pixel_start(pixel: Point2i) {
        if enabled() {
            PIXEL_START.with(|start| *start.borrow_mut() = Some((pixel, Instant::now())));
        }
    }

    pub fn report_pixel_end(pixel: Point2i) {
        let Some((started_pixel, start)) = PIXEL_START.with(|value| value.borrow_mut().take())
        else {
            return;
        };
        debug_assert_eq!(started_pixel, pixel);
        let elapsed = start.elapsed().as_secs_f64() as Float * 1000.0;
        let callbacks = callbacks().lock().unwrap().clone();
        for (index, callback) in callbacks.iter().enumerate() {
            callback(pixel, index);
        }
        let mut guard = state().lock().unwrap();
        let Some(stats) = guard.as_mut() else {
            return;
        };
        let x = pixel.x - stats.bounds.min.x;
        let y = pixel.y - stats.bounds.min.y;
        if x >= 0
            && y >= 0
            && x < stats.bounds.max.x - stats.bounds.min.x
            && y < stats.bounds.max.y - stats.bounds.min.y
        {
            let width = stats.bounds.max.x - stats.bounds.min.x;
            stats.elapsed_ms[(y * width + x) as usize] += elapsed;
        }
    }

    pub fn write() -> Result<(), String> {
        let guard = state().lock().unwrap();
        let Some(stats) = guard.as_ref() else {
            return Ok(());
        };
        let rgb: Vec<Float> = stats
            .elapsed_ms
            .iter()
            .flat_map(|value| [*value, *value, *value])
            .collect();
        let output = format!("{}-time.exr", stats.base_name);
        let resolution = stats.bounds.diagonal();
        let output_bounds = Bounds2i::from(((0, 0), (resolution.x, resolution.y)));
        write_image(&output, &rgb, &output_bounds, &resolution)
            .map_err(|error| error.to_string())?;

        for (name, values) in &stats.counters {
            let image: Vec<Float> = values
                .iter()
                .flat_map(|value| [*value, *value, *value])
                .collect();
            let output = format!("{}-{}.exr", stats.base_name, name);
            write_image(&output, &image, &output_bounds, &resolution)
                .map_err(|error| error.to_string())?;
        }
        for (name, numerators, denominators) in &stats.ratios {
            let image: Vec<Float> = numerators
                .iter()
                .zip(denominators)
                .flat_map(|(numerator, denominator)| {
                    let ratio = if *denominator == 0.0 {
                        0.0
                    } else {
                        numerator / denominator
                    };
                    [*numerator, *denominator, ratio]
                })
                .collect();
            let output = format!("{}-{}.exr", stats.base_name, name);
            write_image(&output, &image, &output_bounds, &resolution)
                .map_err(|error| error.to_string())?;
        }
        Ok(())
    }
}

#[cfg(not(feature = "stats"))]
mod _impl {
    use crate::util::base::Point2i;
    use crate::util::geometry::Bounds2i;

    use std::sync::Arc;

    pub struct PixelCounter;

    impl PixelCounter {
        pub fn add(&self, _value: i64) {}
        pub fn inc(&self) {}
    }

    pub struct PixelRatio;

    impl PixelRatio {
        pub fn add_num(&self, _value: i64) {}
        pub fn add_denom(&self, _value: i64) {}
    }

    pub fn register_pixel_counter(_name: &str) -> Arc<PixelCounter> {
        Arc::new(PixelCounter)
    }

    pub fn register_pixel_ratio(_name: &str) -> Arc<PixelRatio> {
        Arc::new(PixelRatio)
    }

    pub fn enabled() -> bool {
        false
    }

    pub fn enable(_bounds: Bounds2i, _film_name: &str) {}
    pub fn report_counter(_pixel: Point2i, _stat_index: usize, _name: &str, _value: i64) {}
    pub fn report_ratio(
        _pixel: Point2i,
        _stat_index: usize,
        _name: &str,
        _numerator: i64,
        _denominator: i64,
    ) {
    }
    pub fn report_pixel_start(_pixel: Point2i) {}
    pub fn report_pixel_end(_pixel: Point2i) {}
    pub fn write() -> Result<(), String> {
        Ok(())
    }
}

pub use _impl::*;
