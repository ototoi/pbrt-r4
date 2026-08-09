use indicatif::*;

pub struct ProgressReporter {
    pb: ProgressBar,
}

impl ProgressReporter {
    pub fn new(total_work: usize, title: &str) -> Self {
        let pb = ProgressBar::new(total_work as u64);
        pb.set_draw_target(ProgressDrawTarget::stdout());
        //    format!("{}: ", title) + "[{wide_bar}]  ({elapsed_precise}|{eta_precise}) ";
        let template = format!("{{spinner:.bold.green}} {}: ", title)
            + "[{wide_bar:.cyan}]  ({elapsed_precise}|{eta_precise}) ";
        pb.set_style(
            ProgressStyle::with_template(&template)
                .unwrap()
                //.progress_chars("++ "),
                .progress_chars("█▇▆▅▄▃▂▁  "), //.progress_chars("=> ")
        );
        pb.tick();
        ProgressReporter { pb }
    }
    pub fn update(&mut self, num: usize) {
        if num != 0 {
            self.pb.inc(num as u64);
        }
    }
    pub fn done(&mut self) {
        self.pb.finish();
    }
}
