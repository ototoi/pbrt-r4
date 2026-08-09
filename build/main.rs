mod mipmap;
mod sensor;
mod spectrum;
mod utils;

fn main() {
    spectrum::build();
    sensor::build();
    mipmap::build();
}
