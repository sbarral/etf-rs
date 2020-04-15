use etf::Float;
use libm;
use std::fmt::Display;

use rand::RngCore;
use rand_pcg;

mod collisions;
mod goodness_of_fit;
mod test_distributions;

pub use collisions::*;
pub use goodness_of_fit::*;
pub use test_distributions::*;

pub trait TestFloat: Float + Display {
    fn erf(self) -> Self;
    fn sqrt(self) -> Self;
    fn cast_f64(u: f64) -> Self;
    fn as_f64(&self) -> f64;
    fn cast_u64(u: u64) -> Self;
    fn as_u64(&self) -> u64;
    fn as_usize(&self) -> usize;
}
impl TestFloat for f32 {
    fn erf(self) -> Self {
        libm::erff(self)
    }
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    fn cast_f64(u: f64) -> Self {
        u as f32
    }
    fn as_f64(&self) -> f64 {
        *self as f64
    }
    fn cast_u64(u: u64) -> Self {
        u as f32
    }
    fn as_u64(&self) -> u64 {
        *self as u64
    }
    fn as_usize(&self) -> usize {
        *self as usize
    }
}
impl TestFloat for f64 {
    fn erf(self) -> Self {
        libm::erf(self)
    }
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    fn cast_f64(u: f64) -> Self {
        u
    }
    fn as_f64(&self) -> f64 {
        *self
    }
    fn cast_u64(u: u64) -> Self {
        u as f64
    }
    fn as_u64(&self) -> u64 {
        *self as u64
    }
    fn as_usize(&self) -> usize {
        *self as usize
    }
}
fn new_test_rng() -> impl RngCore {
    rand_pcg::Lcg128Xsl64::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7ac28fa16a64abf96)
}