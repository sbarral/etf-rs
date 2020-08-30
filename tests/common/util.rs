use etf::num::Float;
use std::fmt::Display;

use rand::RngCore;
use rand_pcg;


pub fn test_rng() -> impl RngCore {
    rand_pcg::Lcg128Xsl64::new(0xcafef00dd15ea5e5, 0xa02bdbf7bb3c0a7ac28fa16a64abf96)
}

pub trait TestFloat: Float + Display {
    fn as_f64(self) -> f64;
    fn as_u64(self) -> u64;
    fn as_usize(self) -> usize;
    fn cast_f64(u: f64) -> Self;
    fn cast_u64(u: u64) -> Self;
}

impl TestFloat for f32 {
    fn as_f64(self) -> f64 {
        self as f64
    }
    fn as_u64(self) -> u64 {
        self as u64
    }
    fn as_usize(self) -> usize {
        self as usize
    }
    fn cast_f64(u: f64) -> Self {
        u as f32
    }
    fn cast_u64(u: u64) -> Self {
        u as f32
    }
}
impl TestFloat for f64 {
    fn as_u64(self) -> u64 {
        self as u64
    }
    fn as_usize(self) -> usize {
        self as usize
    }
    fn as_f64(self) -> f64 {
        self
    }
    fn cast_f64(u: f64) -> Self {
        u
    }
    fn cast_u64(u: u64) -> Self {
        u as f64
    }
}