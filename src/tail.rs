use crate::num::Float;

use rand_core::RngCore;

/// Distribution tail trait.
pub trait Tail<T: Float> {
    fn switch(&self) -> Option<T::GenInt>;
    fn sample<R: RngCore>(&self, rng: &mut R) -> Option<T>;
}

/// Empty tail properties.
#[derive(Copy, Clone)]
pub struct TailNone {}

impl<T: Float> Tail<T> for TailNone {
    fn switch(&self) -> Option<T::GenInt> {
        return None;
    }
    fn sample<R: RngCore>(&self, _rng: &mut R) -> Option<T> {
        return None;
    }
}

/// Distribution tail properties.
#[derive(Copy, Clone)]
pub struct TailSome<T: Float, G: Fn(&mut RngCore) -> Option<T>> {
    pub func: G,
    pub switch: T::GenInt,
}

impl<T: Float, G: Fn(&mut RngCore) -> Option<T>> Tail<T> for TailSome<T, G> {
    fn switch(&self) -> Option<T::GenInt> {
        return Some(self.switch);
    }
    fn sample<R: RngCore>(&self, rng: &mut R) -> Option<T> {
        return (self.func)(rng);
    }
}
