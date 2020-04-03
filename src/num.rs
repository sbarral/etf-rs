use rand::{Rng, RngCore};
use std;
use std::cmp::PartialOrd;
use std::fmt::Debug;
use std::ops::{Add, AddAssign, BitAnd, Div, Mul, Neg, Shl, Shr, Sub};

/// Unsigned integer traits.
pub trait Int:
    Copy
    + Clone
    + Debug
    + Sub<Output = Self>
    + Shl<u32, Output = Self>
    + Shr<u32, Output = Self>
    + BitAnd<Output = Self>
    + PartialOrd
    + Default
{
    #[doc(hidden)]
    const BITS: u32;
    #[doc(hidden)]
    const ZERO: Self;
    #[doc(hidden)]
    const ONE: Self;

    #[doc(hidden)]
    fn gen<R: RngCore + ?Sized>(rng: &mut R) -> Self;

    #[doc(hidden)]
    fn as_usize(self) -> usize;
}

impl Int for u32 {
    #[doc(hidden)]
    const BITS: u32 = 32;
    #[doc(hidden)]
    const ZERO: Self = 0u32;
    #[doc(hidden)]
    const ONE: Self = 1u32;

    #[doc(hidden)]
    fn gen<R: RngCore + ?Sized>(rng: &mut R) -> Self {
        rng.next_u32()
    }
    #[doc(hidden)]
    fn as_usize(self) -> usize {
        self as usize
    }
}

impl Int for u64 {
    #[doc(hidden)]
    const BITS: u32 = 64;
    #[doc(hidden)]
    const ZERO: Self = 0u64;
    #[doc(hidden)]
    const ONE: Self = 1u64;

    #[doc(hidden)]
    fn gen<R: RngCore + ?Sized>(rng: &mut R) -> Self {
        rng.next_u64()
    }
    #[doc(hidden)]
    fn as_usize(self) -> usize {
        self as usize
    }
}

/// Floating point trait.
pub trait Float:
    Copy
    + Clone
    + Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + PartialOrd
    + Default
{
    #[doc(hidden)]
    const SIGNIFICAND_BITS: u32;
    #[doc(hidden)]
    const ZERO: Self;
    #[doc(hidden)]
    const ONE: Self;
    #[doc(hidden)]
    const INFINITY: Self;

    #[doc(hidden)]
    type GenInt: Int; // Unsigned integer used for float generation

    #[doc(hidden)]
    fn cast_usize(u: usize) -> Self;
    #[doc(hidden)]
    fn cast_gen_int(u: Self::GenInt) -> Self;
    #[doc(hidden)]
    fn round_as_gen_int(self) -> Self::GenInt;
    #[doc(hidden)]
    fn min(self, other: Self) -> Self;
    #[doc(hidden)]
    fn max(self, other: Self) -> Self;
    #[doc(hidden)]
    fn abs(self) -> Self;
    #[doc(hidden)]
    fn ln(self) -> Self;
    #[doc(hidden)]
    fn exp(self) -> Self;
    #[doc(hidden)]
    fn powf(self, exponent: Self) -> Self;
    #[doc(hidden)]
    fn gen<R: RngCore + ?Sized>(rng: &mut R) -> Self;
}

impl Float for f32 {
    #[doc(hidden)]
    const SIGNIFICAND_BITS: u32 = 23;
    #[doc(hidden)]
    const ZERO: Self = 0f32;
    #[doc(hidden)]
    const ONE: Self = 1f32;
    #[doc(hidden)]
    const INFINITY: Self = std::f32::INFINITY;

    #[doc(hidden)]
    type GenInt = u32;

    #[doc(hidden)]
    fn cast_usize(u: usize) -> Self {
        u as Self
    }
    #[doc(hidden)]
    fn cast_gen_int(u: Self::GenInt) -> Self {
        u as Self
    }
    #[doc(hidden)]
    fn round_as_gen_int(self) -> Self::GenInt {
        self.round() as Self::GenInt
    }
    #[doc(hidden)]
    fn min(self, other: Self) -> Self {
        self.min(other)
    }
    #[doc(hidden)]
    fn max(self, other: Self) -> Self {
        self.max(other)
    }
    #[doc(hidden)]
    fn abs(self) -> Self {
        self.abs()
    }
    #[doc(hidden)]
    fn ln(self) -> Self {
        self.ln()
    }
    #[doc(hidden)]
    fn exp(self) -> Self {
        self.exp()
    }
    #[doc(hidden)]
    fn powf(self, exponent: Self) -> Self {
        self.powf(exponent)
    }
    #[doc(hidden)]
    fn gen<R: RngCore + ?Sized>(rng: &mut R) -> Self {
        rng.gen()
    }
}

impl Float for f64 {
    #[doc(hidden)]
    const SIGNIFICAND_BITS: u32 = 52;
    #[doc(hidden)]
    const ZERO: Self = 0f64;
    #[doc(hidden)]
    const ONE: Self = 1f64;
    #[doc(hidden)]
    const INFINITY: Self = std::f64::INFINITY;

    #[doc(hidden)]
    type GenInt = u64;

    #[doc(hidden)]
    fn cast_usize(u: usize) -> Self {
        u as Self
    }
    #[doc(hidden)]
    fn cast_gen_int(u: Self::GenInt) -> Self {
        u as Self
    }
    #[doc(hidden)]
    fn round_as_gen_int(self) -> Self::GenInt {
        self.round() as Self::GenInt
    }
    #[doc(hidden)]
    fn min(self, other: Self) -> Self {
        self.min(other)
    }
    #[doc(hidden)]
    fn max(self, other: Self) -> Self {
        self.max(other)
    }
    #[doc(hidden)]
    fn abs(self) -> Self {
        self.abs()
    }
    #[doc(hidden)]
    fn ln(self) -> Self {
        self.ln()
    }
    #[doc(hidden)]
    fn exp(self) -> Self {
        self.exp()
    }
    #[doc(hidden)]
    fn powf(self, exponent: Self) -> Self {
        self.powf(exponent)
    }
    #[doc(hidden)]
    fn gen<R: RngCore + ?Sized>(rng: &mut R) -> Self {
        rng.gen()
    }
}
