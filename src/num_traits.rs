use rand::Rng;
use std::ops::{Add, Sub, Div, Mul, Shl, Shr, BitAnd};
use std::cmp::PartialOrd;

/// Unsigned integer traits.
pub trait Int:
    Copy
    + Clone
    + Sub<Output = Self>
    + Shl<u32, Output = Self>
    + Shr<u32, Output = Self>
    + BitAnd<Output = Self>
    + PartialOrd {
    #[doc(hidden)]
    const BITS: u32;
    #[doc(hidden)]
    const ZERO: Self;
    #[doc(hidden)]
    const ONE: Self;

    #[doc(hidden)]
    fn gen<R: Rng + ?Sized>(rng: &mut R) -> Self;

    #[doc(hidden)]
    fn as_usize(self) -> usize;
}

impl Int for u32 {
    const BITS: u32 = 32;
    const ZERO: Self = 0u32;
    const ONE: Self = 1u32;

    fn gen<R: Rng + ?Sized>(rng: &mut R) -> Self {
        rng.gen()
    }

    fn as_usize(self) -> usize {
        self as usize
    }
}

impl Int for u64 {
    const BITS: u32 = 64;
    const ZERO: Self = 0u64;
    const ONE: Self = 1u64;

    fn gen<R: Rng + ?Sized>(rng: &mut R) -> Self {
        rng.gen()
    }

    fn as_usize(self) -> usize {
        self as usize
    }
}

/// Floating point trait.
pub trait Float:
    Copy
    + Clone
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + PartialOrd
{
    #[doc(hidden)]
    const SIGNIFICAND_BITS: u32;
    #[doc(hidden)]
    const ZERO: Self;
    #[doc(hidden)]
    const ONE: Self;

    #[doc(hidden)]
    type GenInt: Int; // Unsigned integer used for float generation

    #[doc(hidden)]
    fn gen<R: Rng + ?Sized>(rng: &mut R) -> Self;

    #[doc(hidden)]
    fn cast_gen_int(u: Self::GenInt) -> Self;

    #[doc(hidden)]
    fn as_gen_int(self) -> Self::GenInt;
}

impl Float for f32 {
    const SIGNIFICAND_BITS: u32 = 23;
    const ZERO: Self = 0f32;
    const ONE: Self = 1f32;

    type GenInt = u32;

    fn gen<R: Rng + ?Sized>(rng: &mut R) -> Self {
        rng.gen()
    }
    fn cast_gen_int(u: Self::GenInt) -> Self {
        u as Self
    }
    fn as_gen_int(self) -> Self::GenInt {
        self as Self::GenInt
    }
}

impl Float for f64 {
    const SIGNIFICAND_BITS: u32 = 52;
    const ZERO: Self = 0f64;
    const ONE: Self = 1f64;

    type GenInt = u64;

    fn gen<R: Rng + ?Sized>(rng: &mut R) -> Self {
        rng.gen()
    }
    fn cast_gen_int(u: Self::GenInt) -> Self {
        u as Self
    }
    fn as_gen_int(self) -> Self::GenInt {
        self as Self::GenInt
    }
}
