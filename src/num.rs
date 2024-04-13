//! Numeric types.

use rand_core::RngCore;
use std::fmt::{Debug, Display};
use std::ops::{
    Add, AddAssign, BitAnd, BitOr, BitXor, Div, DivAssign, Mul, MulAssign, Neg, Shl, Shr, Sub,
    SubAssign,
};

/// An unsigned integer type.
pub trait UInt:
    private::Sealed
    + Copy
    + Clone
    + Default
    + Debug
    + Display
    + Ord
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Shl<u32, Output = Self>
    + Shr<u32, Output = Self>
    + BitAnd<Output = Self>
    + BitOr<Output = Self>
    + BitXor<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
{
    #[doc(hidden)]
    const BITS: u32;
    #[doc(hidden)]
    const ZERO: Self;
    #[doc(hidden)]
    const ONE: Self;
    #[doc(hidden)]
    const MAX: Self;

    #[doc(hidden)]
    fn gen<R: RngCore + ?Sized>(rng: &mut R) -> Self;

    #[doc(hidden)]
    fn as_usize(self) -> usize;

    #[doc(hidden)]
    fn arithmetic_right_shift(self, shift: u32) -> Self;
}

impl UInt for u32 {
    #[doc(hidden)]
    const BITS: u32 = 32;
    #[doc(hidden)]
    const ZERO: Self = 0u32;
    #[doc(hidden)]
    const ONE: Self = 1u32;
    #[doc(hidden)]
    const MAX: Self = u32::MAX;

    #[doc(hidden)]
    #[inline]
    fn gen<R: RngCore + ?Sized>(rng: &mut R) -> Self {
        rng.next_u32()
    }
    #[doc(hidden)]
    #[inline]
    fn as_usize(self) -> usize {
        self as usize
    }
    #[doc(hidden)]
    #[inline]
    fn arithmetic_right_shift(self, shift: u32) -> Self {
        ((self as i32) >> shift) as u32
    }
}

impl UInt for u64 {
    #[doc(hidden)]
    const BITS: u32 = 64;
    #[doc(hidden)]
    const ZERO: Self = 0u64;
    #[doc(hidden)]
    const ONE: Self = 1u64;
    #[doc(hidden)]
    const MAX: Self = u64::MAX;

    #[doc(hidden)]
    #[inline]
    fn gen<R: RngCore + ?Sized>(rng: &mut R) -> Self {
        rng.next_u64()
    }
    #[doc(hidden)]
    #[inline]
    fn as_usize(self) -> usize {
        self as usize
    }
    #[doc(hidden)]
    #[inline]
    fn arithmetic_right_shift(self, shift: u32) -> Self {
        ((self as i64) >> shift) as u64
    }
}

/// A floating point type.
pub trait Float:
    private::Sealed
    + Copy
    + Clone
    + Default
    + Debug
    + Display
    + PartialOrd
    + From<f32>
    + Into<f64>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
{
    #[doc(hidden)]
    const SIGNIFICAND_BITS: u32;
    #[doc(hidden)]
    const ZERO: Self;
    #[doc(hidden)]
    const ONE: Self;
    #[doc(hidden)]
    const TWO: Self;
    #[doc(hidden)]
    const ONE_HALF: Self;
    #[doc(hidden)]
    const INFINITY: Self;
    #[doc(hidden)]
    const PI: Self;

    #[doc(hidden)]
    type UInt: UInt; // Unsigned integer used for float generation

    #[doc(hidden)]
    fn as_uint(self) -> Self::UInt;
    #[doc(hidden)]
    fn round_as_uint(self) -> Self::UInt;
    #[doc(hidden)]
    fn cast_u32(u: u32) -> Self;
    #[doc(hidden)]
    fn cast_usize(u: usize) -> Self;
    #[doc(hidden)]
    fn cast_uint(u: Self::UInt) -> Self;
    #[doc(hidden)]
    fn from_bits(u: Self::UInt) -> Self;
    #[doc(hidden)]
    fn to_bits(self) -> Self::UInt;
    #[doc(hidden)]
    #[inline]
    fn bitxor(self, u: Self::UInt) -> Self {
        Self::from_bits(self.to_bits() ^ u)
    }
    #[doc(hidden)]
    fn min(self, other: Self) -> Self;
    #[doc(hidden)]
    fn max(self, other: Self) -> Self;
    #[doc(hidden)]
    fn abs(self) -> Self;
    #[doc(hidden)]
    fn sqrt(self) -> Self;
    #[doc(hidden)]
    fn tan(self) -> Self;
    #[doc(hidden)]
    fn atan(self) -> Self;
    #[doc(hidden)]
    fn ln(self) -> Self;
    #[doc(hidden)]
    fn log2(self) -> Self;
    #[doc(hidden)]
    fn exp(self) -> Self;
    #[doc(hidden)]
    fn powf(self, exponent: Self) -> Self;
    #[doc(hidden)]
    fn erf(self) -> Self;
    #[doc(hidden)]
    fn erfc(self) -> Self;
    #[doc(hidden)]
    fn mul_add(self, a: Self, b: Self) -> Self;
    #[doc(hidden)]
    fn is_nan(self) -> bool;
    #[doc(hidden)]
    #[inline]
    fn gen<R: RngCore + ?Sized>(rng: &mut R) -> Self {
        let scale = Self::ONE / Self::cast_uint(Self::UInt::ONE << (Self::SIGNIFICAND_BITS + 1));
        let r = Self::UInt::gen(rng) >> (Self::UInt::BITS - Self::SIGNIFICAND_BITS - 1);

        scale * Self::cast_uint(r)
    }
}

impl Float for f32 {
    #[doc(hidden)]
    const SIGNIFICAND_BITS: u32 = 23;
    #[doc(hidden)]
    const ZERO: Self = 0f32;
    #[doc(hidden)]
    const ONE: Self = 1f32;
    #[doc(hidden)]
    const TWO: Self = 2f32;
    #[doc(hidden)]
    const ONE_HALF: Self = 0.5f32;
    #[doc(hidden)]
    const INFINITY: Self = std::f32::INFINITY;
    #[doc(hidden)]
    const PI: Self = std::f32::consts::PI;

    #[doc(hidden)]
    type UInt = u32;

    #[doc(hidden)]
    #[inline]
    fn as_uint(self) -> Self::UInt {
        self as Self::UInt
    }
    #[doc(hidden)]
    #[inline]
    fn round_as_uint(self) -> Self::UInt {
        self.round() as Self::UInt
    }
    #[doc(hidden)]
    #[inline]
    fn cast_u32(u: u32) -> Self {
        u as Self
    }
    #[doc(hidden)]
    #[inline]
    fn cast_usize(u: usize) -> Self {
        u as Self
    }
    #[doc(hidden)]
    #[inline]
    fn cast_uint(u: Self::UInt) -> Self {
        u as Self
    }
    #[doc(hidden)]
    #[inline]
    fn from_bits(u: Self::UInt) -> Self {
        Self::from_bits(u)
    }
    #[doc(hidden)]
    #[inline]
    fn to_bits(self) -> Self::UInt {
        self.to_bits()
    }
    #[doc(hidden)]
    #[inline]
    fn min(self, other: Self) -> Self {
        self.min(other)
    }
    #[doc(hidden)]
    #[inline]
    fn max(self, other: Self) -> Self {
        self.max(other)
    }
    #[doc(hidden)]
    #[inline]
    fn abs(self) -> Self {
        self.abs()
    }
    #[doc(hidden)]
    #[inline]
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    #[doc(hidden)]
    #[inline]
    fn tan(self) -> Self {
        self.tan()
    }
    #[doc(hidden)]
    #[inline]
    fn atan(self) -> Self {
        self.atan()
    }
    #[doc(hidden)]
    #[inline]
    fn ln(self) -> Self {
        self.ln()
    }
    #[doc(hidden)]
    #[inline]
    fn log2(self) -> Self {
        self.log2()
    }
    #[doc(hidden)]
    #[inline]
    fn exp(self) -> Self {
        self.exp()
    }
    #[doc(hidden)]
    #[inline]
    fn powf(self, exponent: Self) -> Self {
        self.powf(exponent)
    }
    #[doc(hidden)]
    #[inline]
    fn erf(self) -> Self {
        unsafe { cmath::erff(self) }
    }
    #[doc(hidden)]
    #[inline]
    fn erfc(self) -> Self {
        unsafe { cmath::erfcf(self) }
    }
    #[doc(hidden)]
    #[inline]
    fn is_nan(self) -> bool {
        self.is_nan()
    }
    #[doc(hidden)]
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        self.mul_add(a, b)
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
    const TWO: Self = 2f64;
    #[doc(hidden)]
    const ONE_HALF: Self = 0.5f64;
    #[doc(hidden)]
    const INFINITY: Self = std::f64::INFINITY;
    #[doc(hidden)]
    const PI: Self = std::f64::consts::PI;

    #[doc(hidden)]
    type UInt = u64;

    #[doc(hidden)]
    #[inline]
    fn as_uint(self) -> Self::UInt {
        self as Self::UInt
    }
    #[doc(hidden)]
    #[inline]
    fn round_as_uint(self) -> Self::UInt {
        self.round() as Self::UInt
    }
    #[doc(hidden)]
    #[inline]
    fn cast_u32(u: u32) -> Self {
        u as Self
    }
    #[doc(hidden)]
    #[inline]
    fn cast_usize(u: usize) -> Self {
        u as Self
    }
    #[doc(hidden)]
    #[inline]
    fn cast_uint(u: Self::UInt) -> Self {
        u as Self
    }
    #[doc(hidden)]
    #[inline]
    fn from_bits(u: Self::UInt) -> Self {
        Self::from_bits(u)
    }
    #[doc(hidden)]
    #[inline]
    fn to_bits(self) -> Self::UInt {
        self.to_bits()
    }
    #[doc(hidden)]
    #[inline]
    fn min(self, other: Self) -> Self {
        self.min(other)
    }
    #[doc(hidden)]
    #[inline]
    fn max(self, other: Self) -> Self {
        self.max(other)
    }
    #[doc(hidden)]
    #[inline]
    fn abs(self) -> Self {
        self.abs()
    }
    #[doc(hidden)]
    #[inline]
    fn sqrt(self) -> Self {
        self.sqrt()
    }
    #[doc(hidden)]
    #[inline]
    fn tan(self) -> Self {
        self.tan()
    }
    #[doc(hidden)]
    #[inline]
    fn atan(self) -> Self {
        self.atan()
    }
    #[doc(hidden)]
    #[inline]
    fn ln(self) -> Self {
        self.ln()
    }
    #[doc(hidden)]
    #[inline]
    fn log2(self) -> Self {
        self.log2()
    }
    #[doc(hidden)]
    #[inline]
    fn exp(self) -> Self {
        self.exp()
    }
    #[doc(hidden)]
    #[inline]
    fn powf(self, exponent: Self) -> Self {
        self.powf(exponent)
    }
    #[doc(hidden)]
    #[inline]
    fn erf(self) -> Self {
        unsafe { cmath::erf(self) }
    }
    #[doc(hidden)]
    #[inline]
    fn erfc(self) -> Self {
        unsafe { cmath::erfc(self) }
    }
    #[doc(hidden)]
    #[inline]
    fn is_nan(self) -> bool {
        self.is_nan()
    }
    #[doc(hidden)]
    #[inline]
    fn mul_add(self, a: Self, b: Self) -> Self {
        self.mul_add(a, b)
    }
}

/// Prevent implementation of public traits to leave open the possibility to
/// extend these traits in the future.
mod private {
    pub trait Sealed {}

    impl Sealed for u32 {}
    impl Sealed for u64 {}
    impl Sealed for f32 {}
    impl Sealed for f64 {}
}

mod cmath {
    // System-provided special functions.
    #[link(name = "m")]
    extern "C" {
        pub fn erff(x: f32) -> f32;
        pub fn erfcf(x: f32) -> f32;
        pub fn erf(x: f64) -> f64;
        pub fn erfc(x: f64) -> f64;
    }
}
