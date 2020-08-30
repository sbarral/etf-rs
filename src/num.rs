use rand::Rng;
use std;
use std::cmp::PartialOrd;
use std::fmt::{Debug, Display};
use std::ops::{
    Add, AddAssign, BitAnd, BitOr, Div, DivAssign, Mul, MulAssign, Neg, Shl, Shr, Sub, SubAssign,
};

/// An unsigned integer type.
pub trait UInt:
    private::Sealed
    + Copy
    + Clone
    + Debug
    + Display
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Shl<u32, Output = Self>
    + Shr<u32, Output = Self>
    + BitAnd<Output = Self>
    + BitOr<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + PartialOrd
{
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

impl UInt for u32 {
    #[doc(hidden)]
    const BITS: u32 = 32;
    #[doc(hidden)]
    const ZERO: Self = 0u32;
    #[doc(hidden)]
    const ONE: Self = 1u32;

    #[doc(hidden)]
    fn gen<R: Rng + ?Sized>(rng: &mut R) -> Self {
        rng.next_u32()
    }
    #[doc(hidden)]
    fn as_usize(self) -> usize {
        self as usize
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
    fn gen<R: Rng + ?Sized>(rng: &mut R) -> Self {
        rng.next_u64()
    }
    #[doc(hidden)]
    fn as_usize(self) -> usize {
        self as usize
    }
}

/// A floating point type.
pub trait Float:
    private::Sealed
    + Copy
    + Clone
    + Debug
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
    + PartialOrd
    + From<f32>
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
    const PI: Self;

    #[doc(hidden)]
    type UInt: UInt; // Unsigned integer used for float generation

    #[doc(hidden)]
    fn as_uint(self) -> Self::UInt;
    #[doc(hidden)]
    fn round_as_uint(self) -> Self::UInt;
    #[doc(hidden)]
    fn cast_usize(u: usize) -> Self;
    #[doc(hidden)]
    fn cast_uint(u: Self::UInt) -> Self;
    #[doc(hidden)]
    fn from_uint_bits(u: Self::UInt) -> Self;
    #[doc(hidden)]
    fn to_uint_bits(self) -> Self::UInt;
    #[doc(hidden)]
    fn min(self, other: Self) -> Self;
    #[doc(hidden)]
    fn max(self, other: Self) -> Self;
    #[doc(hidden)]
    fn abs(self) -> Self;
    #[doc(hidden)]
    fn sqrt(self) -> Self;
    #[doc(hidden)]
    fn ln(self) -> Self;
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
    fn copysign(self, sign: Self) -> Self;
    #[doc(hidden)]
    fn gen<R: Rng + ?Sized>(rng: &mut R) -> Self;
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
    const PI: Self = 3.14159265358979323846;

    #[doc(hidden)]
    type UInt = u32;

    #[doc(hidden)]
    fn as_uint(self) -> Self::UInt {
        self as Self::UInt
    }
    #[doc(hidden)]
    fn round_as_uint(self) -> Self::UInt {
        self.round() as Self::UInt
    }
    #[doc(hidden)]
    fn cast_usize(u: usize) -> Self {
        u as Self
    }
    #[doc(hidden)]
    fn cast_uint(u: Self::UInt) -> Self {
        u as Self
    }
    #[doc(hidden)]
    fn from_uint_bits(u: Self::UInt) -> Self {
        Self::from_bits(u)
    }
    #[doc(hidden)]
    fn to_uint_bits(self) -> Self::UInt {
        self.to_bits()
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
    fn sqrt(self) -> Self {
        self.sqrt()
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
    fn erf(self) -> Self {
        unsafe { cmath::erff(self) }
    }
    #[doc(hidden)]
    fn erfc(self) -> Self {
        unsafe { cmath::erfcf(self) }
    }
    #[doc(hidden)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        self.mul_add(a, b)
    }
    #[doc(hidden)]
    fn copysign(self, sign: Self) -> Self {
        self.copysign(sign)
    }
    #[doc(hidden)]
    fn gen<R: Rng + ?Sized>(rng: &mut R) -> Self {
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
    const PI: Self = 3.14159265358979323846;

    #[doc(hidden)]
    type UInt = u64;

    #[doc(hidden)]
    fn as_uint(self) -> Self::UInt {
        self as Self::UInt
    }
    #[doc(hidden)]
    fn round_as_uint(self) -> Self::UInt {
        self.round() as Self::UInt
    }
    #[doc(hidden)]
    fn cast_usize(u: usize) -> Self {
        u as Self
    }
    #[doc(hidden)]
    fn cast_uint(u: Self::UInt) -> Self {
        u as Self
    }
    #[doc(hidden)]
    fn from_uint_bits(u: Self::UInt) -> Self {
        Self::from_bits(u)
    }
    #[doc(hidden)]
    fn to_uint_bits(self) -> Self::UInt {
        self.to_bits()
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
    fn sqrt(self) -> Self {
        self.sqrt()
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
    fn erf(self) -> Self {
        unsafe { cmath::erf(self) }
    }
    #[doc(hidden)]
    fn erfc(self) -> Self {
        unsafe { cmath::erfc(self) }
    }
    #[doc(hidden)]
    fn mul_add(self, a: Self, b: Self) -> Self {
        self.mul_add(a, b)
    }
    #[doc(hidden)]
    fn copysign(self, sign: Self) -> Self {
        self.copysign(sign)
    }
    #[doc(hidden)]
    fn gen<R: Rng + ?Sized>(rng: &mut R) -> Self {
        rng.gen()
    }
}

/// A univariate function.
///
/// This trait is currently needed to work around the impossibility to
/// implement the `Fn(T) -> T` trait for custom types. It may be deprecated
/// once [Fn traits](https://github.com/rust-lang/rust/issues/29625) are
/// stabilized.
pub trait Func<T> {
    fn eval(&self, x: T) -> T;
}

impl<T: Float, F: Fn(T) -> T> Func<T> for F {
    fn eval(&self, x: T) -> T {
        (*self)(x)
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
    /// System-provided special functions.
    #[link(name = "m")]
    extern "C" {
        pub fn erff(x: f32) -> f32;
        pub fn erfcf(x: f32) -> f32;
        pub fn erf(x: f64) -> f64;
        pub fn erfc(x: f64) -> f64;
    }
}
