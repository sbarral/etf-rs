//! Primitive ETF distributions and related utilities.

// Internal traits.
use crate::num::{Float, UInt};
use partition::*;
pub use table_validation::*;

// External traits.
use rand_core::RngCore;
use std::marker::PhantomData;

// Modules.
pub mod partition;
mod table_validation;
pub mod util;

/// Univariate function.
///
/// This trait is automatically implemented for `Fn(T) -> T` types. It is mostly
/// useful as a way to work around the current impossibility to implement the
/// `Fn(T) -> T` trait for custom functor types (see
/// [this issue](https://github.com/rust-lang/rust/issues/29625)),
/// but also provides an optimization opportunity for the implementations of the
/// wedge acceptance-rejection test.
///
pub trait UnivariateFn<T: Float> {
    /// Evaluates the function at `x`.
    fn eval(&self, x: T) -> T;

    /// Tests the inequality `a * f(x) > b` where `a` and `b` are strictly
    /// positive.
    ///
    /// This function has a trivial default implementation, which can be
    /// overridden with an optimized implementation. For instance, if the
    /// probability distribution function is of the form `f(x) = 1/g(x)`, the
    /// default implementation can be replaced by the faster division-less test
    /// `a > b * g(x)`; similarly, a faster implementation may exist when the
    /// evaluation of the inverse function `f⁻¹` is less expensive than that of
    /// `f`.
    #[inline]
    fn test(&self, x: T, a: T, b: T) -> bool {
        a * self.eval(x) > b
    }
}

impl<T: Float, F: Fn(T) -> T> UnivariateFn<T> for F {
    #[inline]
    fn eval(&self, x: T) -> T {
        (*self)(x)
    }
}

/// Univariate probability distribution.
pub trait Distribution<T> {
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> T;
}

/// Tail sampling envelope distribution.
pub trait Envelope<T> {
    /// Draw a sample from the envelope distribution and returns it if it passes
    /// the acceptance-rejection check.
    fn try_sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> Option<T>;
}

/// Distribution with bounded support.
#[derive(Clone)]
pub struct DistAny<P, T: Float, F> {
    data: Data<T>,
    func: F,
    phantom_table_size: PhantomData<P>,
}

impl<P, T, F> DistAny<P, T, F>
where
    P: ValidPartitionSize<T>,
    T: Float,
    F: UnivariateFn<T>,
{
    pub fn new(func: F, table: &InitTable<P, T>) -> Self {
        DistAny {
            data: process_table(T::ZERO, table, T::UInt::MAX),
            func: func,
            phantom_table_size: PhantomData,
        }
    }
}

impl<P, T, F> Distribution<T> for DistAny<P, T, F>
where
    P: Partition,
    T: Float,
    F: UnivariateFn<T>,
{
    #[inline]
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> T {
        let u_mask = (T::UInt::ONE << (T::UInt::BITS - P::BITS )) - T::UInt::ONE;

        loop {

            let r = T::UInt::gen(rng);

            // Extract the significand from the rightmost bits.
            let u = r & u_mask;

            // Extract the table index from the P::BITS leftmost bits after the
            // sign bit.
            let i = (r >> (T::UInt::BITS - P::BITS)).as_usize();

            // Test for the common case (point below yinf).
            let d = &self.data.table[i];
            if u <= d.wedge_switch {
                if cfg!(feature = "fma") {
                    return T::cast_uint(u).mul_add(d.alpha, d.beta);
                } else {
                    return d.alpha * T::cast_uint(u) + d.beta;
                }
            }

            // Wedge sampling, test y<f(x).
            let dx = self.data.table[i + 1].beta - d.beta;
            let x = d.beta + T::gen(rng) * dx;
            if self
                .func
                .test(x, dx, T::cast_uint(u) * self.data.scaled_xysup)
            {
                return x;
            }
        }
    }
}

/// Distribution with rejection-sampled tail(s).
#[derive(Clone)]
pub struct DistAnyTailed<P, T: Float, F, E> {
    data: Data<T>,
    func: F,
    tail_envelope: E,
    tail_switch: T::UInt,
    phantom_table_size: PhantomData<P>,
}

impl<P, T, F, E> DistAnyTailed<P, T, F, E>
where
    P: ValidPartitionSize<T>,
    T: Float,
    F: UnivariateFn<T>,
    E: Envelope<T>,
{
    pub fn new(func: F, table: &InitTable<P, T>, tail_envelope: E, tail_area: T) -> Self {
        let tail_switch = compute_tail_switch(table, tail_area);

        DistAnyTailed {
            data: process_table(T::ZERO, table, tail_switch),
            func: func,
            tail_envelope: tail_envelope,
            tail_switch: tail_switch,
            phantom_table_size: PhantomData,
        }
    }
}

impl<P, T, F, E> Distribution<T> for DistAnyTailed<P, T, F, E>
where
    P: Partition,
    T: Float,
    F: UnivariateFn<T>,
    E: Envelope<T>,
{
    #[inline]
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> T {
        let u_mask = (T::UInt::ONE << (T::UInt::BITS - P::BITS )) - T::UInt::ONE;

        loop {

            let r = T::UInt::gen(rng);

            // Extract the significand from the rightmost bits.
            let u = r & u_mask;

            // Extract the table index from the P::BITS leftmost bits after the
            // sign bit.
            let i = (r >> (T::UInt::BITS - P::BITS)).as_usize();

            // Test for the common case (point below yinf).
            let d = &self.data.table[i];
            if u <= d.wedge_switch {
                if cfg!(feature = "fma") {
                    return T::cast_uint(u).mul_add(d.alpha, d.beta);
                } else {
                    return d.alpha * T::cast_uint(u) + d.beta;
                }
            }

            // Check if the tail should be sampled.
            if u > self.tail_switch {
                if let Some(x) = self.tail_envelope.try_sample(rng) {
                    return x;
                }
                continue;
            }

            // Wedge sampling, test y<f(x).
            let dx = self.data.table[i + 1].beta - d.beta;
            let x = d.beta + T::gen(rng) * dx;
            if self
                .func
                .test(x, dx, T::cast_uint(u) * self.data.scaled_xysup)
            {
                return x;
            }
        }
    }
}

/// Distribution with symmetric probability density function about the origin
/// and bounded support.
#[derive(Clone)]
pub struct DistCentral<P, T: Float, F> {
    data: Data<T>,
    func: F,
    phantom_table_size: PhantomData<P>,
}

impl<P, T, F> DistCentral<P, T, F>
where
    P: ValidSymmetricPartitionSize<T>,
    T: Float,
    F: UnivariateFn<T>,
{
    pub fn new(func: F, table: &InitTable<P, T>) -> Self {
        DistCentral {
            data: process_table(T::ZERO, table, T::UInt::MAX),
            func: func,
            phantom_table_size: PhantomData,
        }
    }
}

impl<P, T, F> Distribution<T> for DistCentral<P, T, F>
where
    P: Partition,
    T: Float,
    F: UnivariateFn<T>,
{
    #[inline]
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> T {
        let u_mask = (T::UInt::ONE << (T::UInt::BITS - P::BITS - 1)) - T::UInt::ONE;
        let i_mask = (T::UInt::ONE << P::BITS) - T::UInt::ONE;
        let s_mask = T::UInt::ONE << (T::UInt::BITS - 1);

        loop {

            let mut r = T::UInt::gen(rng);

            // Extract the significand from the rightmost bits.
            let u = r & u_mask;

            // Extract the table index from the P::BITS leftmost bits after the
            // sign bit.
            r = r.arithmetic_right_shift(T::UInt::BITS - P::BITS - 1);
            let i = (r & i_mask).as_usize();

            // Use the leftmost bit as the IEEE sign bit.
            let s = r & s_mask;

            // Test for the common case (point below yinf).
            let d = &self.data.table[i];
            if u <= d.wedge_switch {
                if cfg!(feature = "fma") {
                    return T::bitxor(T::cast_uint(u).mul_add(d.alpha, d.beta), s);
                } else {
                    return T::bitxor(d.alpha * T::cast_uint(u) + d.beta, s);
                }
            }

            // Wedge sampling, test y<f(x).
            let dx = self.data.table[i + 1].beta - d.beta;
            let x = d.beta + T::gen(rng) * dx;
            if self
                .func
                .test(x, dx, T::cast_uint(u) * self.data.scaled_xysup)
            {
                return T::bitxor(x, s);
            }
        }
    }
}

/// Distribution with symmetric probability density function about the origin
/// and rejection-sampled tail(s).
#[derive(Clone)]
pub struct DistCentralTailed<P, T: Float, F, E> {
    data: Data<T>,
    func: F,
    tail_envelope: E,
    tail_switch: T::UInt,
    phantom_table_size: PhantomData<P>,
}

impl<P, T, F, E> DistCentralTailed<P, T, F, E>
where
    P: ValidSymmetricPartitionSize<T>,
    T: Float,
    F: UnivariateFn<T>,
    E: Envelope<T>,
{
    pub fn new(func: F, table: &InitTable<P, T>, tail_envelope: E, tail_area: T) -> Self {
        let tail_switch = compute_tail_switch(table, tail_area);
        DistCentralTailed {
            data: process_table(T::ZERO, table, tail_switch),
            func,
            tail_envelope,
            tail_switch,
            phantom_table_size: PhantomData,
        }
    }
}

impl<P, T, F, E> Distribution<T> for DistCentralTailed<P, T, F, E>
where
    P: Partition,
    T: Float,
    F: UnivariateFn<T>,
    E: Envelope<T>,
{
    #[inline]
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> T {
        let u_mask = (T::UInt::ONE << (T::UInt::BITS - P::BITS - 1)) - T::UInt::ONE;
        let i_mask = (T::UInt::ONE << P::BITS) - T::UInt::ONE;
        let s_mask = T::UInt::ONE << (T::UInt::BITS - 1);

        loop {

            let mut r = T::UInt::gen(rng);

            // Extract the significand from the rightmost bits.
            let u = r & u_mask;

            // Extract the table index from the P::BITS leftmost bits after the
            // sign bit.
            r = r.arithmetic_right_shift(T::UInt::BITS - P::BITS - 1);
            let i = (r & i_mask).as_usize();

            // Use the leftmost bit as the IEEE sign bit.
            let s = r & s_mask;

            // Test for the common case (point below yinf).
            let d = &self.data.table[i];
            if u <= d.wedge_switch {
                if cfg!(feature = "fma") {
                    return T::bitxor(T::cast_uint(u).mul_add(d.alpha, d.beta), s);
                } else {
                    return T::bitxor(d.alpha * T::cast_uint(u) + d.beta, s);
                }
            }

            // Check if the tail should be sampled.
            if u > self.tail_switch {
                if let Some(x) = self.tail_envelope.try_sample(rng) {
                    return T::bitxor(x, s);
                }
                continue;
            }

            // Wedge sampling, test y<f(x).
            let dx = self.data.table[i + 1].beta - d.beta;
            let x = d.beta + T::gen(rng) * dx;
            if self
                .func
                .test(x, dx, T::cast_uint(u) * self.data.scaled_xysup)
            {
                return T::bitxor(x, s);
            }
        }
    }
}

/// Distribution with symmetric probability density function and bounded support.
#[derive(Clone)]
pub struct DistSymmetric<P, T: Float, F> {
    data: Data<T>,
    func: F,
    x0: T,
    phantom_table_size: PhantomData<P>,
}

impl<P, T, F> DistSymmetric<P, T, F>
where
    P: ValidSymmetricPartitionSize<T>,
    T: Float,
    F: UnivariateFn<T>,
{
    pub fn new(x0: T, func: F, table: &InitTable<P, T>) -> Self {
        DistSymmetric {
            data: process_table(x0, table, T::UInt::MAX),
            func: func,
            x0: x0,
            phantom_table_size: PhantomData,
        }
    }
}

impl<P, T, F> Distribution<T> for DistSymmetric<P, T, F>
where
    P: Partition,
    T: Float,
    F: UnivariateFn<T>,
{
    #[inline]
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> T {
        let u_mask = (T::UInt::ONE << (T::UInt::BITS - P::BITS - 1)) - T::UInt::ONE;
        let i_mask = (T::UInt::ONE << P::BITS) - T::UInt::ONE;
        let s_mask = T::UInt::ONE << (T::UInt::BITS - 1);

        loop {

            let mut r = T::UInt::gen(rng);

            // Extract the significand from the rightmost bits.
            let u = r & u_mask;

            // Extract the table index from the P::BITS leftmost bits after the
            // sign bit.
            r = r.arithmetic_right_shift(T::UInt::BITS - P::BITS - 1);
            let i = (r & i_mask).as_usize();

            // Use the leftmost bit as the IEEE sign bit.
            let s = r & s_mask;

            // Test for the common case (point below yinf).
            let d = &self.data.table[i];
            if u <= d.wedge_switch {
                if cfg!(feature = "fma") {
                    return self.x0 + T::bitxor(T::cast_uint(u).mul_add(d.alpha, d.beta), s);
                } else {
                    return self.x0 + T::bitxor(d.alpha * T::cast_uint(u) + d.beta, s);
                }
            }

            // Wedge sampling, test y<f(x).
            let dx = self.data.table[i + 1].beta - d.beta;
            let delta_x = d.beta + T::gen(rng) * dx;
            if self.func.test(
                self.x0 + delta_x,
                dx,
                T::cast_uint(u) * self.data.scaled_xysup,
            ) {
                return self.x0 + T::bitxor(delta_x, s);
            }
        }
    }
}

/// Distribution with symmetric probability density function and rejection-sampled tail(s).
#[derive(Clone)]
pub struct DistSymmetricTailed<P, T: Float, F, E> {
    data: Data<T>,
    func: F,
    x0: T,
    tail_envelope: E,
    tail_switch: T::UInt,
    phantom_table_size: PhantomData<P>,
}

impl<P, T, F, E> DistSymmetricTailed<P, T, F, E>
where
    P: ValidSymmetricPartitionSize<T>,
    T: Float,
    F: UnivariateFn<T>,
    E: Envelope<T>,
{
    pub fn new(x0: T, func: F, table: &InitTable<P, T>, tail_envelope: E, tail_area: T) -> Self {
        let tail_switch = compute_tail_switch(table, tail_area);

        DistSymmetricTailed {
            data: process_table(x0, table, tail_switch),
            func: func,
            x0: x0,
            tail_envelope: tail_envelope,
            tail_switch: tail_switch,
            phantom_table_size: PhantomData,
        }
    }
}

impl<P, T, F, E> Distribution<T> for DistSymmetricTailed<P, T, F, E>
where
    P: Partition,
    T: Float,
    F: UnivariateFn<T>,
    E: Envelope<T>,
{
    #[inline]
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> T {
        let u_mask = (T::UInt::ONE << (T::UInt::BITS - P::BITS - 1)) - T::UInt::ONE;
        let i_mask = (T::UInt::ONE << P::BITS) - T::UInt::ONE;
        let s_mask = T::UInt::ONE << (T::UInt::BITS - 1);

        loop {

            let mut r = T::UInt::gen(rng);

            // Extract the significand from the rightmost bits.
            let u = r & u_mask;

            // Extract the table index from the P::BITS leftmost bits after the
            // sign bit.
            r = r.arithmetic_right_shift(T::UInt::BITS - P::BITS - 1);
            let i = (r & i_mask).as_usize();

            // Use the leftmost bit as the IEEE sign bit.
            let s = r & s_mask;

            // Test for the common case (point below yinf).
            let d = &self.data.table[i];
            if u <= d.wedge_switch {
                if cfg!(feature = "fma") {
                    return self.x0 + T::bitxor(T::cast_uint(u).mul_add(d.alpha, d.beta), s);
                } else {
                    return self.x0 + T::bitxor(d.alpha * T::cast_uint(u) + d.beta, s);
                }
            }

            // Check if the tail should be sampled.
            if u > self.tail_switch {
                if let Some(x) = self.tail_envelope.try_sample(rng) {
                    return self.x0 + T::bitxor(x - self.x0, s);
                }
                continue;
            }

            // Wedge sampling, test y<f(x).
            let dx = self.data.table[i + 1].beta - d.beta;
            let delta_x = d.beta + T::gen(rng) * dx;
            if self.func.test(
                self.x0 + delta_x,
                dx,
                T::cast_uint(u) * self.data.scaled_xysup,
            ) {
                return self.x0 + T::bitxor(delta_x, s);
            }
        }
    }
}

// Distribution table datum.
#[repr(C)]
#[derive(Copy, Clone)]
struct TableDatum<T: Float> {
    alpha: T,              // (x[i+1] - x[i]) / wedge_switch[i]
    beta: T,               // x[i] - x0
    wedge_switch: T::UInt, // (yinf / ysup) * tail_switch
}

#[derive(Clone)]
struct Data<T: Float> {
    table: Vec<TableDatum<T>>,
    scaled_xysup: T, // dx * ysup / tail_switch
}

// Generates an optimized lookup table from a quadrature table.
fn process_table<P, T>(x0: T, init_table: &InitTable<P, T>, tail_switch: T::UInt) -> Data<T>
where
    P: Partition,
    T: Float,
{
    let max_bit_loss = T::ONE;
    let n = P::SIZE;
    let mut table = Vec::with_capacity(n + 1);

    // Convenient aliases.
    let x = init_table.x.as_ref();
    let yinf = init_table.yinf.as_ref();
    let ysup = init_table.ysup.as_ref();

    // Compute the final table.
    for i in 0..n {
        // When a rectangular quartile is sampled, the position between x[i] and
        // x[i+1] is generated using a random number within the range
        // [0:(yinf/ysup)*tail_switch]. When yinf/ysup is very small, however,
        // this implies that the position is computed with a very coarse
        // resolution. In order to avoid this loss of sampling quality, yinf is
        // in such case set to 0, which unconditionally forces the use of the
        // more expensive but higher quality wedge sampling algorithm.
        let w = yinf[i] / ysup[i] * T::cast_uint(tail_switch);
        let bit_loss = T::cast_u32(T::SIGNIFICAND_BITS) - w.log2();
        let (wedge_switch, alpha) = if bit_loss <= max_bit_loss {
            // Coefficients for the baseline sampling algorithm.
            (w.round_as_uint(), (x[i + 1] - x[i]) / w)
        } else {
            // Degraded case: force wedge sampling algorithm.
            (T::UInt::ZERO, T::ZERO)
        };

        table.push(TableDatum {
            alpha,
            beta: x[i] - x0,
            wedge_switch,
        });
    }

    // Last datum is dummy except for the x value.
    table.push(TableDatum {
        alpha: T::ZERO, // never used
        beta: x[n] - x0,
        wedge_switch: T::UInt::ZERO, // never used
    });

    // Scaled area of a single rectangle.
    let scaled_xysup = (x[1] - x[0]) * ysup[0] / T::cast_uint(tail_switch);

    Data {
        table,
        scaled_xysup,
    }
}

// Computes the integer used as a threshold for tail sampling.
fn compute_tail_switch<P, T>(init_table: &InitTable<P, T>, tail_area: T) -> T::UInt
where
    P: Partition,
    T: Float,
{
    // Convenient aliases.
    let x = init_table.x.as_ref();
    let ysup = init_table.ysup.as_ref();

    let mut area = T::ZERO;
    for i in 0..P::SIZE {
        area = area + (x[i + 1] - x[i]) * ysup[i];
    }
    let max_switch = T::cast_uint((T::UInt::ONE << (T::UInt::BITS - P::BITS - 1)) - T::UInt::ONE);
    
    (max_switch * (area / (area + tail_area))).round_as_uint()
}

