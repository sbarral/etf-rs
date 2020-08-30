//! Primitive ETF distributions and related utilities.

// Internal traits.
use crate::num::{UInt, Float, Func};
use partition::*;
pub use table_validation::*;

// External traits.
use rand::{distributions::Distribution, Rng};
use std::marker::PhantomData;

// Modules.
pub mod partition;
mod table_validation;
pub mod util;

/// Tail sampling envelope distribution.
pub trait Envelope<T> {
    /// Draw a sample from the envelope distribution and returns it if it passes
    /// the acceptance-rejection check.
    fn try_sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<T>;
}

/*
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
    F: Func<T>,
{
    pub fn new(func: F, table: &InitTable<P, T>) -> Self {
        let tail_switch = T::UInt::ONE << (T::UInt::BITS - P::BITS);

        DistAny {
            data: process_table(T::ZERO, table, tail_switch),
            func: func,
            phantom_table_size: PhantomData,
        }
    }
}

impl<P, T, F> Distribution<T> for DistAny<P, T, F>
where
    P: Partition,
    T: Float,
    F: Func<T>,
{
    #[inline(always)]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
        loop {
            let r = T::UInt::gen(rng);

            // Extract the significand from the leftmost bits after the table index.
            let u_mask = (T::UInt::ONE << (T::UInt::BITS - P::BITS)) - T::UInt::ONE;
            let u = r & u_mask;

            // Extract the table index from the leftmost bits.
            let i = r >> (T::UInt::BITS - P::BITS);

            // Test for the common case (point below yinf).
            let i: usize = i.as_usize();
            let d = &self.data.table[i];
            if u < d.scaled_yratio {
                return d.x + d.scaled_dx * T::cast_uint(u);
            }

            // Wedge sampling, test y<f(x).
            let dx = self.data.table[i + 1].x - d.x;
            let x = d.x + T::gen(rng) * dx;
            if T::cast_uint(u) * self.data.scaled_xysup < self.func.eval(x)*dx {
                return x;
            }
        }
    }
}

/// Distribution with rejection-sampled tail.
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
    F: Func<T>,
    E: Envelope<T>,
{
    pub fn new(func: F, table: &InitTable<P, T>, tail_envelope: E, tail_area: T) -> Self {
        let tail_switch = compute_tail_switch(table, tail_area, 0);

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
    F: Func<T>,
    E: Envelope<T>,
{
    #[inline(always)]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
        loop {
            let r = T::UInt::gen(rng);

            // Extract the significand from the leftmost bits after the table index.
            let u_mask = (T::UInt::ONE << (T::UInt::BITS - P::BITS)) - T::UInt::ONE;
            let u = r & u_mask;

            // Extract the table index from the leftmost bits.
            let i = r >> (T::UInt::BITS - P::BITS);

            // Test for the common case (point below yinf).
            let i = i.as_usize();
            let d = &self.data.table[i];
            if u < d.scaled_yratio {
                return d.x + d.scaled_dx * T::cast_uint(u);
            }

            // Check if the tail should be sampled.
            if u >= self.tail_switch {
                if let Some(x) = self.tail_envelope.try_sample(rng) {
                    return x;
                }
                continue;
            }

            // Wedge sampling, test y<f(x).
            let dx = self.data.table[i + 1].x - d.x;
            let x = d.x + T::gen(rng) * dx;
            if T::cast_uint(u) * self.data.scaled_xysup < self.func.eval(x)*dx {
                return x;
            }
        }
    }
}

/// Distribution with symmetric probability density function about the origin and bounded support.
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
    F: Func<T>,
{
    pub fn new(func: F, table: &InitTable<P, T>) -> Self {
        let tail_switch = T::UInt::ONE << (T::UInt::BITS - P::BITS - 1);

        DistCentral {
            data: process_table(T::ZERO, table, tail_switch),
            func: func,
            phantom_table_size: PhantomData,
        }
    }
}

impl<P, T, F> Distribution<T> for DistCentral<P, T, F>
where
    P: Partition,
    T: Float,
    F: Func<T>,
{
    #[inline(always)]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
        loop {
            let r = T::UInt::gen(rng);

            // Extract the significand from the leftmost bits after the table index.
            let u_mask = (T::UInt::ONE << (T::UInt::BITS - P::BITS - 1)) - T::UInt::ONE;
            let u = r & u_mask;

            // Extract the table index from the P leftmost bits after the sign bit.
            let i_mask = (T::UInt::ONE << P::BITS) - T::UInt::ONE;
            let i = (r >> (T::UInt::BITS - P::BITS - 1)) & i_mask;

            // Extract the sign from the leftmost bit.
            let s = if (r >> (T::UInt::BITS - 1)) != T::UInt::ZERO {
                T::ONE
            } else {
                -T::ONE
            };

            // Test for the common case (point below yinf).
            let i = i.as_usize();
            let d = &self.data.table[i];
            if u < d.scaled_yratio {
                return s * (d.x + d.scaled_dx * T::cast_uint(u));
            }

            // Wedge sampling, test y<f(x).
            let dx = self.data.table[i + 1].x - d.x;
            let x = d.x + T::gen(rng) * dx;
            if T::cast_uint(u) * self.data.scaled_xysup < self.func.eval(x)*dx {
                return s * x;
            }
        }
    }
}
*/

/// Distribution with symmetric probability density function about the origin and rejection-sampled tail.
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
    F: Func<T>,
    E: Envelope<T>,
{
    pub fn new(func: F, table: &InitTable<P, T>, tail_envelope: E, tail_area: T) -> Self {
        let tail_switch = compute_tail_switch(table, tail_area, 1);
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
    F: Func<T>,
    E: Envelope<T>,
{
    #[inline(always)]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
        loop {
            let r = T::UInt::gen(rng);

            // Extract the significand from the leftmost bits after the table index.
            let u_mask = (T::UInt::ONE << (T::UInt::BITS - P::BITS - 1)) - T::UInt::ONE;
            let u = r & u_mask;

            // Extract the table index from the P leftmost bits after the sign bit.
            let i_mask = (T::UInt::ONE << P::BITS) - T::UInt::ONE;
            let i = ((r >> (T::UInt::BITS - P::BITS - 1)) & i_mask).as_usize();

            // Extract the sign from the leftmost bit.
            let s = if (r >> (T::UInt::BITS - 1)) != T::UInt::ZERO {
                T::ONE
            } else {
                -T::ONE
            };
            
            // Test for the common case (point below yinf).
            let d = &self.data.table[i];
            if u < d.scaled_yratio {
                return s * (d.x + d.scaled_dx * T::cast_uint(u));
            }

            // Check if the tail should be sampled.
            if u >= self.tail_switch {
                if let Some(x) = self.tail_envelope.try_sample(rng) {
                    return s * x;
                }
                continue;
            }

            // Wedge sampling, test y<f(x).
            let dx = self.data.table[i + 1].x - d.x;
            let x = d.x + T::gen(rng) * dx;
            if T::cast_uint(u) * self.data.scaled_xysup < self.func.eval(x)*dx {
                return s * x;
            }
        }
    }
}

/*
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
    F: Func<T>,
{
    pub fn new(x0: T, func: F, table: &InitTable<P, T>) -> Self {
        let tail_switch = T::UInt::ONE << (T::UInt::BITS - P::BITS - 1);

        DistSymmetric {
            data: process_table(x0, table, tail_switch),
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
    F: Func<T>,
{
    #[inline(always)]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
        loop {
            let r = T::UInt::gen(rng);

            // Extract the significand from the leftmost bits after the table index.
            let u_mask = (T::UInt::ONE << (T::UInt::BITS - P::BITS - 1)) - T::UInt::ONE;
            let u = r & u_mask;

            // Extract the table index from the P leftmost bits after the sign bit.
            let i_mask = (T::UInt::ONE << P::BITS) - T::UInt::ONE;
            let i = (r >> (T::UInt::BITS - P::BITS - 1)) & i_mask;

            // Extract the sign from the leftmost bit.
            let s = if (r >> (T::UInt::BITS - 1)) != T::UInt::ZERO {
                T::ONE
            } else {
                -T::ONE
            };

            // Test for the common case (point below yinf).
            let i = i.as_usize();
            let d = &self.data.table[i];
            if u < d.scaled_yratio {
                return self.x0 + s * (d.x + d.scaled_dx * T::cast_uint(u));
            }

            // Wedge sampling, test y<f(x).
            let dx = self.data.table[i + 1].x - d.x;
            let x = d.x + T::gen(rng) * dx;
            if T::cast_uint(u) * self.data.scaled_xysup < self.func.eval(x + self.x0)*dx {
                return self.x0 + s * x;
            }
        }
    }
}
*/
/// Distribution with symmetric probability density function and rejection-sampled tail.
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
    F: Func<T>,
    E: Envelope<T>,
{
    pub fn new(x0: T, func: F, table: &InitTable<P, T>, tail_envelope: E, tail_area: T) -> Self {
        let tail_switch = compute_tail_switch(table, tail_area, 1);

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
    F: Func<T>,
    E: Envelope<T>,
{
    #[inline(always)]
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
        loop {
            let r = T::UInt::gen(rng);

            // Extract the significand from the leftmost bits after the table index.
            let u_mask = (T::UInt::ONE << (T::UInt::BITS - P::BITS - 1)) - T::UInt::ONE;
            let u = r & u_mask;

            // Extract the table index from the P leftmost bits after the sign bit.
            let i_mask = (T::UInt::ONE << P::BITS) - T::UInt::ONE;
            let i = (r >> (T::UInt::BITS - P::BITS - 1)) & i_mask;

            // Extract the sign from the leftmost bit.
            let s = if (r >> (T::UInt::BITS - 1)) != T::UInt::ZERO {
                T::ONE
            } else {
                -T::ONE
            };

            // Test for the common case (point below yinf).
            let i = i.as_usize();
            let d = &self.data.table[i];
            if u < d.scaled_yratio {
                return self.x0 + s * (d.x + d.scaled_dx * T::cast_uint(u));
            }

            // Check if the tail should be sampled.
            if u >= self.tail_switch {
                if let Some(x) = self.tail_envelope.try_sample(rng) {
                    return self.x0 + s * (x - self.x0);
                }
                continue;
            }

            // Wedge sampling, test y<f(x).
            let dx = self.data.table[i + 1].x - d.x;
            let x = d.x + T::gen(rng) * dx;
            if T::cast_uint(u) * self.data.scaled_xysup < self.func.eval(x + self.x0)*dx {
                return self.x0 + s * x;
            }
        }
    }
}


// Distribution table datum.
#[derive(Copy, Clone)]
struct TableDatum<T: Float> {
    x: T,
    scaled_dx: T,           // dx / scaled_yratio
    scaled_yratio: T::UInt, // (yinf / ysup) * tail_switch
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
    let one_half = T::ONE / (T::ONE + T::ONE);
    let n = P::SIZE;
    let mut table = Vec::with_capacity(n + 1);

    // Convenient aliases.
    let x = init_table.x.as_ref();
    let yinf = init_table.yinf.as_ref();
    let ysup = init_table.ysup.as_ref();

    // Compute the final table.
    for i in 0..n {
        let yratio = yinf[i] / ysup[i];
        let scaled_yratio = if yratio >= one_half {
            // Use baseline algorithm.
            (yratio * T::cast_uint(tail_switch)).round_as_uint()
        } else {
            // Because the random number is mapped to [0:ysup], if yinf < 0.5*ysup then
            // more than 1 bit of accuracy will be lost after a random number in [0:ysup]
            // is narrowed down to [0:yinf] and subsequently reused as a random number within
            // this latter interval.
            // To prevent this loss of accuracy, wedge sampling is forced by setting yinf=0.
            T::UInt::ZERO
        };
        // dx is scaled such that, once multiplied by a random number less than the
        // critical wedge sampling threshold, its value will be in [0:dx].
        let scaled_dx = (x[i + 1] - x[i]) / T::cast_uint(scaled_yratio);

        table.push(TableDatum {
            x: x[i] - x0,
            scaled_dx: scaled_dx,
            scaled_yratio: scaled_yratio,
        });
    }

    // Last datum is dummy except for the x value.
    table.push(TableDatum {
        x: x[n] - x0,
        scaled_dx: T::ZERO,
        scaled_yratio: T::UInt::ZERO,
    });

    // Scaled area of a single rectangle.
    let scaled_xysup = (x[1] - x[0])*ysup[0]/T::cast_uint(tail_switch);

    Data {
        table,
        scaled_xysup,
    }
}

// Computes the integer used as a threshold for tail sampling.
fn compute_tail_switch<P, T>(init_table: &InitTable<P, T>, tail_area: T, sign_bits: u32) -> T::UInt
where
    P: Partition,
    T: Float,
{
    let max_switch = T::UInt::ONE << (T::UInt::BITS - P::BITS - sign_bits);

    // Convenient aliases.
    let x = init_table.x.as_ref();
    let ysup = init_table.ysup.as_ref();

    let mut area = T::ZERO;
    for i in 0..P::SIZE {
        area = area + (x[i + 1] - x[i]) * ysup[i];
    }
    let switch = T::cast_uint(max_switch) * (area / (area + tail_area));
    switch.round_as_uint()
}
