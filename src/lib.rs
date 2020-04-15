// Internal traits.
pub use num::*;
use partition::*;
pub use table_validation::*;

// External traits.
use rand::{distributions::Distribution, Rng, RngCore};
use std::marker::PhantomData;

// Modules.
mod num;
pub mod partition;
mod table_validation;
pub mod util;

/// Tail sampling envelope distribution.
pub trait Envelope<T> {
    /// Draw a sample from the envelope distribution and returns it if it passes
    /// the acceptance-rejection check.
    fn try_sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<T>;
}

/// Distribution with arbitrarily-shaped probability density function and bounded support.
#[derive(Clone)]
pub struct DistAny<P, T: Float, F> {
    data: Vec<Datum<T>>,
    func: F,
    phantom_table_size: PhantomData<P>,
}

impl<P, T, F> DistAny<P, T, F>
where
    P: ValidPartitionSize<T>,
    T: Float,
    F: Fn(T) -> T,
{
    pub fn new(func: F, table: &InitTable<P, T>) -> Self {
        let tail_switch = T::GenInt::ONE << (T::GenInt::BITS - P::BITS);

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
    F: Fn(T) -> T,
{
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> T {
        loop {
            let r = T::GenInt::gen(rng);

            // Extract the significand from the leftmost bits after the table index.
            let u_mask = (T::GenInt::ONE << (T::GenInt::BITS - P::BITS)) - T::GenInt::ONE;
            let u = r & u_mask;

            // Extract the table index from the leftmost bits.
            let i = r >> (T::GenInt::BITS - P::BITS);

            // Test for the common case (point below yinf).
            let i: usize = i.as_usize();
            let d = &self.data[i];
            if u < d.scaled_yratio {
                return d.x + d.scaled_dx * T::cast_gen_int(u);
            }

            // Wedge sampling, test y<f(x).
            let x0 = d.x;
            let x1 = self.data[i + 1].x;
            let x = x0 + T::gen(rng) * (x1 - x0);
            if T::cast_gen_int(u) * d.scaled_ysup < (self.func)(x) {
                return x;
            }
        }
    }
}

/// Distribution with arbitrarily-shaped probability density function and rejection-sampled tail.
#[derive(Clone)]
pub struct DistAnyTailed<P, T: Float, F, E> {
    data: Vec<Datum<T>>,
    func: F,
    tail_envelope: E,
    tail_switch: T::GenInt,
    phantom_table_size: PhantomData<P>,
}

impl<P, T, F, E> DistAnyTailed<P, T, F, E>
where
    P: ValidPartitionSize<T>,
    T: Float,
    F: Fn(T) -> T,
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
    F: Fn(T) -> T,
    E: Envelope<T>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
        loop {
            let r = T::GenInt::gen(rng);

            // Extract the significand from the leftmost bits after the table index.
            let u_mask = (T::GenInt::ONE << (T::GenInt::BITS - P::BITS)) - T::GenInt::ONE;
            let u = r & u_mask;

            // Extract the table index from the leftmost bits.
            let i = r >> (T::GenInt::BITS - P::BITS);

            // Test for the common case (point below yinf).
            let i = i.as_usize();
            let d = &self.data[i];
            if u < d.scaled_yratio {
                return d.x + d.scaled_dx * T::cast_gen_int(u);
            }

            // Check if the tail should be sampled.
            if u >= self.tail_switch {
                if let Some(x) = self.tail_envelope.try_sample(rng) {
                    return x;
                }
                continue;
            }

            // Wedge sampling, test y<f(x).
            let x0 = d.x;
            let x1 = self.data[i + 1].x;
            let x = x0 + T::gen(rng) * (x1 - x0);
            if T::cast_gen_int(u) * d.scaled_ysup < (self.func)(x) {
                return x;
            }
        }
    }
}

/// Distribution with symmetric probability density function about the origin and bounded support.
#[derive(Clone)]
pub struct DistCentral<P, T: Float, F> {
    data: Vec<Datum<T>>,
    func: F,
    phantom_table_size: PhantomData<P>,
}

impl<P, T, F> DistCentral<P, T, F>
where
    P: ValidSymmetricPartitionSize<T>,
    T: Float,
    F: Fn(T) -> T,
{
    pub fn new(func: F, table: &InitTable<P, T>) -> Self {
        let tail_switch = T::GenInt::ONE << (T::GenInt::BITS - P::BITS - 1);

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
    F: Fn(T) -> T,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
        loop {
            let r = T::GenInt::gen(rng);

            // Extract the significand from the leftmost bits after the table index.
            let u_mask = (T::GenInt::ONE << (T::GenInt::BITS - P::BITS - 1)) - T::GenInt::ONE;
            let u = r & u_mask;

            // Extract the table index from the P leftmost bits after the sign bit.
            let i_mask = (T::GenInt::ONE << P::BITS) - T::GenInt::ONE;
            let i = (r >> (T::GenInt::BITS - P::BITS - 1)) & i_mask;

            // Extract the sign from the leftmost bit.
            let s = if (r >> (T::GenInt::BITS - 1)) != T::GenInt::ZERO {
                T::ONE
            } else {
                -T::ONE
            };

            // Test for the common case (point below yinf).
            let i = i.as_usize();
            let d = &self.data[i];
            if u < d.scaled_yratio {
                return s * (d.x + d.scaled_dx * T::cast_gen_int(u));
            }

            // Wedge sampling, test y<f(x).
            let x0 = d.x;
            let x1 = self.data[i + 1].x;
            let x = x0 + T::gen(rng) * (x1 - x0); // here x is relative to the origin
            if T::cast_gen_int(u) * d.scaled_ysup < (self.func)(x) {
                return s * x;
            }
        }
    }
}

/// Distribution with symmetric probability density function about the origin and rejection-sampled tail.
#[derive(Clone)]
pub struct DistCentralTailed<P, T: Float, F, E> {
    data: Vec<Datum<T>>,
    func: F,
    tail_envelope: E,
    tail_switch: T::GenInt,
    phantom_table_size: PhantomData<P>,
}

impl<P, T, F, E> DistCentralTailed<P, T, F, E>
where
    P: ValidSymmetricPartitionSize<T>,
    T: Float,
    F: Fn(T) -> T,
    E: Envelope<T>,
{
    pub fn new(func: F, table: &InitTable<P, T>, tail_envelope: E, tail_area: T) -> Self {
        let tail_switch = compute_tail_switch(table, tail_area, 1);

        DistCentralTailed {
            data: process_table(T::ZERO, table, tail_switch),
            func: func,
            tail_envelope: tail_envelope,
            tail_switch: tail_switch,
            phantom_table_size: PhantomData,
        }
    }
}

impl<P, T, F, E> Distribution<T> for DistCentralTailed<P, T, F, E>
where
    P: Partition,
    T: Float,
    F: Fn(T) -> T,
    E: Envelope<T>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
        loop {
            let r = T::GenInt::gen(rng);

            // Extract the significand from the leftmost bits after the table index.
            let u_mask = (T::GenInt::ONE << (T::GenInt::BITS - P::BITS - 1)) - T::GenInt::ONE;
            let u = r & u_mask;

            // Extract the table index from the P leftmost bits after the sign bit.
            let i_mask = (T::GenInt::ONE << P::BITS) - T::GenInt::ONE;
            let i = (r >> (T::GenInt::BITS - P::BITS - 1)) & i_mask;

            // Extract the sign from the leftmost bit.
            let s = if (r >> (T::GenInt::BITS - 1)) != T::GenInt::ZERO {
                T::ONE
            } else {
                -T::ONE
            };

            // Test for the common case (point below yinf).
            let i = i.as_usize();
            let d = &self.data[i];
            if u < d.scaled_yratio {
                return s * (d.x + d.scaled_dx * T::cast_gen_int(u));
            }

            // Check if the tail should be sampled.
            if u >= self.tail_switch {
                if let Some(x) = self.tail_envelope.try_sample(rng) {
                    return s * x;
                }
                continue;
            }

            // Wedge sampling, test y<f(x).
            let x0 = d.x;
            let x1 = self.data[i + 1].x;
            let x = x0 + T::gen(rng) * (x1 - x0); // here x is relative to the origin
            if T::cast_gen_int(u) * d.scaled_ysup < (self.func)(x) {
                return s * x;
            }
        }
    }
}

/// Distribution with symmetric probability density function and bounded support.
#[derive(Clone)]
pub struct DistSymmetric<P, T: Float, F> {
    data: Vec<Datum<T>>,
    func: F,
    x0: T,
    phantom_table_size: PhantomData<P>,
}

impl<P, T, F> DistSymmetric<P, T, F>
where
    P: ValidSymmetricPartitionSize<T>,
    T: Float,
    F: Fn(T) -> T,
{
    pub fn new(x0: T, func: F, table: &InitTable<P, T>) -> Self {
        let tail_switch = T::GenInt::ONE << (T::GenInt::BITS - P::BITS - 1);

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
    F: Fn(T) -> T,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
        loop {
            let r = T::GenInt::gen(rng);

            // Extract the significand from the leftmost bits after the table index.
            let u_mask = (T::GenInt::ONE << (T::GenInt::BITS - P::BITS - 1)) - T::GenInt::ONE;
            let u = r & u_mask;

            // Extract the table index from the P leftmost bits after the sign bit.
            let i_mask = (T::GenInt::ONE << P::BITS) - T::GenInt::ONE;
            let i = (r >> (T::GenInt::BITS - P::BITS - 1)) & i_mask;

            // Extract the sign from the leftmost bit.
            let s = if (r >> (T::GenInt::BITS - 1)) != T::GenInt::ZERO {
                T::ONE
            } else {
                -T::ONE
            };

            // Test for the common case (point below yinf).
            let i = i.as_usize();
            let d = &self.data[i];
            if u < d.scaled_yratio {
                return self.x0 + s * (d.x + d.scaled_dx * T::cast_gen_int(u));
            }

            // Wedge sampling, test y<f(x).
            let x0 = d.x;
            let x1 = self.data[i + 1].x;
            let x = x0 + T::gen(rng) * (x1 - x0); // here x is relative to the origin
            if T::cast_gen_int(u) * d.scaled_ysup < (self.func)(x + self.x0) {
                return self.x0 + s * x;
            }
        }
    }
}

/// Distribution with symmetric probability density function and rejection-sampled tail.
#[derive(Clone)]
pub struct DistSymmetricTailed<P, T: Float, F, E> {
    data: Vec<Datum<T>>,
    func: F,
    x0: T,
    tail_envelope: E,
    tail_switch: T::GenInt,
    phantom_table_size: PhantomData<P>,
}

impl<P, T, F, E> DistSymmetricTailed<P, T, F, E>
where
    P: ValidSymmetricPartitionSize<T>,
    T: Float,
    F: Fn(T) -> T,
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
    F: Fn(T) -> T,
    E: Envelope<T>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
        loop {
            let r = T::GenInt::gen(rng);

            // Extract the significand from the leftmost bits after the table index.
            let u_mask = (T::GenInt::ONE << (T::GenInt::BITS - P::BITS - 1)) - T::GenInt::ONE;
            let u = r & u_mask;

            // Extract the table index from the P leftmost bits after the sign bit.
            let i_mask = (T::GenInt::ONE << P::BITS) - T::GenInt::ONE;
            let i = (r >> (T::GenInt::BITS - P::BITS - 1)) & i_mask;

            // Extract the sign from the leftmost bit.
            let s = if (r >> (T::GenInt::BITS - 1)) != T::GenInt::ZERO {
                T::ONE
            } else {
                -T::ONE
            };

            // Test for the common case (point below yinf).
            let i = i.as_usize();
            let d = &self.data[i];
            if u < d.scaled_yratio {
                return self.x0 + s * (d.x + d.scaled_dx * T::cast_gen_int(u));
            }

            // Check if the tail should be sampled.
            if u >= self.tail_switch {
                if let Some(x) = self.tail_envelope.try_sample(rng) {
                    return self.x0 + s * (x - self.x0);
                }
                continue;
            }

            // Wedge sampling, test y<f(x).
            let x0 = d.x;
            let x1 = self.data[i + 1].x;
            let x = x0 + T::gen(rng) * (x1 - x0); // here x is relative to the origin
            if T::cast_gen_int(u) * d.scaled_ysup < (self.func)(x + self.x0) {
                return self.x0 + s * x;
            }
        }
    }
}

// Distribution table datum.
#[derive(Copy, Clone)]
struct Datum<T: Float> {
    x: T,
    scaled_dx: T,             // dx/scaled_yratio
    scaled_ysup: T,           // ysup/tail_switch
    scaled_yratio: T::GenInt, // (yinf/ysup)*tail_switch
}

// Generates an optimized lookup table from a quadrature table.
fn process_table<P, T>(x0: T, table: &InitTable<P, T>, tail_switch: T::GenInt) -> Vec<Datum<T>>
where
    P: Partition,
    T: Float,
{
    let one_half = T::ONE / (T::ONE + T::ONE);
    let n = P::SIZE;
    let mut data = Vec::with_capacity(n + 1);

    // Convenient aliases.
    let x = table.x.as_ref();
    let yinf = table.yinf.as_ref();
    let ysup = table.ysup.as_ref();

    // Compute the final table.
    for i in 0..n {
        let yratio = yinf[i] / ysup[i];
        let scaled_yratio = if yratio >= one_half {
            // Use baseline algorithm.
            (yratio * T::cast_gen_int(tail_switch)).round_as_gen_int()
        } else {
            // Because the random number is mapped to [0:ysup], if yinf < 0.5*ysup then
            // more than 1 bit of accuracy will be lost after a random number in [0:ysup]
            // is narrowed down to [0:yinf] and subsequently reused as a random number within
            // this latter interval.
            // To prevent this loss of accuracy, wedge sampling is forced by setting yinf=0.
            T::GenInt::ZERO
        };
        // ysup is scaled such that, once multiplied by an integer random number less that
        // the tail sampling threshold, its value will be in [0:ysup].
        let scaled_ysup = ysup[i] / T::cast_gen_int(tail_switch);
        // dx is scaled such that, once multiplied by a random number less than the
        // critical wedge sampling threshold, its value will be in [0:dx].
        let scaled_dx = (x[i + 1] - x[i]) / T::cast_gen_int(scaled_yratio);

        data.push(Datum {
            x: x[i] - x0,
            scaled_dx: scaled_dx,
            scaled_ysup: scaled_ysup,
            scaled_yratio: scaled_yratio,
        });
    }

    // Last datum is dummy except for the x value.
    data.push(Datum {
        x: x[n] - x0,
        scaled_dx: T::ZERO,
        scaled_ysup: T::ZERO,
        scaled_yratio: T::GenInt::ZERO,
    });

    data
}

// Computes the integer used as a threshold for tail sampling.
fn compute_tail_switch<P, T>(table: &InitTable<P, T>, tail_area: T, sign_bits: u32) -> T::GenInt
where
    P: Partition,
    T: Float,
{
    let max_switch = T::GenInt::ONE << (T::GenInt::BITS - P::BITS - sign_bits);

    // Convenient aliases.
    let x = table.x.as_ref();
    let ysup = table.ysup.as_ref();

    let mut area = T::ZERO;
    for i in 0..P::SIZE {
        area = area + (x[i + 1] - x[i]) * ysup[i];
    }
    let switch = T::cast_gen_int(max_switch) * (area / (area + tail_area));

    switch.round_as_gen_int()
}
