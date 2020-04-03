// Internal traits.
use num::*;
use table::*;

// External traits.
use rand::distributions::Distribution;
use rand::{Rng, RngCore};
use std::marker::PhantomData;

// Modules.
pub mod num;
pub mod table;
pub mod util;

/// Tail sampling envelope distribution.
pub trait Envelope<T> {
    /// Draw a sample from the envelope distribution and returns it if it passes
    /// the acceptance-rejection check.
    fn try_sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Option<T>;
}

/// Distribution with arbitrarily-shaped PDF.
#[derive(Clone)]
pub struct DistAny<T: Float, F, N> {
    data: Vec<Datum<T>>,
    func: F,
    phantom_table_size: PhantomData<N>,
}

impl<T, F, N> DistAny<T, F, N>
where
    T: Float,
    F: Fn(T) -> T,
    N: ValidTableSize<T>,
{
    pub fn new<A>(func: F, table: &A) -> Self
    where
        A: Table<T, Size = N>,
    {
        let tail_switch = T::GenInt::ONE << (T::GenInt::BITS - A::Size::BITS);

        DistAny {
            data: process_table(T::ZERO, table, tail_switch),
            func: func,
            phantom_table_size: PhantomData,
        }
    }
}

impl<T, F, N> Distribution<T> for DistAny<T, F, N>
where
    T: Float,
    F: Fn(T) -> T,
    N: TableSize,
{
    fn sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> T {
        loop {
            let r = T::GenInt::gen(rng);

            // Extract the significand from the leftmost bits after the table index.
            let u_mask = (T::GenInt::ONE << (T::GenInt::BITS - N::BITS)) - T::GenInt::ONE;
            let u = r & u_mask;

            // Extract the table index from the leftmost bits.
            let i = r >> (T::GenInt::BITS - N::BITS);

            // Test for the common case (point below yinf).
            let d = &self.data[i.as_usize()];
            if u < d.scaled_yratio {
                return d.x + d.scaled_dx * T::cast_gen_int(u);
            }

            // Wedge sampling, test y<f(x).
            let x0 = d.x;
            let x1 = self.data[i.as_usize() + 1].x;
            let x = x0 + T::gen(rng) * (x1 - x0);
            if T::cast_gen_int(u) * d.scaled_ysup < (self.func)(x) {
                return x;
            }
        }
    }
}

/// Distribution with arbitrarily-shaped PDF and rejection-sampled tail.
#[derive(Clone)]
pub struct DistAnyTailed<T: Float, F, N, E> {
    data: Vec<Datum<T>>,
    func: F,
    tail_envelope: E,
    tail_switch: T::GenInt,
    phantom_table_size: PhantomData<N>,
}

impl<T, F, N, E> DistAnyTailed<T, F, N, E>
where
    T: Float,
    F: Fn(T) -> T,
    N: ValidTableSize<T>,
    E: Envelope<T>,
{
    pub fn new<A>(func: F, table: &A, tail_envelope: E, tail_area: T) -> Self
    where
        A: Table<T, Size = N>,
    {
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

impl<T, F, N, E> Distribution<T> for DistAnyTailed<T, F, N, E>
where
    T: Float,
    F: Fn(T) -> T,
    N: TableSize,
    E: Envelope<T>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
        loop {
            let r = T::GenInt::gen(rng);

            // Extract the significand from the leftmost bits after the table index.
            let u_mask = (T::GenInt::ONE << (T::GenInt::BITS - N::BITS)) - T::GenInt::ONE;
            let u = r & u_mask;

            // Extract the table index from the leftmost bits.
            let i = r >> (T::GenInt::BITS - N::BITS);

            // Test for the common case (point below yinf).
            let d = &self.data[i.as_usize()];
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
            let x1 = self.data[i.as_usize() + 1].x;
            let x = x0 + T::gen(rng) * (x1 - x0);
            if T::cast_gen_int(u) * d.scaled_ysup < (self.func)(x) {
                return x;
            }
        }
    }
}

/// Distribution with symmetric PDF about the origin.
#[derive(Clone)]
pub struct DistCentral<T: Float, F, N> {
    data: Vec<Datum<T>>,
    func: F,
    phantom_table_size: PhantomData<N>,
}

impl<T, F, N> DistCentral<T, F, N>
where
    T: Float,
    F: Fn(T) -> T,
    N: ValidSymmetricTableSize<T>,
{
    pub fn new<A>(func: F, table: &A) -> Self
    where
        A: Table<T, Size = N>,
    {
        let tail_switch = T::GenInt::ONE << (T::GenInt::BITS - A::Size::BITS - 1);

        DistCentral {
            data: process_table(T::ZERO, table, tail_switch),
            func: func,
            phantom_table_size: PhantomData,
        }
    }
}

impl<T, F, N> Distribution<T> for DistCentral<T, F, N>
where
    T: Float,
    F: Fn(T) -> T,
    N: TableSize,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
        loop {
            let r = T::GenInt::gen(rng);

            // Extract the significand from the leftmost bits after the table index.
            let u_mask = (T::GenInt::ONE << (T::GenInt::BITS - N::BITS - 1)) - T::GenInt::ONE;
            let u = r & u_mask;

            // Extract the table index from the N leftmost bits after the sign bit.
            let i_mask = (T::GenInt::ONE << N::BITS) - T::GenInt::ONE;
            let i = (r >> (T::GenInt::BITS - N::BITS - 1)) & i_mask;

            // Extract the sign from the leftmost bit.
            let s = if (r >> (T::GenInt::BITS - 1)) != T::GenInt::ZERO {
                T::ONE
            } else {
                -T::ONE
            };

            // Test for the common case (point below yinf).
            let d = &self.data[i.as_usize()];
            if u < d.scaled_yratio {
                return s * (d.x + d.scaled_dx * T::cast_gen_int(u));
            }

            // Wedge sampling, test y<f(x).
            let x0 = d.x;
            let x1 = self.data[i.as_usize() + 1].x;
            let x = x0 + T::gen(rng) * (x1 - x0); // here x is relative to the origin
            if T::cast_gen_int(u) * d.scaled_ysup < (self.func)(x) {
                return s * x;
            }
        }
    }
}

/// Distribution with symmetric PDF about the origin and rejection-sampled tail.
#[derive(Clone)]
pub struct DistCentralTailed<T: Float, F, N, E> {
    data: Vec<Datum<T>>,
    func: F,
    tail_envelope: E,
    tail_switch: T::GenInt,
    phantom_table_size: PhantomData<N>,
}

impl<T, F, N, E> DistCentralTailed<T, F, N, E>
where
    T: Float,
    F: Fn(T) -> T,
    N: ValidSymmetricTableSize<T>,
    E: Envelope<T>,
{
    pub fn new<A>(func: F, table: &A, tail_envelope: E, tail_area: T) -> Self
    where
        A: Table<T, Size = N>,
    {
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

impl<T, F, N, E> Distribution<T> for DistCentralTailed<T, F, N, E>
where
    T: Float,
    F: Fn(T) -> T,
    N: TableSize,
    E: Envelope<T>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
        loop {
            let r = T::GenInt::gen(rng);

            // Extract the significand from the leftmost bits after the table index.
            let u_mask = (T::GenInt::ONE << (T::GenInt::BITS - N::BITS - 1)) - T::GenInt::ONE;
            let u = r & u_mask;

            // Extract the table index from the N leftmost bits after the sign bit.
            let i_mask = (T::GenInt::ONE << N::BITS) - T::GenInt::ONE;
            let i = (r >> (T::GenInt::BITS - N::BITS - 1)) & i_mask;

            // Extract the sign from the leftmost bit.
            let s = if (r >> (T::GenInt::BITS - 1)) != T::GenInt::ZERO {
                T::ONE
            } else {
                -T::ONE
            };

            // Test for the common case (point below yinf).
            let d = &self.data[i.as_usize()];
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
            let x1 = self.data[i.as_usize() + 1].x;
            let x = x0 + T::gen(rng) * (x1 - x0); // here x is relative to the origin
            if T::cast_gen_int(u) * d.scaled_ysup < (self.func)(x) {
                return s * x;
            }
        }
    }
}

/// Distribution with symmetric PDF.
#[derive(Clone)]
pub struct DistSymmetric<T: Float, F, N> {
    data: Vec<Datum<T>>,
    func: F,
    x0: T,
    phantom_table_size: PhantomData<N>,
}

impl<T, F, N> DistSymmetric<T, F, N>
where
    T: Float,
    F: Fn(T) -> T,
    N: ValidSymmetricTableSize<T>,
{
    pub fn new<A>(x0: T, func: F, table: &A) -> Self
    where
        A: Table<T, Size = N>,
    {
        let tail_switch = T::GenInt::ONE << (T::GenInt::BITS - A::Size::BITS - 1);

        DistSymmetric {
            data: process_table(x0, table, tail_switch),
            func: func,
            x0: x0,
            phantom_table_size: PhantomData,
        }
    }
}

impl<T, F, N> Distribution<T> for DistSymmetric<T, F, N>
where
    T: Float,
    F: Fn(T) -> T,
    N: TableSize,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
        loop {
            let r = T::GenInt::gen(rng);

            // Extract the significand from the leftmost bits after the table index.
            let u_mask = (T::GenInt::ONE << (T::GenInt::BITS - N::BITS - 1)) - T::GenInt::ONE;
            let u = r & u_mask;

            // Extract the table index from the N leftmost bits after the sign bit.
            let i_mask = (T::GenInt::ONE << N::BITS) - T::GenInt::ONE;
            let i = (r >> (T::GenInt::BITS - N::BITS - 1)) & i_mask;

            // Extract the sign from the leftmost bit.
            let s = if (r >> (T::GenInt::BITS - 1)) != T::GenInt::ZERO {
                T::ONE
            } else {
                -T::ONE
            };

            // Test for the common case (point below yinf).
            let d = &self.data[i.as_usize()];
            if u < d.scaled_yratio {
                return self.x0 + s * (d.x + d.scaled_dx * T::cast_gen_int(u));
            }

            // Wedge sampling, test y<f(x).
            let x0 = d.x;
            let x1 = self.data[i.as_usize() + 1].x;
            let x = x0 + T::gen(rng) * (x1 - x0); // here x is relative to the origin
            if T::cast_gen_int(u) * d.scaled_ysup < (self.func)(x + self.x0) {
                return self.x0 + s * x;
            }
        }
    }
}

/// Distribution with symmetric PDF and rejection-sampled tail.
#[derive(Clone)]
pub struct DistSymmetricTailed<T: Float, F, N, E> {
    data: Vec<Datum<T>>,
    func: F,
    x0: T,
    tail_envelope: E,
    tail_switch: T::GenInt,
    phantom_table_size: PhantomData<N>,
}

impl<T, F, N, E> DistSymmetricTailed<T, F, N, E>
where
    T: Float,
    F: Fn(T) -> T,
    N: ValidSymmetricTableSize<T>,
    E: Envelope<T>,
{
    pub fn new<A>(x0: T, func: F, table: &A, tail_envelope: E, tail_area: T) -> Self
    where
        A: Table<T, Size = N>,
    {
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

impl<T, F, N, E> Distribution<T> for DistSymmetricTailed<T, F, N, E>
where
    T: Float,
    F: Fn(T) -> T,
    N: TableSize,
    E: Envelope<T>,
{
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> T {
        loop {
            let r = T::GenInt::gen(rng);

            // Extract the significand from the leftmost bits after the table index.
            let u_mask = (T::GenInt::ONE << (T::GenInt::BITS - N::BITS - 1)) - T::GenInt::ONE;
            let u = r & u_mask;

            // Extract the table index from the N leftmost bits after the sign bit.
            let i_mask = (T::GenInt::ONE << N::BITS) - T::GenInt::ONE;
            let i = (r >> (T::GenInt::BITS - N::BITS - 1)) & i_mask;

            // Extract the sign from the leftmost bit.
            let s = if (r >> (T::GenInt::BITS - 1)) != T::GenInt::ZERO {
                T::ONE
            } else {
                -T::ONE
            };

            // Test for the common case (point below yinf).
            let d = &self.data[i.as_usize()];
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
            let x1 = self.data[i.as_usize() + 1].x;
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
fn process_table<T, A>(x0: T, table: &A, tail_switch: T::GenInt) -> Vec<Datum<T>>
where
    T: Float,
    A: Table<T>,
{
    let one_half = T::ONE / (T::ONE + T::ONE);
    let n: usize = 1 << A::Size::BITS;
    let mut data = Vec::with_capacity(n + 1);

    // Convenient alias.
    let table = table.as_view();

    // Compute the final table.
    for i in 0..n {
        let yratio = table.yinf[i] / table.ysup[i];
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
        let scaled_ysup = table.ysup[i] / T::cast_gen_int(tail_switch);
        // dx is scaled such that, once multiplied by a random number less than the
        // critical wedge sampling threshold, its value will be in [0:dx].
        let scaled_dx = (table.x[i + 1] - table.x[i]) / T::cast_gen_int(scaled_yratio);

        data.push(Datum {
            x: table.x[i] - x0,
            scaled_dx: scaled_dx,
            scaled_ysup: scaled_ysup,
            scaled_yratio: scaled_yratio,
        });
    }

    // Last datum is dummy except for the x value.
    data.push(Datum {
        x: table.x[n] - x0,
        scaled_dx: T::ZERO,
        scaled_ysup: T::ZERO,
        scaled_yratio: T::GenInt::ZERO,
    });

    data
}

// Computes the integer used as a threshold for tail sampling.
fn compute_tail_switch<T, A>(table: &A, tail_area: T, sign_bits: u32) -> T::GenInt
where
    T: Float,
    A: Table<T>,
{
    let n: usize = 1 << A::Size::BITS;
    let max_switch = T::GenInt::ONE << (T::GenInt::BITS - A::Size::BITS - sign_bits);

    let table = table.as_view();
    let mut area = T::ZERO;
    for i in 0..n {
        area = area + (table.x[i + 1] - table.x[i]) * table.ysup[i];
    }
    let switch = T::cast_gen_int(max_switch) * (area / (area + tail_area));

    switch.round_as_gen_int()
}
