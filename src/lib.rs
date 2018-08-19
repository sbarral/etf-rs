extern crate rand;

// Re-exports.
pub use num_traits::Float;
pub use tables::*;

use num_traits::Int;

// External traits.
use rand::{Rng, RngCore};
use std::marker::PhantomData;

// Modules.
mod num_traits;
mod tables;
mod util;

/// Marker trait for distribution shape.
pub trait Shape<T: Float>: Copy + Clone {
    #[doc(hidden)]
    fn origin(&self) -> T {
        T::ZERO
    }
}

/// Properties of an arbitrarily shaped distribution.
#[derive(Copy, Clone)]
pub struct ShapeAny;

/// Properties of a distribution that is symmetric about the origin.
#[derive(Copy, Clone)]
pub struct ShapeCentral;

/// Properties of a general symmetric distribution.
#[derive(Copy, Clone)]
pub struct ShapeSymmetric<T: Float> {
    x0: T,
}

impl<T: Float> Shape<T> for ShapeAny {}
impl<T: Float> Shape<T> for ShapeCentral {}
impl<T: Float> Shape<T> for ShapeSymmetric<T> {
    #[doc(hidden)]
    fn origin(&self) -> T {
        self.x0
    }
}

/// Distribution table datum.
#[derive(Copy, Clone)]
struct Datum<T: Float> {
    x: T,
    scaled_dx: T,             // dx/scaled_yratio
    scaled_ysup: T,           // ysup/tail_switch
    scaled_yratio: T::GenInt, // (yinf/ysup)*tail_switch
}

fn process_table<S, T, A>(shape: S, table: &A, tail_switch: T::GenInt) -> Vec<Datum<T>>
where
    S: Shape<T>,
    T: Float,
    A: Table<T>,
{
    let one_half = T::ONE / (T::ONE + T::ONE);
    let n: usize = 1 << A::Size::BITS;
    let mut data = Vec::with_capacity(n + 1);

    // Convenient aliases.
    let x = table.x();
    let yinf = table.yinf();
    let ysup = table.ysup();

    // Compute the final table.
    for i in 0..n {
        let yratio = yinf[i] / ysup[i];
        let scaled_yratio = if yratio >= one_half {
            // Use baseline algorithm.
            (yratio * T::cast_gen_int(tail_switch)).round_as_gen_int()
        } else {
            // Because the random number is mapped to [0:ysup], if yinf < 0.5*ysup then
            // more than 1 bit of accuracy will be lost after a random number in [0:ysup]
            // is narrowed down to [0:yinf] and is reused as a random number within
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
            x: x[i] - shape.origin(),
            scaled_dx: scaled_dx,
            scaled_ysup: scaled_ysup,
            scaled_yratio: scaled_yratio,
        });
    }

    // Last datum is dummy except for the x value.
    data.push(Datum {
        x: x[n] - shape.origin(),
        scaled_dx: T::ZERO,
        scaled_ysup: T::ZERO,
        scaled_yratio: T::GenInt::ZERO,
    });

    data
}

fn compute_tail_switch<T, A>(table: &A, tail_area: T, sign_bits: u32) -> T::GenInt
where
    T: Float,
    A: Table<T>,
{
    let n: usize = 1 << A::Size::BITS;
    let max_switch = T::GenInt::ONE << (T::GenInt::BITS - A::Size::BITS - sign_bits);

    // Convenient aliases.
    let x = table.x();
    let ysup = table.ysup();

    let mut area = T::ZERO;
    for i in 0..n {
        area = area + (x[i + 1] - x[i]) * ysup[i];
    }
    let switch = T::cast_gen_int(max_switch) * (area / (area + tail_area));

    switch.round_as_gen_int()
}

/// Distribution.
#[derive(Clone)]
pub struct Dist<T, F, N, S, G>
where
    T: Float,
    F: Fn(T) -> T,
    N: TableSize,
    S: Shape<T>,
    G: Tail<T>,
{
    data: Vec<Datum<T>>,
    func: F,
    tail: G,
    shape: S,
    phantom_table_size: PhantomData<N>,
}

impl<T, F, N, G> Dist<T, F, N, ShapeAny, G>
where
    T: Float,
    F: Fn(T) -> T,
    N: TableSize,
    G: Tail<T>,
{
    pub fn sample<R: Rng>(&self, rng: &mut R) -> T {
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
            if let Some(switch) = self.tail.switch() {
                if u >= switch {
                    if let Some(x) = self.tail.sample(rng) {
                        return x;
                    }
                }
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

impl<T, F, N, G> Dist<T, F, N, ShapeCentral, G>
where
    T: Float,
    F: Fn(T) -> T,
    N: TableSize,
    G: Tail<T>,
{
    pub fn sample<R: Rng>(&self, rng: &mut R) -> T {
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
                T::MINUS_ONE
            };

            // Test for the common case (point below yinf).
            let d = &self.data[i.as_usize()];
            if u < d.scaled_yratio {
                return s * (d.x + d.scaled_dx * T::cast_gen_int(u));
            }

            // Check if the tail should be sampled.
            if let Some(switch) = self.tail.switch() {
                if u >= switch {
                    if let Some(x) = self.tail.sample(rng) {
                        return s * x;
                    }
                }
            }

            // Wedge sampling, test y<f(x).
            let x0 = d.x;
            let x1 = self.data[i.as_usize() + 1].x;
            let x = x0 + T::gen(rng) * (x1 - x0);
            if T::cast_gen_int(u) * d.scaled_ysup < (self.func)(x) {
                return s * x;
            }
        }
    }
}

impl<T, F, N, G> Dist<T, F, N, ShapeSymmetric<T>, G>
where
    T: Float,
    F: Fn(T) -> T,
    N: TableSize,
    G: Tail<T>,
{
    pub fn sample<R: Rng>(&self, rng: &mut R) -> T {
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
                T::MINUS_ONE
            };

            // Test for the common case (point below yinf).
            let d = &self.data[i.as_usize()];
            if u < d.scaled_yratio {
                return self.shape.x0 + s * (d.x + d.scaled_dx * T::cast_gen_int(u));
            }

            // Check if the tail should be sampled.
            if let Some(switch) = self.tail.switch() {
                if u >= switch {
                    if let Some(x) = self.tail.sample(rng) {
                        return self.shape.x0 + s * (x - self.shape.x0);
                    }
                }
            }

            // Wedge sampling, test y<f(x).
            let x0 = d.x;
            let x1 = self.data[i.as_usize() + 1].x;
            let x = x0 + T::gen(rng) * (x1 - x0); // here x is relative to the origin
            if T::cast_gen_int(u) * d.scaled_ysup < (self.func)(x + self.shape.x0) {
                return self.shape.x0 + s * x;
            }
        }
    }
}

impl<T, F, N> Dist<T, F, N, ShapeAny, TailNone>
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
        let shape = ShapeAny {};

        Dist {
            data: process_table(shape, table, tail_switch),
            func: func,
            tail: TailNone {},
            shape: shape,
            phantom_table_size: PhantomData,
        }
    }
}

impl<T, F, N> Dist<T, F, N, ShapeCentral, TailNone>
where
    T: Float,
    F: Fn(T) -> T,
    N: ValidSymmetricTableSize<T>,
{
    pub fn new_central<A>(func: F, table: &A) -> Self
    where
        A: Table<T, Size = N>,
    {
        let tail_switch = T::GenInt::ONE << (T::GenInt::BITS - A::Size::BITS - 1);
        let shape = ShapeCentral {};

        Dist {
            data: process_table(shape, table, tail_switch),
            func: func,
            tail: TailNone {},
            shape: shape,
            phantom_table_size: PhantomData,
        }
    }
}

impl<T, F, N> Dist<T, F, N, ShapeSymmetric<T>, TailNone>
where
    T: Float,
    F: Fn(T) -> T,
    N: ValidSymmetricTableSize<T>,
{
    pub fn new_symmetric<A>(x0: T, func: F, table: &A) -> Self
    where
        A: Table<T, Size = N>,
    {
        let tail_switch = T::GenInt::ONE << (T::GenInt::BITS - A::Size::BITS - 1);
        let shape = ShapeSymmetric { x0: x0 };

        Dist {
            data: process_table(shape, table, tail_switch),
            func: func,
            tail: TailNone {},
            shape: shape,
            phantom_table_size: PhantomData,
        }
    }
}

impl<T, F, N, G> Dist<T, F, N, ShapeAny, TailSome<T, G>>
where
    T: Float,
    F: Fn(T) -> T,
    N: ValidTableSize<T>,
    G: Fn(&mut RngCore) -> Option<T>,
{
    pub fn new_tailed<A>(func: F, table: &A, tail_func: G, tail_area: T) -> Self
    where
        A: Table<T, Size = N>,
    {
        let tail_switch = compute_tail_switch(table, tail_area, 0);
        let shape = ShapeAny {};

        Dist {
            data: process_table(shape, table, tail_switch),
            func: func,
            tail: TailSome {
                func: tail_func,
                switch: tail_switch,
            },
            shape: shape,
            phantom_table_size: PhantomData,
        }
    }
}

impl<T, F, N, G> Dist<T, F, N, ShapeCentral, TailSome<T, G>>
where
    T: Float,
    F: Fn(T) -> T,
    N: ValidSymmetricTableSize<T>,
    G: Fn(&mut RngCore) -> Option<T>,
{
    pub fn new_central_tailed<A>(func: F, table: &A, tail_func: G, tail_area: T) -> Self
    where
        A: Table<T, Size = N>,
    {
        let tail_switch = compute_tail_switch(table, tail_area, 1);
        let shape = ShapeCentral {};

        Dist {
            data: process_table(shape, table, tail_switch),
            func: func,
            tail: TailSome {
                func: tail_func,
                switch: tail_switch,
            },
            shape: shape,
            phantom_table_size: PhantomData,
        }
    }
}

impl<T, F, N, G> Dist<T, F, N, ShapeSymmetric<T>, TailSome<T, G>>
where
    T: Float,
    F: Fn(T) -> T,
    N: ValidSymmetricTableSize<T>,
    G: Fn(&mut RngCore) -> Option<T>,
{
    pub fn new_symmetric_tailed<A>(x0: T, func: F, table: &A, tail_func: G, tail_area: T) -> Self
    where
        A: Table<T, Size = N>,
    {
        let tail_switch = compute_tail_switch(table, tail_area, 1);
        let shape = ShapeSymmetric { x0: x0 };

        Dist {
            data: process_table(shape, table, tail_switch),
            func: func,
            tail: TailSome {
                func: tail_func,
                switch: tail_switch,
            },
            shape: shape,
            phantom_table_size: PhantomData,
        }
    }
}

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
    func: G,
    switch: T::GenInt,
}

impl<T: Float, G: Fn(&mut RngCore) -> Option<T>> Tail<T> for TailSome<T, G> {
    fn switch(&self) -> Option<T::GenInt> {
        return Some(self.switch);
    }
    fn sample<R: RngCore>(&self, rng: &mut R) -> Option<T> {
        return (self.func)(rng);
    }
}
