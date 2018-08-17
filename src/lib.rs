extern crate rand;

// Re-exports.
pub use tables::*;

// Internal trails.
use num_traits::*;

// External traits.
use rand::{Rng, RngCore};
use std::marker::PhantomData;

// Modules.
mod num_traits;
mod tables;
mod util;

/// Shape marker trait for distributions
pub trait Shape: Copy + Clone {
    const SIGN_BITS: u32;
}

/// Parameters for arbitrarily shaped distributions.
#[derive(Copy, Clone)]
pub struct ShapeAny;

/// Parameters for symmetric distributions.
#[derive(Copy, Clone)]
pub struct ShapeSymmetric<T: Float> {
    x0: T,
}

/// Parameters for distributions that are symmetric about the origin.
#[derive(Copy, Clone)]
pub struct ShapeCentral;

impl Shape for ShapeAny {
    const SIGN_BITS: u32 = 0;
}
impl<T: Float> Shape for ShapeSymmetric<T> {
    const SIGN_BITS: u32 = 1; // one bit required to resolve the locations WRT x0
}
impl Shape for ShapeCentral {
    const SIGN_BITS: u32 = 1; // one bit required to resolve the location WRT to 0
}

/// Distribution table datum.
#[derive(Copy, Clone)]
struct Datum<T: Float> {
    x: T,
    scaled_dx: T,             // dx/scaled_yratio
    scaled_ysup: T,           // ysup/ext_switch
    scaled_yratio: T::GenInt, // (yinf/ysup)*ext_switch
}

/// Distribution.
#[derive(Clone)]
pub struct Dist<T: Float, F: Fn(T) -> T, N: TableSize, S: Shape, E: Extension<T>> {
    data: Vec<Datum<T>>,
    func: F,
    ext: E,
    shape: S,
    phantom_table_size: PhantomData<N>,
}

impl<T: Float, F: Fn(T) -> T, N: TableSize, E: Extension<T>> Dist<T, F, N, ShapeAny, E> {
    pub fn sample<R: Rng>(&self, rng: &mut R) -> T {
        loop {
            let r = T::GenInt::gen(rng);

            // Extract the significand from the leftmost bits after the table index.
            let mask = (T::GenInt::ONE << (T::GenInt::BITS - N::BITS)) - T::GenInt::ONE;
            let u = r & mask;

            // Extract the table index from the leftmost bits.
            let i = r >> (T::GenInt::BITS - N::BITS);

            // Note that the following test will also fail if 'u' is greater or
            // equal to the outer switch value since all 'fratio' values are
            // lower than the switch value.
            let d = &self.data[i.as_usize()];
            if u < d.scaled_yratio {
                return d.x + d.scaled_dx * T::cast_gen_int(u);
            }
            
            if let Some(switch) = self.ext.switch() {
                if u >= switch {
                    if let Some(x) = self.ext.sample(rng) {
                        return x;
                    }
                }
            }

            // Otherwise it is a wedge, test y<f(x) (rejection sampling).
            let x0 = d.x;
            let x1 = self.data[i.as_usize() + 1].x;
            let x = x0 + T::gen(rng)*(x1 - x0);
            if T::cast_gen_int(u)*d.scaled_ysup < (self.func)(x) {
                return x;
            }
        }
    }
}

/// Distribution builder.
pub struct DistBuilder<'a, T: Float, F: Fn(T) -> T, A: Table<T> + 'a, S: Shape> {
    func: F,
    table: &'a A,
    shape: S,
    phantom_float: PhantomData<T>,
}

impl<'a, T: Float, F: Fn(T) -> T, A: Table<T> + 'a> DistBuilder<'a, T, F, A, ShapeAny> {
    pub fn new(func: F, table: &'a A) -> Self
    where
        A::Size: ValidTableSize<T>,
    {
        DistBuilder {
            func: func,
            table: table,
            shape: ShapeAny {},
            phantom_float: PhantomData,
        }
    }
}

impl<'a, T: Float, F: Fn(T) -> T, A: Table<T> + 'a> DistBuilder<'a, T, F, A, ShapeSymmetric<T>> {
    pub fn new_symmetric(func: F, table: &'a A, x0: T) -> Self
    where
        A::Size: ValidSymmetricTableSize<T>,
    {
        DistBuilder {
            func: func,
            table: table,
            shape: ShapeSymmetric { x0 },
            phantom_float: PhantomData,
        }
    }
}

impl<'a, T: Float, F: Fn(T) -> T, A: Table<T> + 'a> DistBuilder<'a, T, F, A, ShapeCentral> {
    pub fn new_central(func: F, table: &'a A) -> Self
    where
        A::Size: ValidSymmetricTableSize<T>,
    {
        DistBuilder {
            func: func,
            table: table,
            shape: ShapeCentral {},
            phantom_float: PhantomData,
        }
    }
}

impl<'a, T: Float, F: Fn(T) -> T, A: Table<T> + 'a, S: Shape> DistBuilder<'a, T, F, A, S> {
    pub fn standalone(self) -> Dist<T, F, A::Size, S, ExtensionNone<T>> {
        let one_half = T::ONE / (T::ONE + T::ONE);
        let max_int = T::GenInt::ONE << (T::GenInt::BITS - A::Size::BITS - S::SIGN_BITS);

        let n: usize = 1 << A::Size::BITS;
        let mut data = Vec::with_capacity(n + 1);

        // Convenient aliases.
        let x = self.table.x();
        let yinf = self.table.yinf();
        let ysup = self.table.ysup();

        // Compute the final table.
        for i in 0..n {
            let yratio = yinf[i] / ysup[i];
            let scaled_yratio = if yratio >= one_half {
                // Use baseline algorithm.
                (yratio * T::cast_gen_int(max_int)).as_gen_int()
            } else {
                // Because the random number is mapped to [0:ysup], if yinf < 0.5*ysup then
                // more than 1 bit of accuracy will be lost after a random number in [0:ysup]
                // is narrowed down to [0:yinf] and is reused as a random number within
                // this latter interval.
                // To prevent this loss of accuracy, wedge sampling is forced by setting yinf=0.
                T::GenInt::ZERO
            };
            // ysup is scaled such that, once multiplied by an integer random number, its value
            // will be in [0:y_sup)
            let scaled_ysup = ysup[i] / T::cast_gen_int(max_int);
            let scaled_dx = (x[i + 1] - x[i]) / T::cast_gen_int(scaled_yratio);

            data.push(Datum {
                x: x[i],
                scaled_dx: scaled_dx,
                scaled_ysup: scaled_ysup,
                scaled_yratio: scaled_yratio,
            });
        }

        // Last datum is dummy except for the x value.
        data.push(Datum {
            x: x[n],
            scaled_dx: T::ZERO,
            scaled_ysup: T::ZERO,
            scaled_yratio: T::GenInt::ZERO,
        });

        Dist {
            data: data,
            func: self.func,
            shape: self.shape,
            ext: ExtensionNone {
                phantom_float: PhantomData,
            },
            phantom_table_size: PhantomData,
        }
    }

    pub fn extended<E: Fn(&mut RngCore) -> T>(
        self,
        ext_dist: E,
        ext_weight: T,
    ) -> Dist<T, F, A::Size, S, ExtensionSome<T, E>> {
        Dist {
            data: Vec::new(),
            func: self.func,
            shape: self.shape,
            ext: ExtensionSome {
                switch: T::GenInt::ONE,
                dist: ext_dist,
                phantom_float: PhantomData,
            },
            phantom_table_size: PhantomData,
        }
    }

    pub fn sup_extended<E: Fn(&mut RngCore) -> Option<T>>(
        self,
        sup_ext_dist: E,
        sup_ext_weight: T,
    ) -> Dist<T, F, A::Size, S, ExtensionOption<T, E>> {
        Dist {
            data: Vec::new(),
            func: self.func,
            shape: self.shape,
            ext: ExtensionOption {
                switch: T::GenInt::ONE,
                dist: sup_ext_dist,
                phantom_float: PhantomData,
            },
            phantom_table_size: PhantomData,
        }
    }
}

/// Distribution extension trait.
pub trait Extension<T: Float> {
    fn switch(&self) -> Option<T::GenInt>;
    fn sample<R: RngCore>(&self, rng: &mut R) -> Option<T>;
}

#[derive(Copy, Clone)]
pub struct ExtensionNone<T: Float> {
    phantom_float: PhantomData<T>,
}

impl<T: Float> Extension<T> for ExtensionNone<T> {
    fn switch(&self) -> Option<T::GenInt> {
        return None;
    }
    fn sample<R: RngCore>(&self, _rng: &mut R) -> Option<T> {
        return None;
    }
}

#[derive(Copy, Clone)]
pub struct ExtensionSome<T: Float, F: Fn(&mut RngCore) -> T> {
    switch: T::GenInt,
    dist: F,
    phantom_float: PhantomData<T>,
}

impl<T: Float, F: Fn(&mut RngCore) -> T> Extension<T> for ExtensionSome<T, F> {
    fn switch(&self) -> Option<T::GenInt> {
        return Some(self.switch);
    }
    fn sample<R: RngCore>(&self, rng: &mut R) -> Option<T> {
        return Some((self.dist)(rng));
    }
}

#[derive(Copy, Clone)]
pub struct ExtensionOption<T: Float, F: Fn(&mut RngCore) -> Option<T>> {
    switch: T::GenInt,
    dist: F,
    phantom_float: PhantomData<T>,
}

impl<T: Float, F: Fn(&mut RngCore) -> Option<T>> Extension<T> for ExtensionOption<T, F> {
    fn switch(&self) -> Option<T::GenInt> {
        return Some(self.switch);
    }
    fn sample<R: RngCore>(&self, rng: &mut R) -> Option<T> {
        return (self.dist)(rng);
    }
}
