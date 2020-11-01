//! Interval partitions.

use crate::num::Float;

use std::convert::TryInto;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};


/// Tabulation datum type (internal use only).
#[repr(C)]
#[derive(Copy, Clone, Default, Debug)]
pub struct Datum<T: Float> {
    pub(crate) alpha: T,              // (x[i+1] - x[i]) / wedge_switch[i]
    pub(crate) beta: T,               // x[i] - x0
    pub(crate) wedge_switch: T::UInt, // (yinf / ysup) * tail_switch
}

/// Fixed-size partition of an interval into subintervals.
pub trait Partition<T: Float> {
    const BITS: u32;
    const SIZE: usize;
    
    type NodeArray: Clone + Default + Index<usize, Output=T> + IndexMut<usize>;
    type IntervalArray: Clone + Default + Index<usize, Output=T> + IndexMut<usize>;
    type DataArray: Clone + Default + Index<usize, Output=Datum<T>> + IndexMut<usize>;
}

macro_rules! setup_partition {
    ($p:ident, $ia:ident, $na:ident, $da:ident, $sz:expr, $szs:expr, $szsd:expr, $bits:expr) => {
        #[doc = "A partition with"]
        #[doc = $szs]
        #[doc = "subintervals."]
        #[derive(Clone, Default)]
        pub struct $p<T> {
            _phantom: PhantomData<T>
        }
        
        impl<T: Float> Partition<T> for $p<T> {
            const BITS: u32 = $bits;
            const SIZE: usize = $sz;
            
            type IntervalArray = $ia<T>;
            type NodeArray = $na<T>;
            type DataArray = $da<T>;
        }
        
        #[doc = "Value array of size"]
        #[doc = $szsd]
        #[derive(Clone)]
        pub struct $ia<T>(Box<[T; $sz]>);

        impl<T> Index<usize> for $ia<T> {
            type Output = T;
            fn index(&self, index: usize) -> &T {
                &self.0[index]
            }
        }

        impl<T> IndexMut<usize> for $ia<T> {
            fn index_mut(&mut self, index: usize) -> &mut T {
                &mut self.0[index]
            }
        }
        
        impl<T: Default + Copy + Debug> Default for $ia<T> {
            fn default() -> Self {
                // Get the boxed array from a Vec to avoid placing a temporary array on the stack.
                let mut vec = Vec::new();
                vec.resize_with($sz, Default::default);
                let boxed_slice = vec.into_boxed_slice();
                let boxed_array = boxed_slice.try_into().unwrap();
                
                Self(boxed_array)
            }
        }
        
        
        #[doc = "Value array of size"]
        #[doc = $szs]
        #[doc = "+ 1."]
        #[derive(Clone)]
        pub struct $na<T>(Box<[T; $sz + 1]>);

        impl<T> Index<usize> for $na<T> {
            type Output = T;
            fn index(&self, index: usize) -> &T {
                &self.0[index]
            }
        }

        impl<T> IndexMut<usize> for $na<T> {
            fn index_mut(&mut self, index: usize) -> &mut T {
                &mut self.0[index]
            }
        }
        
        impl<T: Default + Copy + Debug> Default for $na<T> {
            fn default() -> Self {
                // Get the boxed array from a Vec to avoid placing a temporary array on the stack.
                let mut vec = Vec::new();
                vec.resize_with($sz + 1, Default::default);
                let boxed_slice = vec.into_boxed_slice();
                let boxed_array = boxed_slice.try_into().unwrap();
                
                Self(boxed_array)
            }
        }
        
        #[doc = "Data array of size"]
        #[doc = $szs]
        #[doc = "+ 1."]
        #[derive(Clone)]
        pub struct $da<T: Float>(Box<[Datum<T>; $sz + 1]>);

        impl<T: Float> Index<usize> for $da<T> {
            type Output = Datum<T>;
            fn index(&self, index: usize) -> &Datum<T> {
                &self.0[index]
            }
        }

        impl<T: Float> IndexMut<usize> for $da<T> {
            fn index_mut(&mut self, index: usize) -> &mut Datum<T> {
                &mut self.0[index]
            }
        }
        
        impl<T: Float> Default for $da<T> {
            fn default() -> Self {
                // Get the boxed array from a Vec to avoid placing a temporary array on the stack.
                let mut vec = Vec::new();
                vec.resize_with($sz + 1, Default::default);
                let boxed_slice = vec.into_boxed_slice();
                let boxed_array = boxed_slice.try_into().unwrap();
                
                Self(boxed_array)
            }
        }
    };

    ($p:ident, $ia:ident, $na:ident, $da:ident, $sz:expr, $bits:expr) => {
        setup_partition!($p, $ia, $na, $da, $sz, stringify!($sz), concat!($sz, "."), $bits);
    }
}


setup_partition!(P16, IntervalArray16, NodeArray16, DataArray16, 16, 4);
setup_partition!(P32, IntervalArray32, NodeArray32, DataArray32, 32, 5);
setup_partition!(P64, IntervalArray64, NodeArray64, DataArray64, 64, 6);
setup_partition!(P128, IntervalArray128, NodeArray128, DataArray128, 128, 7);
setup_partition!(P256, IntervalArray256, NodeArray256, DataArray256, 256, 8);
setup_partition!(P512, IntervalArray512, NodeArray512, DataArray512, 512, 9);
setup_partition!(P1024, IntervalArray1024, NodeArray1024, DataArray1024, 1024, 10);
setup_partition!(P2048, IntervalArray2048, NodeArray2048, DataArray2048, 2048, 11);
setup_partition!(P4096, IntervalArray4096, NodeArray4096, DataArray4096, 4096, 12);

/// Dataset for ETF distribution generation.
#[derive(Clone, PartialEq, PartialOrd)]
pub struct InitTable<P: Partition<T>, T: Float> {
    pub x: P::NodeArray,
    pub yinf: P::IntervalArray,
    pub ysup: P::IntervalArray,
}

impl<P: Partition<T>, T: Float> Default for InitTable<P, T> {
    fn default() -> Self {
        Self {
            x: P::NodeArray::default(),
            yinf: P::IntervalArray::default(),
            ysup: P::IntervalArray::default(),
        
        }
    }
}