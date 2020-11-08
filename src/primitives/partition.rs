//! Interval partitions and related data structures.

use super::storage::{Datum, Storage};
use crate::num::Float;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut};

/// Marker trait for the partition of an interval into N subintervals.
pub trait Partition<T: Float>: Clone {
    const BITS: u32;
    const SIZE: usize;
    #[doc(hidden)]
    type IntervalStorage: Storage<T>;
    #[doc(hidden)]
    type NodeStorage: Storage<T>;
    #[doc(hidden)]
    type DataStorage: Storage<Datum<T>>;
}

macro_rules! make_partition {
    ($p:ident, $sz:expr, $szs:expr, $bits:expr) => {
        #[doc = "A partition with"]
        #[doc = $szs]
        #[doc = "subintervals."]
        #[derive(Clone)]
        pub struct $p<T> {
            _phantom: PhantomData<T>
        }
        impl<T: Float> Partition<T> for $p<T> {
            const BITS: u32 = $bits;
            const SIZE: usize = $sz;
            type IntervalStorage = [T; $sz];
            type NodeStorage = [T; $sz + 1];
            type DataStorage = [Datum<T>; $sz + 1];
        }
    };

    ($p:ident, $sz:expr, $bits:expr) => {
        make_partition!($p, $sz, stringify!($sz), $bits);
    }
}
make_partition!(P16, 16, 4);
make_partition!(P32, 32, 5);
make_partition!(P64, 64, 6);
make_partition!(P128, 128, 7);
make_partition!(P256, 256, 8);
make_partition!(P512, 512, 9);
make_partition!(P1024, 1024, 10);
make_partition!(P2048, 2048, 11);
make_partition!(P4096, 4096, 12);

/// Array of N values defined over the subintervals of an N-subinterval partition.
#[derive(Clone)]
pub struct IntervalArray<P: Partition<T>, T: Float>(Box<P::IntervalStorage>);

impl<P: Partition<T>, T: Float> Default for IntervalArray<P, T> {
    fn default() -> Self {
        Self(P::IntervalStorage::allocate())
    }
}
impl<P: Partition<T>, T: Float> Index<usize> for IntervalArray<P, T> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        &(*self.0).as_ref()[index]
    }
}
impl<P: Partition<T>, T: Float> IndexMut<usize> for IntervalArray<P, T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut (*self.0).as_mut()[index]
    }
}

/// Array of N+1 values defined over the nodes of an N-subinterval partition .
#[derive(Clone)]
pub struct NodeArray<P: Partition<T>, T: Float>(Box<P::NodeStorage>);

impl<P: Partition<T>, T: Float> Default for NodeArray<P, T> {
    fn default() -> Self {
        Self(P::NodeStorage::allocate())
    }
}
impl<P: Partition<T>, T: Float> Index<usize> for NodeArray<P, T> {
    type Output = T;
    fn index(&self, index: usize) -> &T {
        &(*self.0).as_ref()[index]
    }
}
impl<P: Partition<T>, T: Float> IndexMut<usize> for NodeArray<P, T> {
    fn index_mut(&mut self, index: usize) -> &mut T {
        &mut (*self.0).as_mut()[index]
    }
}

/// Array of N+1 data defined over the nodes of an N-subinterval partition.
#[derive(Clone)]
pub(crate) struct DataArray<P: Partition<T>, T: Float>(Box<P::DataStorage>);

impl<P: Partition<T>, T: Float> Default for DataArray<P, T> {
    fn default() -> Self {
        Self(P::DataStorage::allocate())
    }
}
impl<P: Partition<T>, T: Float> Index<usize> for DataArray<P, T> {
    type Output = Datum<T>;
    fn index(&self, index: usize) -> &Datum<T> {
        &(*self.0).as_ref()[index]
    }
}
impl<P: Partition<T>, T: Float> IndexMut<usize> for DataArray<P, T> {
    fn index_mut(&mut self, index: usize) -> &mut Datum<T> {
        &mut (*self.0).as_mut()[index]
    }
}

/// ETF distribution initialization table.
#[derive(Clone)]
pub struct InitTable<P: Partition<T>, T: Float> {
    pub x: NodeArray<P, T>,
    pub yinf: IntervalArray<P, T>,
    pub ysup: IntervalArray<P, T>,
}

impl<P: Partition<T>, T: Float> Default for InitTable<P, T> {
    fn default() -> Self {
        Self {
            x: NodeArray::default(),
            yinf: IntervalArray::default(),
            ysup: IntervalArray::default(),
        }
    }
}
