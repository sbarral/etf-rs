use crate::num::Float;

use std::borrow::{Borrow, BorrowMut};
use std::hash::Hash;
use std::marker::PhantomData;

/// Marker trait for partition size.
pub trait Partition: Copy + Clone + Default + PartialEq + PartialOrd + Eq + Ord + Hash {
    const BITS: u32;
    const SIZE: usize;
}

/// A fixed-size array with `N+1` elements where `N` is the partition size.
///
/// The array content can be accessed via the `AsRef` and `AsMut` traits.
#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub struct NodeArray<P: Partition, T: Float> {
    inner: Vec<T>,
    phantom: PhantomData<P>,
}
impl<P: Partition, T: Float> NodeArray<P, T> {
    pub fn new() -> Self {
        let mut inner = Vec::with_capacity(P::SIZE + 1);
        inner.resize_with(P::SIZE + 1, T::zero);

        Self {
            inner: inner,
            phantom: PhantomData,
        }
    }
}
impl<P: Partition, T: Float> Default for NodeArray<P, T> {
    fn default() -> Self {
        Self::new()
    }
}
impl<P: Partition, T: Float> AsRef<[T]> for NodeArray<P, T> {
    fn as_ref(&self) -> &[T] {
        &self.inner
    }
}
impl<P: Partition, T: Float> AsMut<[T]> for NodeArray<P, T> {
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.inner
    }
}
impl<P: Partition, T: Float> Borrow<[T]> for NodeArray<P, T> {
    fn borrow(&self) -> &[T] {
        &self.inner
    }
}
impl<P: Partition, T: Float> BorrowMut<[T]> for NodeArray<P, T> {
    fn borrow_mut(&mut self) -> &mut [T] {
        &mut self.inner
    }
}

/// A fixed-size array with `N` elements where `N` is the partition size.
///
/// The array content can be accessed via the `AsRef` and `AsMut` traits.
#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub struct IntervalArray<P: Partition, T: Float> {
    inner: Vec<T>,
    phantom: PhantomData<P>,
}
impl<P: Partition, T: Float> IntervalArray<P, T> {
    pub fn new() -> Self {
        let mut inner = Vec::with_capacity(P::SIZE);
        inner.resize_with(P::SIZE, T::zero);
        Self {
            inner: inner,
            phantom: PhantomData,
        }
    }
}
impl<P: Partition, T: Float> Default for IntervalArray<P, T> {
    fn default() -> Self {
        Self::new()
    }
}
impl<P: Partition, T: Float> AsRef<[T]> for IntervalArray<P, T> {
    fn as_ref(&self) -> &[T] {
        &self.inner
    }
}
impl<P: Partition, T: Float> AsMut<[T]> for IntervalArray<P, T> {
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.inner
    }
}
impl<P: Partition, T: Float> Borrow<[T]> for IntervalArray<P, T> {
    fn borrow(&self) -> &[T] {
        &self.inner
    }
}
impl<P: Partition, T: Float> BorrowMut<[T]> for IntervalArray<P, T> {
    fn borrow_mut(&mut self) -> &mut [T] {
        &mut self.inner
    }
}

/// Dataset required to build a distribution.
#[derive(Clone, Default, PartialEq, PartialOrd, Eq, Ord, Hash)]
pub struct InitTable<P: Partition, T: Float> {
    pub x: NodeArray<P, T>,
    pub yinf: IntervalArray<P, T>,
    pub ysup: IntervalArray<P, T>,
}
impl<P: Partition, T: Float> InitTable<P, T> {
    pub fn new() -> Self {
        Self {
            x: NodeArray::new(),
            yinf: IntervalArray::new(),
            ysup: IntervalArray::new(),
        }
    }
}

macro_rules! make_partition_size {
    ($s:expr, $ps:ident) => {
        /// Partition size marker.
        #[derive(Copy, Clone, Default, PartialEq, PartialOrd, Eq, Ord, Hash)]
        pub struct $ps;

        impl Partition for $ps {
            const BITS: u32 = $s;
            const SIZE: usize = 1 << $s;
        }
    };
}

make_partition_size!(4, P16);
make_partition_size!(5, P32);
make_partition_size!(6, P64);
make_partition_size!(7, P128);
make_partition_size!(8, P256);
make_partition_size!(9, P512);
make_partition_size!(10, P1024);
make_partition_size!(11, P2048);
make_partition_size!(12, P4096);
