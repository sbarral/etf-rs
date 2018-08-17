use num_traits::Float;

/// Table size trait.
pub trait TableSize: Copy + Clone {
    const BITS: u32;
}

/// Marker types for table size.
#[derive(Copy, Clone)]
pub struct TableSize128;
#[derive(Copy, Clone)]
pub struct TableSize256;

/// Table size trait implementations.
impl TableSize for TableSize128 {
    const BITS: u32 = 7;
}
impl TableSize for TableSize256 {
    const BITS: u32 = 8;
}

/// Valid table size marker trait.
pub trait ValidTableSize<T: Float> {}

/// Valid symmetric table size marker trait.
pub trait ValidSymmetricTableSize<T: Float> {}

// Valid tables sizes for f32.
impl ValidTableSize<f32> for TableSize128 {}
impl ValidTableSize<f32> for TableSize256 {}

impl ValidSymmetricTableSize<f32> for TableSize128 {}

// Valid tables sizes for f64.
impl ValidTableSize<f64> for TableSize128 {}
impl ValidTableSize<f64> for TableSize256 {}

impl ValidSymmetricTableSize<f64> for TableSize128 {}
impl ValidSymmetricTableSize<f64> for TableSize256 {}

/// Distribution partition table trait.
pub trait Table<T: Float>: Copy + Clone {
    type Size: TableSize;

    fn x(&self) -> &[T];
    fn yinf(&self) -> &[T];
    fn ysup(&self) -> &[T];
}

#[derive(Copy, Clone)]
pub struct Table128<T: Float> {
    pub x: [T; 129],
    pub yinf: [T; 128],
    pub ysup: [T; 128],
}

#[derive(Copy, Clone)]
pub struct Table256<T: Float> {
    pub x: [T; 257],
    pub yinf: [T; 256],
    pub ysup: [T; 256],
}

impl<T: Float> Table<T> for Table128<T> {
    type Size = TableSize128;

    fn x(&self) -> &[T] {
        &self.x
    }
    fn yinf(&self) -> &[T] {
        &self.yinf
    }
    fn ysup(&self) -> &[T] {
        &self.ysup
    }
}
impl<T: Float> Table<T> for Table256<T> {
    type Size = TableSize256;

    fn x(&self) -> &[T] {
        &self.x
    }
    fn yinf(&self) -> &[T] {
        &self.yinf
    }
    fn ysup(&self) -> &[T] {
        &self.ysup
    }
}
