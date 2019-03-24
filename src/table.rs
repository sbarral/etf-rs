use crate::num::Float;

/// Marker trait for table size.
pub trait TableSize: Copy + Clone {
    const BITS: u32;
}

/// Partition trait.
pub trait Partition<T: Float>: Copy + Clone + Default {
    /// Slice view.
    fn as_slice(&self) -> &[T];
    /// Mutable slice view.
    fn as_mut_slice(&mut self) -> &mut [T];
}

/// Quadrature table view.
pub struct TableView<'a, T: Float + 'a> {
    pub x: &'a [T],
    pub yinf: &'a [T],
    pub ysup: &'a [T],
}

/// Quadrature table mutable view.
pub struct TableMutView<'a, T: Float + 'a> {
    pub x: &'a mut [T],
    pub yinf: &'a mut [T],
    pub ysup: &'a mut [T],
}

/// Quadrature table trait.
pub trait Table<T: Float>: Copy + Clone + Default {
    type Size: TableSize;
    type Partition: Partition<T>;

    fn as_view(&self) -> TableView<T>;
    fn as_mut_view(&mut self) -> TableMutView<T>;
}

macro_rules! generate_tables {
    ($s:expr, $ts:ident, $p:ident, $t:ident) => {
        /// Table size marker.
        #[derive(Copy, Clone)]
        pub struct $ts;

        impl TableSize for $ts {
            const BITS: u32 = $s;
        }

        #[derive(Copy, Clone)]
        pub struct $p<T: Float> {
            x: [T; (1 << $s) + 1],
        }

        impl<T: Float> Default for $p<T> {
            fn default() -> Self {
                $p {
                    x: [T::ZERO; (1 << $s) + 1],
                }
            }
        }

        impl<T: Float> Partition<T> for $p<T> {
            fn as_slice(&self) -> &[T] {
                &self.x
            }
            fn as_mut_slice(&mut self) -> &mut [T] {
                &mut self.x
            }
        }

        /// Quadrature table.
        #[derive(Copy, Clone)]
        pub struct $t<T: Float> {
            pub x: [T; (1 << $s) + 1],
            pub yinf: [T; 1 << $s],
            pub ysup: [T; 1 << $s],
        }

        impl<T: Float> Default for $t<T> {
            fn default() -> Self {
                Self {
                    x: [T::ZERO; (1 << $s) + 1],
                    yinf: [T::ZERO; 1 << $s],
                    ysup: [T::ZERO; 1 << $s],
                }
            }
        }

        impl<T: Float> Table<T> for $t<T> {
            type Size = $ts;
            type Partition = $p<T>;

            fn as_view(&self) -> TableView<T> {
                TableView {
                    x: &self.x,
                    yinf: &self.yinf,
                    ysup: &self.ysup,
                }
            }
            fn as_mut_view(&mut self) -> TableMutView<T> {
                TableMutView {
                    x: &mut self.x,
                    yinf: &mut self.yinf,
                    ysup: &mut self.ysup,
                }
            }
        }
    };
}

generate_tables!(4, TableSize16, Partition16, Table16);
generate_tables!(5, TableSize32, Partition32, Table32);
generate_tables!(6, TableSize64, Partition64, Table64);
generate_tables!(7, TableSize128, Partition128, Table128);
generate_tables!(8, TableSize256, Partition256, Table256);
generate_tables!(9, TableSize512, Partition512, Table512);
generate_tables!(10, TableSize1024, Partition1024, Table1024);
generate_tables!(11, TableSize2048, Partition2048, Table2048);
generate_tables!(12, TableSize4096, Partition4096, Table4096);

/// Valid table size marker trait.
pub trait ValidTableSize<T: Float>: TableSize {}

/// Valid symmetric table size marker trait.
pub trait ValidSymmetricTableSize<T: Float>: TableSize {}

// Valid tables sizes for f32.
impl ValidTableSize<f32> for TableSize16 {}
impl ValidTableSize<f32> for TableSize32 {}
impl ValidTableSize<f32> for TableSize64 {}
impl ValidTableSize<f32> for TableSize128 {}
impl ValidTableSize<f32> for TableSize256 {}

// Valid tables sizes for f64.
impl ValidTableSize<f64> for TableSize16 {}
impl ValidTableSize<f64> for TableSize32 {}
impl ValidTableSize<f64> for TableSize64 {}
impl ValidTableSize<f64> for TableSize128 {}
impl ValidTableSize<f64> for TableSize256 {}
impl ValidTableSize<f64> for TableSize512 {}
impl ValidTableSize<f64> for TableSize1024 {}
impl ValidTableSize<f64> for TableSize2048 {}
impl ValidTableSize<f64> for TableSize4096 {}

// Valid tables sizes for f32 (symmetric distributions).
impl ValidSymmetricTableSize<f32> for TableSize16 {}
impl ValidSymmetricTableSize<f32> for TableSize32 {}
impl ValidSymmetricTableSize<f32> for TableSize64 {}
impl ValidSymmetricTableSize<f32> for TableSize128 {}

// Valid tables sizes for f64 (symmetric distributions).
impl ValidSymmetricTableSize<f64> for TableSize16 {}
impl ValidSymmetricTableSize<f64> for TableSize32 {}
impl ValidSymmetricTableSize<f64> for TableSize64 {}
impl ValidSymmetricTableSize<f64> for TableSize128 {}
impl ValidSymmetricTableSize<f64> for TableSize256 {}
impl ValidSymmetricTableSize<f64> for TableSize512 {}
impl ValidSymmetricTableSize<f64> for TableSize1024 {}
impl ValidSymmetricTableSize<f64> for TableSize2048 {}
