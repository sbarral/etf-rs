use num_traits::Float;

/// Marker trait for table size.
pub trait TableSize: Copy + Clone {
    const BITS: u32;
}

/// Partition table trait.
pub trait Table<T: Float>: Copy + Clone {
    type Size: TableSize;

    fn x(&self) -> &[T];
    fn yinf(&self) -> &[T];
    fn ysup(&self) -> &[T];
}

macro_rules! generate_table {
    ($s:expr, $t:ident, $ts:ident) => {
        /// Table size marker.
        #[derive(Copy, Clone)]
        pub struct $ts;

        impl TableSize for $ts {
            const BITS: u32 = $s;
        }

        /// Partition table.
        #[derive(Copy, Clone)]
        pub struct $t<T: Float> {
            pub x: [T; (1 << $s) + 1],
            pub yinf: [T; 1 << $s],
            pub ysup: [T; 1 << $s],
        }

        impl<T: Float> Table<T> for $t<T> {
            type Size = $ts;

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
    };
}

generate_table!(4, Table16, TableSize16);
generate_table!(5, Table32, TableSize32);
generate_table!(6, Table64, TableSize64);
generate_table!(7, Table128, TableSize128);
generate_table!(8, Table256, TableSize256);
generate_table!(9, Table512, TableSize512);
generate_table!(10, Table1024, TableSize1024);
generate_table!(11, Table2048, TableSize2048);
generate_table!(12, Table4096, TableSize4096);

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
