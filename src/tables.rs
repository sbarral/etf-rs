use num_traits::Float;

/// Marker trait for table size.
pub trait TableSize: Copy + Clone {
    const BITS: u32;
}

/// Partition trait.
pub trait Partition<T: Float>: Copy + Clone + Default {
    fn x(&self, i: usize) -> T;
    fn set_x(&mut self, i: usize, x: T);
    fn dx(&self, i: usize) -> T {
        self.x(i+1) - self.x(i)
    }
    /// Number of sub-intervals.
    fn n(&self) -> usize;
}

/// Quadrature table trait.
pub trait Table<T: Float>: Copy + Clone + Default {
    type Size: TableSize;
    type Partition: Partition<T>;

    fn partition(&self) -> &Self::Partition;
    fn partition_mut(&mut self) -> &mut Self::Partition;
    fn yinf(&self, i: usize) -> T;
    fn ysup(&self, i: usize) -> T;
    fn set_yinf(&mut self, i: usize, yinf: T);
    fn set_ysup(&mut self, i: usize, ysup: T);
}

macro_rules! generate_tables {
    ($s:expr, $ts:ident, $p: ident, $t:ident) => {
        /// Table size marker.
        #[derive(Copy, Clone)]
        pub struct $ts;

        impl TableSize for $ts {
            const BITS: u32 = $s;
        }

        /// Partition.
        #[derive(Copy, Clone)]
        pub struct $p<T: Float> {
            pub x: [T; (1 << $s) + 1],
        }

        impl<T: Float> Default for $p<T> {
            fn default() -> Self {
                Self{ x: [T::ZERO; (1 << $s) + 1] }
            }
        }
        
        impl<T: Float> Partition<T> for $p<T> {
            fn x(&self, i: usize) -> T {
                self.x[i]
            }
            fn set_x(&mut self, i: usize, x: T) {
                self.x[i] = x;
            }
            fn n(&self) -> usize {
                self.x.len() - 1
            }
        }

        /// Quadrature table.
        #[derive(Copy, Clone)]
        pub struct $t<T: Float> {
            pub partition: $p<T>,
            pub yinf: [T; 1 << $s],
            pub ysup: [T; 1 << $s],
        }

        impl<T: Float> Default for $t<T> {
            fn default() -> Self {
                Self{
                    partition: $p::default(),
                    yinf: [T::ZERO; 1 << $s],
                    ysup: [T::ZERO; 1 << $s],
                }
            }
        }

        impl<T: Float> Table<T> for $t<T> {
            type Size = $ts;
            type Partition = $p<T>;

            fn partition(&self) -> &Self::Partition {
                &self.partition
            }
            fn partition_mut(&mut self) -> &mut Self::Partition {
                &mut self.partition
            }
            fn yinf(&self, i: usize) -> T {
                self.yinf[i]
            }
            fn ysup(&self, i: usize) -> T {
                self.ysup[i]
            }
            fn set_yinf(&mut self, i: usize, yinf: T) {
                self.yinf[i] = yinf;
            }
            fn set_ysup(&mut self, i: usize, ysup: T) {
                self.ysup[i] = ysup;
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
