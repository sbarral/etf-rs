use crate::num::Float;
use crate::partition::*;


/// Valid table size marker trait.
pub trait ValidPartitionSize<T: Float>: Partition {}

/// Valid symmetric table size marker trait.
pub trait ValidSymmetricPartitionSize<T: Float>: Partition {}

// Valid tables sizes for f32.
impl ValidPartitionSize<f32> for P16 {}
impl ValidPartitionSize<f32> for P32 {}
impl ValidPartitionSize<f32> for P64 {}
impl ValidPartitionSize<f32> for P128 {}
impl ValidPartitionSize<f32> for P256 {}

// Valid tables sizes for f64.
impl ValidPartitionSize<f64> for P16 {}
impl ValidPartitionSize<f64> for P32 {}
impl ValidPartitionSize<f64> for P64 {}
impl ValidPartitionSize<f64> for P128 {}
impl ValidPartitionSize<f64> for P256 {}
impl ValidPartitionSize<f64> for P512 {}
impl ValidPartitionSize<f64> for P1024 {}
impl ValidPartitionSize<f64> for P2048 {}
impl ValidPartitionSize<f64> for P4096 {}

// Valid tables sizes for f32 (symmetric distributions).
impl ValidSymmetricPartitionSize<f32> for P16 {}
impl ValidSymmetricPartitionSize<f32> for P32 {}
impl ValidSymmetricPartitionSize<f32> for P64 {}
impl ValidSymmetricPartitionSize<f32> for P128 {}

// Valid tables sizes for f64 (symmetric distributions).
impl ValidSymmetricPartitionSize<f64> for P16 {}
impl ValidSymmetricPartitionSize<f64> for P32 {}
impl ValidSymmetricPartitionSize<f64> for P64 {}
impl ValidSymmetricPartitionSize<f64> for P128 {}
impl ValidSymmetricPartitionSize<f64> for P256 {}
impl ValidSymmetricPartitionSize<f64> for P512 {}
impl ValidSymmetricPartitionSize<f64> for P1024 {}
impl ValidSymmetricPartitionSize<f64> for P2048 {}
