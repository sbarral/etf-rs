//! Table size validity marker traits.

use crate::num::Float;
use super::partition::*;


/// Table size validity marker trait for asymmetric distributions.
pub trait ValidPartitionSize<T: Float>: Partition {}

/// Table size validity marker trait for symmetric distributions.
pub trait ValidSymmetricPartitionSize<T: Float>: Partition {}

// Valid tables sizes for f32 (asymmetric distributions).
impl ValidPartitionSize<f32> for P16 {}
impl ValidPartitionSize<f32> for P32 {}
impl ValidPartitionSize<f32> for P64 {}
impl ValidPartitionSize<f32> for P128 {}
impl ValidPartitionSize<f32> for P256 {}

// Valid tables sizes for f64 (asymmetric distributions).
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
