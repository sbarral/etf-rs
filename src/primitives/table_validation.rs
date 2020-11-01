//! Table size validity marker traits.

use crate::num::Float;
use super::partition::*;


/// Table size validity marker trait for asymmetric distributions.
pub trait ValidPartitionSize<T: Float>: Partition<T> {}

/// Table size validity marker trait for symmetric distributions.
pub trait ValidSymmetricPartitionSize<T: Float>: Partition<T> {}

// Valid tables sizes for f32 (asymmetric distributions).
impl ValidPartitionSize<f32> for P16<f32> {}
impl ValidPartitionSize<f32> for P32<f32> {}
impl ValidPartitionSize<f32> for P64<f32> {}
impl ValidPartitionSize<f32> for P128<f32> {}
impl ValidPartitionSize<f32> for P256<f32> {}

// Valid tables sizes for f64 (asymmetric distributions).
impl ValidPartitionSize<f64> for P16<f64> {}
impl ValidPartitionSize<f64> for P32<f64> {}
impl ValidPartitionSize<f64> for P64<f64> {}
impl ValidPartitionSize<f64> for P128<f64> {}
impl ValidPartitionSize<f64> for P256<f64> {}
impl ValidPartitionSize<f64> for P512<f64> {}
impl ValidPartitionSize<f64> for P1024<f64> {}
impl ValidPartitionSize<f64> for P2048<f64> {}
impl ValidPartitionSize<f64> for P4096<f64> {}

// Valid tables sizes for f32 (symmetric distributions).
impl ValidSymmetricPartitionSize<f32> for P16<f32> {}
impl ValidSymmetricPartitionSize<f32> for P32<f32> {}
impl ValidSymmetricPartitionSize<f32> for P64<f32> {}
impl ValidSymmetricPartitionSize<f32> for P128<f32> {}

// Valid tables sizes for f64 (symmetric distributions).
impl ValidSymmetricPartitionSize<f64> for P16<f64> {}
impl ValidSymmetricPartitionSize<f64> for P32<f64> {}
impl ValidSymmetricPartitionSize<f64> for P64<f64> {}
impl ValidSymmetricPartitionSize<f64> for P128<f64> {}
impl ValidSymmetricPartitionSize<f64> for P256<f64> {}
impl ValidSymmetricPartitionSize<f64> for P512<f64> {}
impl ValidSymmetricPartitionSize<f64> for P1024<f64> {}
impl ValidSymmetricPartitionSize<f64> for P2048<f64> {}
