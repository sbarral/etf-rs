//! ETF-based implementation of common continuous probability distributions.

pub use normal::{Normal, CentralNormal};
pub use cauchy::Cauchy;

mod normal;
mod cauchy;
