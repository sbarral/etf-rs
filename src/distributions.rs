//! ETF-based implementation of common continuous probability distributions.

pub use cauchy::{Cauchy, CauchyError, CauchyFloat};
pub use normal::{CentralNormal, Normal, NormalError, NormalFloat};

mod cauchy;
mod normal;
