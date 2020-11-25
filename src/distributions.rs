//! ETF-based implementation of common continuous probability distributions.

pub use cauchy::{Cauchy, CauchyError, CauchyFloat};
pub use normal::{CentralNormal, Normal, NormalError, NormalFloat};
pub use chi_squared::{ChiSquared, ChiSquaredFloat};

mod cauchy;
mod normal;
mod chi_squared;
