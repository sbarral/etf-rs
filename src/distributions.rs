//! ETF-based implementation of common continuous probability distributions.

pub use cauchy::{Cauchy, CauchyError, CauchyFloat};
pub use normal::{CentralNormal, Normal, NormalError, NormalFloat};
pub use chi_squared::{ChiSquared, ChiSquaredError, ChiSquaredFloat};
pub use gamma::{Gamma, GammaError, GammaFloat};

mod cauchy;
mod normal;
mod chi_squared;
mod gamma;
