//! ETF-based implementation of common continuous probability distributions.

pub use cauchy::{Cauchy, CauchyError, CauchyFloat};
pub use chi_squared::{ChiSquared, ChiSquaredError, ChiSquaredFloat};
pub use gamma::{Gamma, GammaError, GammaFloat};
pub use gumbel::{Gumbel, GumbelError, GumbelFloat};
pub use normal::{CentralNormal, Normal, NormalError, NormalFloat};

mod cauchy;
mod chi_squared;
mod gamma;
mod gumbel;
mod normal;
