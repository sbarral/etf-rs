mod collisions;
mod goodness_of_fit;
mod util;

pub use collisions::collisions;
pub use util::{test_rng, TestFloat};
pub use goodness_of_fit::goodness_of_fit;