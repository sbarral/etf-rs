//! Utilites for ETF distributions generation. 

use crate::num::{Float, Func};
use super::{Envelope, InitTable, NodeArray, Partition};

use rand_core::RngCore;

mod error;
pub use error::*;

// Tri-diagonal matrix algorithm.
//
// For the sake of efficiency, diagonal terms and RHS are modified in-place.
// All slices have equal length.
fn solve_tma<T: Float>(a: &[T], b: &mut [T], c: &[T], rhs: &mut [T], sol: &mut [T]) {
    let m = a.len();

    // Eliminate the sub-diagonal.
    for i in 1..m {
        let pivot = a[i] / b[i - 1];
        b[i] = b[i] - pivot * c[i - 1];
        rhs[i] = rhs[i] - pivot * rhs[i - 1];
    }

    // Solve the remaining upper bi-diagonal system.
    sol[m - 1] = rhs[m - 1] / b[m - 1];
    for i in (0..m - 1).rev() {
        sol[i] = (rhs[i] - c[i] * sol[i + 1]) / b[i];
    }
}

/// Generates a partition by dividing approximately evenly the area under a
/// function.
///
/// Function `f` is first approximated by a rectangular midpoint quadrature over
/// a regular partition of [`x0`, `x1`] into `m` sub-intervals. Then, a
/// non-regular partition of [`x0`, `x1`] is computed such that the area under
/// the quadrature approximation is equal over each sub-interval.
///
/// If argument `m` is zero, then the number of midpoint quadrature
/// sub-intervals is set equal to the number of sub-intervals of the target
/// partition.
pub fn midpoint_prepartition<P, T, F>(f: &F, x0: T, x1: T, m: usize) -> NodeArray<P, T>
where
    P: Partition,
    T: Float,
    F: Func<T>,
{
    // constants.
    let one_half = T::ONE / (T::ONE + T::ONE);
    let m = if m != 0 { m } else { P::SIZE };

    // Mid-point evaluation.
    let dx = (x1 - x0) / T::cast_usize(m);
    let y: Vec<T> = (0..m)
        .map(|i| f.eval(x0 + (T::cast_usize(i) + one_half) * dx))
        .collect();

    // Pre-allocate the result partition.
    let mut x = NodeArray::new();
    {
        // Choose abscissae that evenly split the area under the curve.
        let x = x.as_mut();
        let n = x.len() - 1;
        let ds = y.iter().fold(T::ZERO, |s, &y_| y_ + s) / T::cast_usize(n); // expected average sub-partition area
        let mut rect = 0;
        let mut x_rect = x0 + dx;
        let mut a_rect = y[0]; // cumulated rectangles area, normalized by 1/|dx|.
        for i in 1..n {
            // Expected cumulated area from x0 to current partition.
            let a = ds * T::cast_usize(i);

            // Integrate `f` from `x0` until `a` is smaller than `a_rect`.
            while a_rect < a {
                rect += 1;
                a_rect = a_rect + y[rect];
                x_rect = x_rect + dx;
            }

            // Interpolate `x`.
            x[i] = x_rect - dx * ((a_rect - a) / y[rect]);
        }
        x[0] = x0;
        x[n] = x1;
    }

    x
}

/// Computes a distribution initialization table using Newton's method.
///
/// A multivariate Newton method is used to compute a partition such that the
/// rectangles making up an upper Riemann sum of function `f` have equal areas.
///
/// If successful, the returned table contains the partition as well as the
/// extrema of the function over each sub-interval.
///
/// Function `f`, its derivative `df` and an ordered sequence of the extrema of
/// `f` (boundary points excluded) must be provided, as well as a reasonable
/// initial guess for the partition.
///
/// Convergence is deemed achieved when the difference between the areas of the
/// largest rectangle and of the smallest rectangle relative to the average area
/// of all rectangles is less than the specified tolerance. If convergence is
/// not reached after the specified maximum number of iterations, a
/// `TabulationError` is returned.
///
/// A relaxation coefficient lower than 1 (resp. greater than 1) may be
/// specified to improve convergence robustness (resp. speed).
pub fn newton_tabulation<P, T, F, DF>(
    f: &F,
    df: &DF,
    x_init: &NodeArray<P, T>,
    x_extrema: &[T],
    tolerance: T,
    relaxation: T,
    max_iter: u32,
) -> TabulationResult<InitTable<P, T>>
where
    P: Partition,
    T: Float,
    F: Func<T>,
    DF: Func<T>,
{
    // Initialize the quadrature table partition with the initial partition.
    let mut table = InitTable::new();
    table.x.as_mut().copy_from_slice(x_init.as_ref());

    // Main vectors.
    let n = P::SIZE;
    let mut y = vec![T::ZERO; n + 1];
    let mut dx = vec![T::ZERO; n - 1];
    let mut dy_dx = vec![T::ZERO; n + 1];
    let mut dysup_dxl = vec![T::ZERO; n];
    let mut dysup_dxr = vec![T::ZERO; n];
    let mut minus_s = vec![T::ZERO; n - 1];
    let mut ds_dxc = vec![T::ZERO; n - 1];
    let mut ds_dxl = vec![T::ZERO; n - 1];
    let mut ds_dxr = vec![T::ZERO; n - 1];

    let y_extrema: Vec<T> = x_extrema.iter().cloned().map(|x| f.eval(x)).collect();
    // Make a vector of the (x,y) tuples of all extrema that are actually
    // wihtin the partition.
    let extrema: Vec<(T, T)> = x_extrema
        .iter()
        .cloned()
        .zip(y_extrema.iter().cloned())
        .filter(|&(x_e, _)| (x_e > table.x.as_ref()[0]) && (x_e < table.x.as_ref()[n]))
        .collect();

    // Boundary values are constants.
    y[0] = f.eval(table.x.as_ref()[0]);
    y[n] = f.eval(table.x.as_ref()[n]);
    dy_dx[0] = T::ZERO;
    dy_dx[n] = T::ZERO;

    // Loop until convergence is achieved or the maximum number of iteration is reached.
    let mut loop_iter = 0..max_iter;
    loop {
        // Convenient aliases.
        let x = table.x.as_mut();
        let yinf = table.yinf.as_mut();
        let ysup = table.ysup.as_mut();

        // Update inner nodes values.
        for i in 1..n {
            y[i] = f.eval(x[i]);
            dy_dx[i] = df.eval(x[i]);
        }

        // Determine the supremum fsup of y within [x[i], x[i+1]),
        // the partial derivatives of fsup with respect to x[i] and x[i+1],
        // the minimum and maximum partition areas and the total area.
        let mut extrema_iter = extrema.iter();
        let mut extremum = extrema_iter.next(); // cached value of the last extremum
        let mut max_area = T::ZERO;
        let mut min_area = T::INFINITY;
        let mut sum_area = T::ZERO;
        for i in 0..n {
            let (ysup_, dysup_dxl_, dysup_dxr_) = if y[i] > y[i + 1] {
                (y[i], dy_dx[i], T::ZERO)
            } else {
                (y[i + 1], T::ZERO, dy_dx[i + 1])
            };
            (*ysup)[i] = ysup_;
            dysup_dxl[i] = dysup_dxl_;
            dysup_dxr[i] = dysup_dxr_;

            // Check if there are suprema between x[i] and x[i+1] and
            // advance the extrema iterator until the current extremum no
            // longer lies between x[i] and x[i+1].
            while let Some(&(x_e, y_e)) = extremum {
                if (x_e > x[i]) != (x_e > x[i + 1]) {
                    if y_e > ysup[i] {
                        ysup[i] = y_e;
                        dysup_dxl[i] = T::ZERO;
                        dysup_dxr[i] = T::ZERO;
                    }
                    extremum = extrema_iter.next();
                } else {
                    break;
                }
            }

            let area = ysup[i] * (x[i + 1] - x[i]).abs();
            max_area = max_area.max(area);
            min_area = min_area.min(area);
            sum_area = sum_area + area;
        }

        // Return the table if convergence was achieved.
        let mean_area = sum_area / T::cast_usize(n);

        if (max_area - min_area) < tolerance * mean_area {
            // At this point the areas differ slightly, which would introduce
            // some bias when the partitions are sampled as the assumption of
            // equiprobability would be violated.
            //
            // A simple way to cancel this bias is to slightly increase the
            // `ysup` values so as to make all areas equal to `max_area`. This
            // makes the top-floor rejection slightly suboptimal, but the loss
            // of efficiency is unlikely to be noticeable for reasonable values
            // of the tolerance.
            for i in 0..n {
                ysup[i] = max_area / (x[i + 1] - x[i]).abs();
            }

            // Determine the infimum yinf of y in [x[i], x[i+1]).
            extrema_iter = extrema.iter();
            extremum = extrema_iter.next();

            for i in 0..n {
                yinf[i] = if y[i] > y[i + 1] { y[i + 1] } else { y[i] };

                // Check if there are minima between x[i] and x[i+1] and
                // advance the extrema iterator until the current extremum no
                // longer lies between x[i] and x[i+1].
                while let Some(&(x_e, y_e)) = extremum {
                    if (x_e > x[i]) != (x_e > x[i + 1]) {
                        if y_e < yinf[i] {
                            yinf[i] = y_e;
                        }
                        extremum = extrema_iter.next();
                    } else {
                        break;
                    }
                }
            }

            return Ok(table);
        }

        // Exit if convergence could not be achieved.
        if loop_iter.next().is_none() {
            return Err(TabulationError::new(max_iter));
        }

        // Difference in area between neigboring rectangles and partial
        // derivatives of s with respect to x[i], x[i+1] and x[i+2].
        for i in 0..(n - 1) {
            minus_s[i] = ysup[i] * (x[i + 1] - x[i]) - ysup[i + 1] * (x[i + 2] - x[i + 1]);

            ds_dxl[i] = ysup[i] - (x[i + 1] - x[i]) * dysup_dxl[i];
            ds_dxc[i] = (x[i + 2] - x[i + 1]) * dysup_dxl[i + 1]
                - (x[i + 1] - x[i]) * dysup_dxr[i]
                - (ysup[i] + ysup[i + 1]);
            ds_dxr[i] = ysup[i + 1] + (x[i + 2] - x[i + 1]) * dysup_dxr[i + 1];
        }

        // Solve the tri-diagonal system S + (dS/dX)*dX = 0 with:
        //         | ds0/dx1 ds0/dx2    0     ...                    0     |
        //         | ds1/dx1 ds1/dx2 ds1/dx3    0     ...            0     |
        // dS/dX = |    0    ds2/dx2 ds2/dx3 ds2/dx4    0     ...    0     |
        //         |                       ...                             |
        //         |    0     ...     0    ds(n-2)/dx(n-2) ds(n-2)/dx(n-2) |
        //
        //
        // and:
        //      | dx1     |         | minus_s0     |
        // dX = | ...     |    -S = | ...    |
        //      | dx(n-1) |         | minus_s(n-2) |
        solve_tma(&ds_dxl, &mut ds_dxc, &ds_dxr, &mut minus_s, &mut dx);

        // Improve robustness by constraining updated positions within
        // the bounds set by former neighbors positions.
        {
            for i in 1..n {
                let (xmin, xmax) = if x[i + 1] > x[i - 1] {
                    (x[i - 1], x[i + 1])
                } else {
                    (x[i + 1], x[i - 1])
                };

                let mut xi = x[i] + relaxation * dx[i - 1];
                xi = xi.min(xmax);
                xi = xi.max(xmin);
                x[i] = xi;
            }
        }
    }
}

/// Distribution envelope based on a shifted Weibull distribution tail.
///
/// The tail of a shifted Weibull probability density function constitutes a
/// reasonably efficient envelope function for many distributions while being at
/// the same time relatively cheap to generate by inverse transform sampling.
///
/// The corresponding envelope function is:
///
///  `f(x) = w*a/|b|*((x-c)/b)^(a-1)*exp[-((x-c)/b)^a]`,
///
/// if `x/b > x0/b`, or `f(x) = 0` otherwise.
///
/// The parameters are:
///
/// * `w`: the *weight* (amplitude) of the envelope relative to the normalized
///   Weibull PDF
/// * `a>0`: the *scale* parameter
/// * `b≠0`: the *shape* parameter
/// * `c`: the *location* parameter
/// * `x0`: the *cut-in* position at which the tail starts
///
/// This generalization of the Weibull PDF can be shifted along the `x` axis
/// (location parameter `c`) and can be mirrored with respect to `x=c` by using
/// a negative `b`. The cut-in position of the tail must satisfy `x0 ≥ c` if
/// `b>0`, or `x0 ≤ c` if `b<0`.
///
/// The weight `w` controls the vertical scaling of the function relative to the
/// normalized Weibull PDF (`w=1`).
///
#[derive(Copy, Clone, Debug)]
pub struct WeibullEnvelope<T, F> {
    a: T,
    inv_a: T,
    b: T,
    inv_b: T,
    c: T,
    x0: T,
    s: T,
    alpha: T,
    f: F,
}

impl<T: Float, F: Func<T>> WeibullEnvelope<T, F> {
    /// Creates a new Weibull tail envelope distribution for a given probability
    /// density function.
    ///
    /// The probability density function `pdf` of the distribution to be sampled
    /// must be below the envelope over the whole sampling range, meaning for
    /// all `x` greater than the cut-in tail position if the shape parameter is
    /// positive, or for all `x` lesser than the cut-in tail position if the
    /// shape parameter is negative.
    pub fn new(weight: T, scale: T, shape: T, location: T, cut_in: T, pdf: F) -> Self {
        Self {
            a: scale,
            inv_a: T::ONE / scale,
            b: shape,
            inv_b: T::ONE / shape,
            c: location,
            x0: cut_in,
            s: weight * T::abs(scale / shape),
            alpha: T::powf((cut_in - location) / shape, scale),
            f: pdf,
        }
    }

    /// Computes the area under the envelope.
    pub fn area(&self) -> T {
        let z0 = T::powf((self.x0 - self.c) * self.inv_b, self.a);

        self.s * T::exp(-z0) * self.inv_a * self.b
    }
}

impl<T: Float, F: Func<T>> Envelope<T> for WeibullEnvelope<T, F> {
    fn try_sample<R: RngCore + ?Sized>(&self, rng: &mut R) -> Option<T> {
        let r = T::gen(rng);
        let x = self.c + self.b * T::powf(self.alpha - T::ln(T::ONE - r), self.inv_a);
        let x_scaled = (x - self.c) * self.inv_b;
        let z = T::powf(x_scaled, self.a - T::ONE);
        let y = self.s * z * T::exp(-x_scaled * z);

        let r_accept = T::gen(rng);
        if y * r_accept <= self.f.eval(x) {
            Some(x)
        } else {
            None
        }
    }
}
