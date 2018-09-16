use num::Float;
use table::{Partition, Table};

use std::cmp;

// Tridiagonal matrix algorithm.
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

    // Solve the remaining upper bidiagonal system.
    sol[m - 1] = rhs[m - 1] / b[m - 1];
    for i in (0..m - 1).rev() {
        sol[i] = (rhs[i] - c[i] * sol[i + 1]) / b[i];
    }
}

/// Divide approximately evenly the area under a function.
///
/// Function `f` is approximated with `m` midpoint rectangles over a regular grid in order to
/// divide an interval [`x0`, `x1`] into the specified number of sub-intervals such that the areas
/// under the midpoint rectangle approximation is the same over each sub-interval.
/// The returned partition consists of the set of abscissae, bounds included.
///
pub fn midpoint_prepartition<T, F, P>(f: &F, x0: T, x1: T, m: usize) -> Box<P>
where
    T: Float,
    F: Fn(T) -> T,
    P: Partition<T>,
{
    // constants.
    let one_half = T::ONE / (T::ONE + T::ONE);
    let m = cmp::max(m, 1); // at least one rectangle

    // Mid-point evaluation.
    let dx = (x1 - x0) / T::cast_usize(m);
    let y: Vec<T> = (0..m)
        .map(|i| f(x0 + (T::cast_usize(i) + one_half) * dx))
        .collect();

    // Pre-allocate the result partition.
    let mut x = Box::new(P::default());
    {
        // Choose abscissae that evenly split the area under the curve.
        let x = x.as_mut_slice();
        let n = x.len() - 1;
        let ds = y.iter().fold(T::ZERO, |s, &y_| y_ + s) / T::cast_usize(n); // expected average sub-partition area
        let mut rect = 0;
        let mut x_rect = x0;
        let mut a_rect = y[0]; // cumulated rectangles area, normalized by 1/|dx|.
        for i in 1..n {
            // Expected cumulated area from x0 to current partition.
            let a = ds * T::cast_usize(i);

            // Integrate `f` from `x0` until `a` is smaller than `a_rect`.
            while a_rect < a {
                rect += 1;
                a_rect += y[rect];
                x_rect += dx;
            }

            // Interpolate `x`.
            x[i] = x_rect - dx * ((a_rect - a) / y[rect]);
        }
        x[0] = x0;
        x[n] = x1;
    }

    x
}

/// Compute an ETF tabulation using Newton's method.
///
/// A multivariate Newton method is used to compute a partition such that the rectangles making
/// up an upper Riemann sum of function `f` have equal areas.
///
/// If succesful, the returned table contains the partition as well as the extrema of the function
/// over each sub-interval.
///
/// The function `f`, its derivative 'df' and an ordered sequence of the extrema of 'f' (boundary
/// points excluded) must be provided, as well as a reasonable initial estimate of the partition.
///
/// Convergence is deemed achieved when the worse relative error on the upper rectangle areas
/// is less than the specified tolerance.
/// If convergence is not reached after the specified maximum number of iterations, `None` is
/// returned.
///
/// A relaxation coefficient lower (resp. greater) than 1 may be specified to improve convergence
/// robustness (resp. speed).
///
pub fn newton_tabulation<T, F, DF, A>(
    f: &F,
    df: &DF,
    x_init: A::Partition,
    x_extrema: &[T],
    tolerance: T,
    relaxation: T,
    max_iter: u16,
) -> Option<Box<A>>
where
    T: Float,
    F: Fn(T) -> T,
    DF: Fn(T) -> T,
    A: Table<T>,
{
    // Initialize the quandrature table partition with the initial partition.
    let mut table = Box::new(A::default());
    table.as_mut_view().x.copy_from_slice(x_init.as_slice());

    // Main vectors.
    let n = table.as_view().yinf.len();
    let mut y = vec![T::ZERO; n + 1];
    let mut dx = vec![T::ZERO; n - 1];
    let mut dy_dx = vec![T::ZERO; n + 1];
    let mut dysup_dxl = vec![T::ZERO; n];
    let mut dysup_dxr = vec![T::ZERO; n];
    let mut minus_s = vec![T::ZERO; n - 1];
    let mut ds_dxc = vec![T::ZERO; n - 1];
    let mut ds_dxl = vec![T::ZERO; n - 1];
    let mut ds_dxr = vec![T::ZERO; n - 1];

    let y_extrema: Vec<T> = x_extrema.iter().cloned().map(f).collect();

    // Boundary values are constants.
    y[0] = f(table.as_view().x[0]);
    y[n] = f(table.as_view().x[n]);
    dy_dx[0] = T::ZERO;
    dy_dx[n] = T::ZERO;

    // Loop until convergence is achieved or the maximum number of iteration is reached.
    let mut loop_iter = 0..max_iter;
    loop {
        // Convenient aliases.
        let table_mut_view = table.as_mut_view();
        let x = table_mut_view.x;
        let yinf = table_mut_view.yinf;
        let ysup = table_mut_view.ysup;

        // Update inner nodes values.
        for i in 1..n {
            y[i] = f(x[i]);
            dy_dx[i] = df(x[i]);
        }

        // Determine the supremum fsup of y within [x[i], x[i+1]),
        // the partial derivatives of fsup with respect to x[i] and x[i+1],
        // the minimum and maximum partition areas and the total area.
        let mut extrema = x_extrema.iter().zip(y_extrema.iter());
        let mut extremum = extrema.next();
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

            // Check if there are extrema within [x[i], x[i+1]).
            while let Some((&x_e, &y_e)) = extremum {
                if (x_e > x[i]) != (x_e > x[i + 1]) {
                    if y_e > ysup[i] {
                        ysup[i] = y_e;
                        dysup_dxl[i] = T::ZERO;
                        dysup_dxr[i] = T::ZERO;
                    }
                    extremum = extrema.next();
                }
            }

            let area = ysup[i] * (x[i + 1] - x[i]).abs();
            max_area = max_area.max(area);
            min_area = min_area.min(area);
            sum_area += area;
        }

        // Return the table if convergence was achieved.
        let mean_area = sum_area / T::cast_usize(n);
        if (max_area - min_area) < tolerance * mean_area {
            // Determine the infimum yinf of y in [x[i], x[i+1]).
            extrema = x_extrema.iter().zip(y_extrema.iter());
            extremum = extrema.next();

            for i in 0..n {
                yinf[i] = if y[i] > y[i + 1] { y[i + 1] } else { y[i] };

                // Check if there are extrema within the (x[i], x[i+1]) range.
                while let Some((&x_e, &y_e)) = extremum {
                    if (x_e > x[i]) != (x_e > x[i + 1]) {
                        if y_e < yinf[i] {
                            yinf[i] = y_e;
                        }
                    }
                    extremum = extrema.next();
                }
            }
            break;
        }

        // Exit if convergence could not be achieved.
        if loop_iter.next().is_none() {
            return None;
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

        // For the sake of stability, updated positions are constrained within
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

    return Some(table);
}
