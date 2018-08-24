use num_traits::Float;
use tables::Partition;

use std::cmp;

// Tridiagonal matrix algorithm.
// For efficiency reasons, the diagonal term and the RHS are modified.
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

/// Computes a partition dividing approximately evenly the area under a
/// function, using the trapezoidal rule to approximate the area.
///
/// The trapezoidal rule is applied to function `f` over a regular grid with `nb_points` grid
/// points (including outer and inner nodes) in order to divide interval [`x0`, `x1`] into the
/// specified number of sub-intervals such that the areas under the trapeze quandrature is the
/// same over each sub-interval.
/// The returned partition consists of the set of abscissae, bounds included.
///
pub fn trapeze_prepartition<T, F, P>(f: F, x0: T, x1: T, nodes: usize) -> Box<P>
where
    T: Float,
    F: Fn(T) -> T,
    P: Partition<T>,
{
    let one_half = T::ONE / (T::ONE + T::ONE);
    let n = cmp::max(nodes, 2); // need at least two grid points

    // Interpolate the function.
    let dx = (x1 - x0) / T::cast_usize(n - 1);
    let mut x: Vec<T> = (0..n - 1).map(|i| x0 + (T::cast_usize(i) * dx)).collect();
    x.push(x1); // set directly to avoid rounding errors from above formula
    let y: Vec<T> = x.iter().cloned().map(f).collect();

    // Total area (scaled by 1/dx).
    let s = one_half * (y[0] + y[n - 1])
        + y.iter()
            .skip(1)
            .take(n - 2)
            .fold(T::ZERO, |sum, &yi| sum + yi);

    // Choose abscissae that evenly split the area under the curve.
    let mut partition = Box::new(P::default());

    let m = partition.n();
    let mut i = 0;
    let mut al = T::ZERO;
    let mut ar = one_half * (y[0] + y[1]);
    for j in 1..(m - 1) {
        // Expected area over the sub-partition [x0, xj].
        let a = s * (T::cast_usize(j) / T::cast_usize(m));

        // Integrate `f` with the trapezoidal rule until the area
        // is at least equal to `a`.
        while ar < a {
            i += 1; // next trapezoid
            al = ar;
            ar = ar + one_half * (y[i] + y[i + 1]);
            // `ar-al` is the area of the next trapezoid.
        }
        partition.set_x(j, x[i] + (x[i + 1] - x[i]) * ((a - al) / (ar - al)));
    }
    partition.set_x(0, x0);
    partition.set_x(m, x1);
    
    partition
}
