extern crate nalgebra;
extern crate typenum;
extern crate num_traits;

use self::nalgebra::{DimName, Matrix};
use self::num_traits::identities::Zero;

pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn f64max(xs: &[f64]) -> f64 {
    let mut max = ::std::f64::MIN;
    for &x in xs {
        if x > max {max = x}
    }
    max
}

pub fn softmax(xs: &[f64], ys: &mut [f64]) {
    // p69
    // softmax := exp(ak) / (Σexp(ai))
    //         => exp(ak - C) / (Σexp(ai - C))
    let c = f64max(xs);
    let expa: f64 = xs.iter().map(|x| (x - c).exp()).sum();
    for (x, y) in xs.iter().zip(ys.iter_mut()) {
        *y = (x - c).exp() / expa;
    }
}

pub fn softmax_v(xs: &[f64]) -> Vec<f64> {
    let mut v = vec![0.0; xs.len()];
    softmax(&xs, v.as_mut_slice());
    v
}

pub fn cross_entropy_error<R, C, S>(output: &Matrix<f64, R, C, S>, label: &Matrix<f64, R, C, S>) -> f64
    where R: DimName,
          C: DimName,
          S: nalgebra::storage::OwnedStorage<f64, R, C>,
          S::Alloc: nalgebra::allocator::OwnedAllocator<f64, R, C, S>,
{
    let ln_output = output.map(|x|(x + 1e-7).ln());
    let y: f64 = label.component_mul(&ln_output).iter().sum();
    -y / label.nrows() as f64
}

pub fn max_index(xs: &[f64]) -> usize {
    xs.iter().enumerate().fold((0, ::std::f64::MIN), |(maxi, max), (i, &x)| {
        if x > max { (i, x) } else { (maxi, max) }
    }).0
}

pub fn numerical_gradient<F, R, C, S>(f: F, xs: &Matrix<f64, R, C, S>) -> Matrix<f64, R, C, S>
    where F: Fn(&Matrix<f64, R, C, S>) -> f64,
          R: DimName,
          C: DimName,
          S: nalgebra::storage::OwnedStorage<f64, R, C>,
          S::Alloc: nalgebra::allocator::OwnedAllocator<f64, R, C, S>,
{
    let h = 1e-4;
    let mut ans = Matrix::zero();
    for i in 0..xs.nrows() {
        for j in 0..xs.ncols() {
            let mut xs2 = xs.clone();
            xs2[(i, j)] = xs[(i, j)] + h;
            let y1 = f(&xs2);
            xs2[(i, j)] = xs[(i, j)] - h;
            let y0 = f(&xs2);
            ans[(i, j)] = (y1 - y0) / (2.*h)
        }
    }
    ans
}

#[cfg(test)]
mod tests {

    use super::*;

    use self::nalgebra::{U10, U11, MatrixNM};
    #[test]
    fn test_cross_entropy_error() {
        let output = MatrixNM::<f64, U11, U10>::from_row_slice(&[
            1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
        ]);
        let label = MatrixNM::<f64, U11, U10>::from_row_slice(&[
            1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 1., 0., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 1., 0., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 1., 0., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 1., 0., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 1., 0., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 1., 0., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 1., 0.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
            0., 0., 0., 0., 0., 0., 0., 0., 0., 1.,
        ]);
        let ans = cross_entropy_error(&output, &label);
        // assert_eq!(ans, 1.0);
        assert!(ans.abs() < 1e-7, "Expected |ans| < 1e-7 but actual ans is {}", ans);

        let output = MatrixNM::<f64, U11, U10>::from_row_slice(&[
            0., 1., 0., 0., 0., 0., 0., 0., 0., 0.,
            1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
            1., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        ]);
        let ans = cross_entropy_error(&output, &label);
        // assert_eq!(ans, 1.0);
        assert!((ans + (1e-7f64).ln()).abs() < 1e-07, "Expected {} but actual is {}", -(1e-7f64.ln()), ans);
    }

    #[test]
    fn test_max_index() {
        assert_eq!(max_index(&[]), 0);
        assert_eq!(max_index(&[1.0, 2.0, 1.5]), 1);
    }

    use self::nalgebra::{U2, RowVectorN};
    fn function_2(xs: &RowVectorN<f64, U2>) -> f64 {
        let x = xs[(0, 0)];
        let y = xs[(0, 1)];
        x*x + y*y
    }

    #[test]
    fn test_numerial_gradient() {
        assert_eq! {
            numerical_gradient(function_2, &RowVectorN::from_row_slice(&[3., 4.])),
            RowVectorN::from_row_slice(&[6.00000000000378, 7.999999999999119])
        };
    }
}
