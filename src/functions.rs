extern crate nalgebra;
extern crate typenum;
extern crate num_traits;

use self::nalgebra::{Dim, DimName, Dynamic, Matrix, U1, MatrixVec};
use self::nalgebra::storage::{Storage, StorageMut, OwnedStorage};
use self::nalgebra::constraint::{ShapeConstraint, SameNumberOfRows, SameNumberOfColumns};
use self::nalgebra::allocator::{Allocator, SameShapeAllocator, OwnedAllocator};

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

#[inline]
pub fn cross_entropy_error<R1, C1, R2, C2, SA, SB>(output: &Matrix<f64, R1, C1, SA>, label: &Matrix<f64, R2, C2, SB>) -> f64
    where R1: Dim,
          C1: Dim,
          R2: Dim,
          C2: Dim,
          SA: OwnedStorage<f64, R1, C1> + Storage<f64, R1, C1>,
          SA::Alloc: OwnedAllocator<f64, R1, C1, SA>,
          SB: Storage<f64, R2, C2>,
          SB::Alloc: SameShapeAllocator<f64, R2, C2, R1, C1, SB>,
          ShapeConstraint: SameNumberOfRows<R2, R1> + SameNumberOfColumns<C2, C1>
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

// pub fn numerical_gradient<F, R, C, S, SR>(f: F, xs: &Matrix<f64, R, C, S>) -> Matrix<f64, R, C, S>
pub fn numerical_gradient<F>(f: &F, xs: &mut [f64], out: &mut [f64])
    where F: Fn() -> f64
{
    let h = 1e-4;
    let mut i = 0;
    for (x, o) in xs.iter_mut().zip(out.iter_mut()) {
        let tmp = *x; // copy
        *x = tmp + h;
        let y1 = f();
        *x = tmp - h;
        let y0 = f();
        *o = (y1 - y0) / (2.*h);
        *x = tmp;
        i += 1;
        println!("{}", i);
    }
}

pub fn rowwise_add<R, C1, C2, SA, SB>(m: &mut Matrix<f64, R, C1, SA>, n: &Matrix<f64, U1, C2, SB>)
    where R: Dim,
          C1: Dim,
          C2: Dim,
          SA: StorageMut<f64, R, C1>,
          SB: Storage<f64, U1, C2>,
          ShapeConstraint: SameNumberOfColumns<C1, C2>
{
    let (rows, cols) = (m.nrows(), m.ncols());
    for i in 0..rows {
        for j in 0..cols {
            m[(i, j)] += n[(0, j)]
        }
    }
}

pub fn pick_columns<R, C, S>(m: &Matrix<f64, R, C, S>, indices: &[usize]) -> Matrix<f64, R, Dynamic, MatrixVec<f64, R, Dynamic>>
    where R: DimName,
          C: Dim,
          S: Storage<f64, R, C>,
          S::Alloc: Allocator<f64, R, U1>,
{
    let colvecs = indices.iter().map(|&i| m.fixed_columns::<U1>(i)).collect::<Vec<_>>();
    Matrix::<f64, R, Dynamic, MatrixVec<f64, R, Dynamic>>::from_columns(&colvecs)
}

pub fn pick_rows<R, C, S>(m: &Matrix<f64, R, C, S>, indices: &[usize]) -> Matrix<f64, Dynamic, C, MatrixVec<f64, Dynamic, C>>
    where R: Dim,
          C: DimName,
          S: Storage<f64, R, C>,
          S::Alloc: Allocator<f64, U1, C>,
{
    let rowvecs = indices.iter().map(|&i| m.fixed_rows::<U1>(i)).collect::<Vec<_>>();
    Matrix::<f64, Dynamic, C, MatrixVec<f64, Dynamic, C>>::from_rows(&rowvecs)
}

#[cfg(test)]
mod tests {

    use super::*;

    use self::nalgebra::{U10, U11, MatrixNM, DMatrix, Matrix2x3, Matrix3, U2, RowVectorN};

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

    #[test]
    fn test_pick_columns() {
        let w = Matrix2x3::from_row_slice(&[1., 2., 3.,
                                            4., 5., 6.]);
        let pw = pick_columns(&w, &[0, 2]);
        assert_eq!(pw[(0, 0)], 1.);
        assert_eq!(pw[(0, 1)], 3.);
        assert_eq!(pw[(1, 0)], 4.);
        assert_eq!(pw[(1, 1)], 6.);
    }

    #[test]
    fn test_pick_rows() {
        let w = Matrix3::from_row_slice(&[1., 2., 3.,
                                          4., 5., 6.,
                                          7., 8., 9.]);
        let pw = pick_rows(&w, &[0, 2]);
        assert_eq!(pw[(0, 0)], 1.);
        assert_eq!(pw[(0, 1)], 2.);
        assert_eq!(pw[(0, 2)], 3.);
        assert_eq!(pw[(1, 0)], 7.);
        assert_eq!(pw[(1, 1)], 8.);
        assert_eq!(pw[(1, 2)], 9.);
    }
}
