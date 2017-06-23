
use ndarray;
use ndarray::{ArrayBase};

#[inline]
pub fn cross_entropy_error<S, D>(output: &ArrayBase<S, D>, label: &ArrayBase<S, D>) -> f64
    where S: ndarray::Data<Elem=f64>,
          D: ndarray::Dimension,
{
    let ref ln_output = output.map(|x|(x + 1e-7).ln());
    let y: f64 = (label * ln_output).scalar_sum();
    -y / label.shape()[0] as f64
}

// pub fn pick_columns<R, C, S>(m: &Matrix<f64, R, C, S>, indices: &[usize]) -> Matrix<f64, R, Dynamic, MatrixVec<f64, R, Dynamic>>
//     where R: DimName,
//           C: Dim,
//           S: Storage<f64, R, C>,
//           S::Alloc: Allocator<f64, R, U1>,
// {
//     let colvecs = indices.iter().map(|&i| m.fixed_columns::<U1>(i)).collect::<Vec<_>>();
//     Matrix::<f64, R, Dynamic, MatrixVec<f64, R, Dynamic>>::from_columns(&colvecs)
// }

// pub fn pick_rows<R, C, S>(m: &Matrix<f64, R, C, S>, indices: &[usize]) -> Matrix<f64, Dynamic, C, MatrixVec<f64, Dynamic, C>>
//     where R: Dim,
//           C: DimName,
//           S: Storage<f64, R, C>,
//           S::Alloc: Allocator<f64, U1, C>,
// {
//     let rowvecs = indices.iter().map(|&i| m.fixed_rows::<U1>(i)).collect::<Vec<_>>();
//     Matrix::<f64, Dynamic, C, MatrixVec<f64, Dynamic, C>>::from_rows(&rowvecs)
// }

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_cross_entropy_error() {
        let output = array![
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]
        ];
        let label = array![
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 1., 0., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 1., 0.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]
        ];
        let ans = cross_entropy_error(&output, &label);

        relative_eq!(ans.abs(), 0., epsilon = 1e-7);

        let output = array![
            [0., 1., 0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
            [1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        ];
        let ans = cross_entropy_error(&output, &label);
        // assert_eq!(ans.abs(), 1.0);
        // relative_eq!(ans, 1.0);
        assert_relative_eq!(ans, -(1e-7f64).ln(), epsilon = 1e-7);
    }

//     fn function_2(xs: &RowVectorN<f64, U2>) -> f64 {
//         let x = xs[(0, 0)];
//         let y = xs[(0, 1)];
//         x*x + y*y
//     }

//     #[test]
//     fn test_numerial_gradient() {
//         assert_eq! {
//             numerical_gradient(function_2, &RowVectorN::from_row_slice(&[3., 4.])),
//             RowVectorN::from_row_slice(&[6.00000000000378, 7.999999999999119])
//         };
//     }

//     #[test]
//     fn test_pick_columns() {
//         let w = Matrix2x3::from_row_slice(&[1., 2., 3.,
//                                             4., 5., 6.]);
//         let pw = pick_columns(&w, &[0, 2]);
//         assert_eq!(pw[(0, 0)], 1.);
//         assert_eq!(pw[(0, 1)], 3.);
//         assert_eq!(pw[(1, 0)], 4.);
//         assert_eq!(pw[(1, 1)], 6.);
//     }

//     #[test]
//     fn test_pick_rows() {
//         let w = Matrix3::from_row_slice(&[1., 2., 3.,
//                                           4., 5., 6.,
//                                           7., 8., 9.]);
//         let pw = pick_rows(&w, &[0, 2]);
//         assert_eq!(pw[(0, 0)], 1.);
//         assert_eq!(pw[(0, 1)], 2.);
//         assert_eq!(pw[(0, 2)], 3.);
//         assert_eq!(pw[(1, 0)], 7.);
//         assert_eq!(pw[(1, 1)], 8.);
//         assert_eq!(pw[(1, 2)], 9.);
//     }

}
