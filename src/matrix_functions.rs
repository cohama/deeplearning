
use ndarray;
use ndarray::{ArrayBase, Array, Array2};

use functions::softmax;

#[inline]
pub fn cross_entropy_error<S, D>(output: &ArrayBase<S, D>, label: &ArrayBase<S, D>) -> f64
    where S: ndarray::Data<Elem=f64>,
          D: ndarray::Dimension,
{
    let ref ln_output = output.map(|x|(x + 1e-7).ln());
    let y: f64 = (label * ln_output).scalar_sum();
    -y / label.shape()[0] as f64
}

pub fn pick_rows<S, T>(a: &ArrayBase<S, ndarray::Ix2>, indice: &[usize]) -> Array2<T>
    where S: ndarray::Data<Elem=T>,
          T: Copy,
{
    Array::from_shape_fn((indice.len(), a.cols()), |(i, j)| {
        a[[indice[i], j]]
    })
}

pub fn numerical_gradient_ndarray<F>(f: &F, xs: &mut Array2<f64>, out: &mut Array2<f64>)
    where F: Fn() -> f64
{
    let h = 1e-4;
    // println!("xs.shape: ({}, {}), out.shape: ({}, {})", xs.rows(), xs.cols(), out.rows(), out.cols());
    for (x, o) in xs.iter_mut().zip(out.iter_mut()) {
        let tmp = *x; // copy
        *x = tmp + h;
        let y1 = f();
        *x = tmp - h;
        let y0 = f();
        *o = (y1 - y0) / (2.*h);
        *x = tmp;
    }
}

pub fn softmax_rowwise(xs: &Array2<f64>) -> Array2<f64> {
    let mut ys = xs.clone();
    for (mut y, x) in ys.genrows_mut().into_iter().zip(xs.genrows()) {
        softmax(x.as_slice().unwrap(), y.as_slice_mut().unwrap());
    }
    ys
}

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


    #[test]
    fn test_pick_rows() {
        let b = Array2::<i32>::from_shape_vec((4, 3), vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]).unwrap();
        assert_eq!{
            pick_rows(&b, &[0, 2]),
            array![[1, 2, 3], [7, 8, 9]]
        };
    }

    #[test]
    fn test_softmax_rowwise() {
        let x = array![[1., 2., 3.], [2., 2., 2.], [20., 5., 1.]];
        let y = softmax_rowwise(&x);
        for r in y.genrows() {
            assert_relative_eq!(r.scalar_sum(), 1.0);
        }
    }
}
