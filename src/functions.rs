extern crate nalgebra;
extern crate typenum;

use self::nalgebra::{DimName, Matrix};

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cross_entropy_error() {
        let output = MatrixRx10::from_row_slice(&[
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
        let label = MatrixRx10::from_row_slice(&[
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

        let output = MatrixRx10::from_row_slice(&[
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

    #[ignore]
    #[test]
    fn test_load_image() {
        let ns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        assert_eq!(format!("{:?}", load_image("/home/cohama/proj/rust/deeplearning/data/mnist/train-images-idx3-ubyte", &ns).unwrap().as_slice().iter().map(|x|format!("{:X}", x)).collect::<Vec<_>>()), "");
    }

    #[ignore]
    #[test]
    fn test_load_label() {
        let ns = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];
        let v = load_label("/home/cohama/proj/rust/deeplearning/data/mnist/train-labels-idx1-ubyte", &ns).unwrap();
        // assert_eq!(format!("{:?}", load_label("/home/cohama/proj/rust/deeplearning/data/mnist/train-labels-idx1-ubyte", &ns).unwrap()), "");
        assert_eq!(format!("{:?}", label_as_onehot(&v)), "");
    }
    #[test]
    fn test_max_index() {
        assert_eq!(max_index(&[]), 0);
        assert_eq!(max_index(&[1.0, 2.0, 1.5]), 1);
    }
}
