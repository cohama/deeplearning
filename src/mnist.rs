extern crate nalgebra;
extern crate typenum;

use self::typenum::{U784};
use self::nalgebra::{DimName, Matrix, MatrixNM, VectorN, U10, U11};
use std::io::Result as IoResult;
use std::io::{Seek, SeekFrom, Read};
use std::fs::File;

use self::nalgebra::convert_ref;

pub type USampleSize = U11;
pub const SAMPLE_SIZE: i32 = 11;

pub type MatrixRx784<N> = MatrixNM<N, USampleSize, U784>;
pub type MatrixRx10<N> = MatrixNM<N, USampleSize, U10>;
pub type VectorR<N> = VectorN<N, USampleSize>;
// pub struct Mnist {
//     pub train_image: MatrixRx784<u8>,
//     pub train_label: MatrixRx10<u8>,
//     pub test_image: MatrixRx784<u8>,
//     pub test_label: MatrixRx10<u8>,
// }

// pub fn load_mnist_data() -> Mnist {
// }


// pub fn load_image(path: &str, indice: &[usize]) -> IoResult<MatrixRx784<u8>> {
pub fn load_image(path: &str, indice: &[usize]) -> IoResult<MatrixRx784<u8>> {
    let mut file = File::open(path)?;
    let mut vv = vec![];
    for n in indice {
        let mut v = vec![];
        file.seek(SeekFrom::Start((16 + n * 784) as u64))?;
        let _ = file.by_ref().take(784).read_to_end(&mut v);
        vv.append(&mut v);
    }
    Ok(MatrixRx784::from_column_slice(vv.as_slice()))
}

pub fn load_label(path: &str, indice: &[usize]) -> IoResult<VectorR<u8>> {
    let mut file = File::open(path)?;
    let mut v = vec![];
    for n in indice {
        file.seek(SeekFrom::Start((16 + n) as u64))?;
        let mut buf = [0; 1];
        file.by_ref().read(&mut buf)?;
        v.push(buf[0]);
    }
    Ok(VectorR::from_column_slice(v.as_slice()))
}

pub fn label_as_onehot(label: &VectorR<u8>) -> MatrixRx10<u8> {
    let mut vv = vec![];
    for v in label.iter() {
        let mut onehot = [0; 10];
        onehot[*v as usize] = 1;
        vv.extend_from_slice(&mut onehot);
    }
    MatrixRx10::from_column_slice(vv.as_slice())
}

pub fn cross_entropy_error<R, C, S1, S2>(output: &Matrix<f64, R, C, S1>, label: &Matrix<u8, R, C, S2>) -> f64
    where R: DimName,
          C: DimName,
          S1: nalgebra::storage::OwnedStorage<f64, R, C>,
          S2: nalgebra::storage::OwnedStorage<u8, R, C>,
          S1::Alloc: nalgebra::allocator::OwnedAllocator<f64, R, C, S1>,
          S2::Alloc: nalgebra::allocator::OwnedAllocator<u8, R, C, S2>,
{
    let label: Matrix<f64, R, C, S1> = convert_ref(label);
    let ln_output = output.map(|x|(x + 1e-7).ln());
    let y: f64 = label.component_mul(&ln_output).iter().sum();
    -y / label.nrows() as f64
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
            1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
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
}
