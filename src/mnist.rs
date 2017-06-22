extern crate nalgebra;
extern crate typenum;

use self::typenum::{U784};
use self::nalgebra::{DVector, Matrix, MatrixVec, U10, Dynamic};
use std::io::Result as IoResult;
use std::io::{Seek, SeekFrom, Read};
use std::fs::File;

pub type MatrixXx784f64 = Matrix<f64, Dynamic, U784, MatrixVec<f64, Dynamic, U784>>;
pub type MatrixXx10f64 = Matrix<f64, Dynamic, U10, MatrixVec<f64, Dynamic, U10>>;
// pub struct Mnist {
//     pub train_image: MatrixRx784<u8>,
//     pub train_label: MatrixRx10<u8>,
//     pub test_image: MatrixRx784<u8>,
//     pub test_label: MatrixRx10<u8>,
// }

// pub fn load_mnist_data() -> Mnist {
// }


pub fn load_image(path: &str, num_of_images: usize) -> IoResult<MatrixXx784f64> {
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(16))?;
    let mut v = vec![];
    let _ = file.take(784 * num_of_images as u64).read_to_end(&mut v);
    Ok(MatrixXx784f64::from_row_slice(num_of_images, v.iter().map(|&x| {
        if x != 0 {println!("x: {}", x)}
        x as f64
    }).collect::<Vec<f64>>().as_slice()))
    // Ok(MatrixXx784f64::from_row_slice(num_of_images, v.iter().map(|x| *x as f64).collect::<Vec<f64>>().as_slice()))
}

pub fn load_label(path: &str, num_of_images: usize) -> IoResult<DVector<u8>> {
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(16))?;
    let mut v = vec![];
    let _ = file.take(num_of_images as u64).read_to_end(&mut v);
    Ok(DVector::from_column_slice(num_of_images, v.as_slice()))
}

pub fn label_as_onehot(label: &DVector<u8>) -> MatrixXx10f64 {
    let mut vv = vec![];
    for v in label.iter() {
        let mut onehot = [0.; 10];
        onehot[*v as usize] = 1.;
        vv.extend_from_slice(&mut onehot);
    }
    MatrixXx10f64::from_row_slice(label.nrows(), vv.as_slice())
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_image() {
        let m = load_image("/home/cohama/proj/rust/deeplearning/data/mnist/train-images-idx3-ubyte", 10).unwrap();
        assert_eq!(m[(0, 152)], 3.0);
        assert_eq!(m[(0, 153)], 18.0);
        assert_eq!(m[(0, 154)], 18.0);
        assert_eq!(m[(0, 155)], 18.0);
    }

    #[test]
    fn test_load_label() {
        let v = load_label("/home/cohama/proj/rust/deeplearning/data/mnist/train-labels-idx1-ubyte", 10).unwrap();
        assert_eq!(v.as_slice(), [1, 4, 3, 5, 3, 6, 1, 7, 2, 8]);
        let _ = label_as_onehot(&v);
    }
}
