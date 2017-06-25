use std::io::Result as IoResult;
use std::io::{Seek, SeekFrom, Read};
use std::fs::File;

pub fn load_image(path: &str, num_of_images: usize) -> IoResult<Vec<f64>> {
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(16))?;
    let mut v = vec![];
    file.take(784 * num_of_images as u64).read_to_end(&mut v)?;
    Ok(v.iter().map(|x| *x as f64).collect())
}

pub fn load_label(path: &str, num_of_images: usize) -> IoResult<Vec<u8>> {
    let mut file = File::open(path)?;
    file.seek(SeekFrom::Start(8))?;
    let mut v = vec![];
    file.take(num_of_images as u64).read_to_end(&mut v)?;
    Ok(v)
}

pub fn label_as_onehot(label: &[u8]) -> Vec<f64> {
    let mut out = vec![];
    for &x in label.iter() {
        let mut onehot = [0.; 10];
        onehot[x as usize] = 1.;
        out.extend_from_slice(&onehot);
    }
    out
}


#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_image() {
        let m = load_image("/home/cohama/proj/rust/deeplearning/data/mnist/train-images-idx3-ubyte", 10).unwrap();
        assert_eq!(m[152..156], [3., 18., 18., 18.]);
    }

    #[test]
    fn test_load_label() {
        let v = load_label("/home/cohama/proj/rust/deeplearning/data/mnist/train-labels-idx1-ubyte", 10).unwrap();
        assert_eq!(v, [5, 0, 4, 1, 9, 2, 1, 3, 1, 4]);
        let _ = label_as_onehot(&v);
    }
}
