extern crate nalgebra;

use functions::{sigmoid, softmax_v};
use self::nalgebra::{DMatrix, DVector};

pub struct TwoLayerNet {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    w1: DMatrix<f64>,
    b1: DVector<f64>,
    w2: DMatrix<f64>,
    b2: DVector<f64>,
}

impl TwoLayerNet {
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> TwoLayerNet {
        TwoLayerNet {
            input_size,
            hidden_size,
            output_size,
            w1: DMatrix::new_random(input_size, hidden_size),
            b1: DVector::from_element(hidden_size, 0.0),
            w2: DMatrix::new_random(hidden_size, output_size),
            b2: DVector::from_element(output_size, 0.0),
        }
    }

    fn predict(&self, input: DVector<f64>) -> DVector<f64> {
        let a1 = input.tr_mul(&self.w1) + &self.b1;
        let z1 = a1.map(|x| sigmoid(x));
        let a2 = z1 * &self.w2 + &self.b2;
        let y = softmax_v(a2.as_slice());
        DVector::from_iterator(self.output_size, y.into_iter())
    }
}
