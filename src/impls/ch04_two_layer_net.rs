extern crate nalgebra;

use functions::{sigmoid, softmax_v};
use self::nalgebra::{DMatrix, RowDVector};

struct TwoLayerNet {
    input_size: usize,
    hidden_size: usize,
    output_size: usize,
    w1: DMatrix<f64>,
    b1: RowDVector<f64>,
    w2: DMatrix<f64>,
    b2: RowDVector<f64>,
}

impl TwoLayerNet {
    fn new(input_size: usize, hidden_size: usize, output_size: usize) -> TwoLayerNet {
        TwoLayerNet {
            input_size,
            hidden_size,
            output_size,
            w1: DMatrix::new_random(input_size, hidden_size),
            b1: RowDVector::from_element(hidden_size, 0.0),
            w2: DMatrix::new_random(hidden_size, output_size),
            b2: RowDVector::from_element(output_size, 0.0),
        }
    }

    fn predict(&self, input: &RowDVector<f64>) -> RowDVector<f64> {
        let a1 = input * &self.w1 + &self.b1;
        let z1 = a1.map(sigmoid);
        let a2 = z1 * &self.w2 + &self.b2;
        let y = softmax_v(a2.as_slice());
        RowDVector::from_iterator(self.output_size, y.into_iter())
        // RowDVector::from_data(y)
    }

    fn loss(&self, input: &RowDVector<f64>, teacher: &RowDVector<f64>) {
        let y = self.predict(input);

    }
}
