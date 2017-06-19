extern crate nalgebra;

use self::nalgebra::{MatrixNM, RowVectorN, U2, U3};
use functions::{softmax_v, cross_entropy_error};

struct SimpleNet {
    w: MatrixNM<f64, U2, U3>,
}

impl SimpleNet {
    fn new(w: &[f64]) -> SimpleNet {
        SimpleNet {
            w: MatrixNM::from_row_slice(w)
        }
    }

    fn predict(&self, x: &RowVectorN<f64, U2>) -> RowVectorN<f64, U3> {
        let z = x * self.w;
        println!("{}", z);
        let y = softmax_v(z.as_slice());
        RowVectorN::from_iterator(y.into_iter())
    }

    #[allow(dead_code)]
    fn loss(&self, input: &RowVectorN<f64, U2>, label: &RowVectorN<f64, U3>) -> f64 {
        let output = self.predict(input);
        cross_entropy_error(&output, &label)
    }
}

pub fn run() {
    let net = SimpleNet::new(&[-0.14788432, -0.95472593,  1.59780929,
                                0.12348974,  0.49662969, -0.01502497]);
    let input = RowVectorN::from_row_slice(&[0.6, 0.9]);
    println!("{}", net.predict(&input));
    let label = RowVectorN::from_row_slice(&[0.0, 0.0, 1.0]);
    println!("{}", net.loss(&input, &label));
}
