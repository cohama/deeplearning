extern crate nalgebra;

use self::nalgebra::{MatrixNM, RowVectorN, U2, U3};
use functions::{softmax_v, cross_entropy_error, numerical_gradient};

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
        x * self.w
    }

    fn loss(&self, input: &RowVectorN<f64, U2>, label: &RowVectorN<f64, U3>) -> f64 {
        let output = self.predict(input);
        let y = softmax_v(output.as_slice());
        let output = RowVectorN::<f64, U3>::from_row_slice(y.as_slice());
        cross_entropy_error(&output, &label)
    }

    fn gradient(&mut self, input: &RowVectorN<f64, U2>, label: &RowVectorN<f64, U3>) -> MatrixNM<f64, U2, U3> {
        let mut v = vec![0.; self.w.as_slice().len()];
        let pself = self as *const SimpleNet;
        let f = || unsafe {
            (*pself).loss(input, label)
        };
        numerical_gradient(&f, self.w.as_mut_slice(), &mut v);
        MatrixNM::<f64, U2, U3>::from_column_slice(&v)
    }
}

pub fn run() {
    let mut net = SimpleNet::new(&[-0.25, 1.4, 1.5,
                                   0.72, 0.97, -0.041]);
    let input = RowVectorN::from_row_slice(&[0.6, 0.9]);
    println!("pred:\n{}", net.predict(&input));
    let label = RowVectorN::from_row_slice(&[0., 0., 1.]);
    println!("loss:\n{}", net.loss(&input, &label));
    println!("grad:\n{}", net.gradient(&input, &label));
}
