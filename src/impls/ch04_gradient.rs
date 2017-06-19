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
        let z = x * self.w;
        let y = softmax_v(z.as_slice());
        RowVectorN::from_iterator(y.into_iter())
    }

    fn loss(&self, input: &RowVectorN<f64, U2>, label: &RowVectorN<f64, U3>) -> f64 {
        let output = self.predict(input);
        cross_entropy_error(&output, &label)
    }

    fn gradient(&self, input: &RowVectorN<f64, U2>, label: &RowVectorN<f64, U3>) -> MatrixNM<f64, U2, U3> {
        let f = |w: &MatrixNM<f64, U2, U3>| {
            let net = SimpleNet {w: *w};
            net.loss(input, label)
        };
        numerical_gradient(f, &self.w)
    }
}

pub fn run() {
    let net = SimpleNet::new(&[-0.25552802,  1.43770442,  1.49892302,
                                0.71706175,  0.9710557 , -0.40855739]);
    let input = RowVectorN::from_row_slice(&[0.6, 0.9]);
    println!("pred:\n{}", net.predict(&input));
    let label = RowVectorN::from_row_slice(&[0.0, 0.0, 1.0]);
    println!("loss:\n{}", net.loss(&input, &label));
    println!("grad:\n{}", net.gradient(&input, &label));
}
