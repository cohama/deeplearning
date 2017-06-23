use ndarray::{Array2};
use functions::{softmax_v, numerical_gradient};
use matrix_functions::{cross_entropy_error};

struct SimpleNet {
    w: Array2<f64>,
}

impl SimpleNet {
    fn new(w: &[f64]) -> SimpleNet {
        SimpleNet {
            w: Array2::from_shape_vec((2, 3), w.to_vec()).unwrap()
        }
    }

    fn predict(&self, x: &Array2<f64>) -> Array2<f64> {
        x.dot(&self.w)
    }

    fn loss(&self, input: &Array2<f64>, label: &Array2<f64>) -> f64 {
        let output = self.predict(input);
        let y = softmax_v(output.as_slice().unwrap());
        let output = Array2::from_shape_vec((1, 3), y).unwrap();
        cross_entropy_error(&output, label)
    }

    fn gradient(&mut self, input: &Array2<f64>, label: &Array2<f64>) -> Array2<f64> {
        let mut v = vec![0.; self.w.len()];
        let pself = self as *const SimpleNet;
        let f = || unsafe {
            (*pself).loss(input, label)
        };
        numerical_gradient(&f, self.w.as_slice_mut().unwrap(), &mut v);
        Array2::from_shape_vec((2, 3), v).unwrap()
    }
}

pub fn run() {
    let mut net = SimpleNet::new(&[-0.25, 1.4, 1.5,
                                   0.72, 0.97, -0.041]);
    let input = array![[0.6, 0.9]];
    println!("pred:\n{}", net.predict(&input));
    let label = array![[0., 0., 1.]];
    println!("loss:\n{}", net.loss(&input, &label));
    println!("grad:\n{}", net.gradient(&input, &label));
}
