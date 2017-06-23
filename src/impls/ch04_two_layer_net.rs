extern crate nalgebra;
extern crate typenum;
extern crate num_traits;

use self::nalgebra::{Matrix, MatrixNM, RowVectorN, U10, U50, Dynamic, MatrixVec};
use self::typenum::{U784};
use self::num_traits::identities::Zero;

use functions::{sigmoid, softmax_v, cross_entropy_error, numerical_gradient, pick_rows, rowwise_add};
use mnist::*;
use rand_ext::random_range;

type InputSize = U784;
type HiddenSize = U50;
type OutputSize = U10;

type MatrixInput = Matrix<f64, Dynamic, InputSize, MatrixVec<f64, Dynamic, InputSize>>;
type MatrixOutput = Matrix<f64, Dynamic, OutputSize, MatrixVec<f64, Dynamic, OutputSize>>;

struct TwoLayerNet {
    w1: MatrixNM<f64, InputSize, HiddenSize>,
    b1: RowVectorN<f64, HiddenSize>,
    w2: MatrixNM<f64, HiddenSize, OutputSize>,
    b2: RowVectorN<f64, OutputSize>,
    grad_w1: MatrixNM<f64, InputSize, HiddenSize>,
    grad_b1: RowVectorN<f64, HiddenSize>,
    grad_w2: MatrixNM<f64, HiddenSize, OutputSize>,
    grad_b2: RowVectorN<f64, OutputSize>,
}

impl TwoLayerNet {
    pub fn new() -> TwoLayerNet {
        TwoLayerNet {
            w1: MatrixNM::new_random(),
            b1: RowVectorN::zero(),
            w2: MatrixNM::new_random(),
            b2: RowVectorN::zero(),
            grad_w1: MatrixNM::zero(),
            grad_b1: RowVectorN::zero(),
            grad_w2: MatrixNM::zero(),
            grad_b2: RowVectorN::zero(),
        }
    }

    fn predict(&self, input: &MatrixInput) -> MatrixOutput {
        let mut a1 = input * &self.w1;
        rowwise_add(&mut a1, &self.b1);
        let z1 = a1.map(sigmoid);
        let mut a2 = z1 * &self.w2;
        rowwise_add(&mut a2, &self.b2);
        let y = softmax_v(a2.as_slice());
        MatrixOutput::from_row_slice(input.nrows(), y.as_ref())
    }

    #[inline]
    fn loss(&self, input: &MatrixInput, label: &MatrixOutput) -> f64 {
        let output = self.predict(input);
        cross_entropy_error(&output, label)
    }

    #[inline]
    fn num_gradient(&mut self, input: &MatrixInput, label: &MatrixOutput) {
        let pself = self as *const TwoLayerNet;
        let f = || unsafe {
            (*pself).loss(input, label)
        };
        numerical_gradient(&f, self.w1.as_mut_slice(), self.grad_w1.as_mut_slice());
        println!("    grad_w1");
        numerical_gradient(&f, self.b1.as_mut_slice(), self.grad_b1.as_mut_slice());
        println!("    grad_b1");
        numerical_gradient(&f, self.w2.as_mut_slice(), self.grad_w2.as_mut_slice());
        println!("    grad_w2");
        numerical_gradient(&f, self.b2.as_mut_slice(), self.grad_b2.as_mut_slice());
        println!("    grad_b2");
    }
}

pub fn run() {
    let data_nums = 60_000;
    let images = load_image("./data/mnist/train-images-idx3-ubyte", data_nums).unwrap();
    println!("train image data loaded. {:?}", images.shape());
    let labels = label_as_onehot(&load_label("./data/mnist/train-labels-idx1-ubyte", data_nums).unwrap());
    println!("train image label loaded. {:?}", labels.shape());

    // panic!();
    let mut train_losses = vec![];

    // hyper parameters
    let iter_nums = 10_000;
    // let train_size = images.nrows();
    let batch_size = 100;
    let learning_rate = 0.1;

    let mut net = TwoLayerNet::new();

    for i in 0..iter_nums {
        println!("iteration {} started.", i);
        let batch_mask = random_range(0, data_nums, batch_size);
        println!("  batch_mask generated {:?}", batch_mask);
        let input = pick_rows(&images, &batch_mask);
        let label = pick_rows(&labels, &batch_mask);

        println!("  batch masked");

        net.num_gradient(&input, &label);

        println!("  num grad");

        net.w1 -= learning_rate * net.grad_w1;
        net.b1 -= learning_rate * net.grad_b1;
        net.w2 -= learning_rate * net.grad_w2;
        net.b2 -= learning_rate * net.grad_b2;

        let loss = net.loss(&input, &label);
        println!("  loss: {}", loss);
        train_losses.push(loss);
        println!("iteration {} finished.", i);
    }
    println!("{:?}", train_losses);
}
