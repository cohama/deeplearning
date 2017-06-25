use ndarray::{Array, Array2};
use ndarray_rand::RandomExt;
use rand::distributions::normal::Normal;
use rand::{thread_rng, sample};

use functions::{sigmoid};
use matrix_functions::{cross_entropy_error, pick_rows, numerical_gradient_ndarray, softmax_rowwise};
use mnist::*;

struct TwoLayerNet {
    w1: Array2<f64>, // INPUT_SIZE, HIDDEN_SIZE>,
    b1: Array2<f64>, // HIDDEN_SIZE>,
    w2: Array2<f64>, // HIDDEN_SIZE, OUTPUT_SIZE>,
    b2: Array2<f64>, // OUTPUT_SIZE>,
    grad_w1: Array2<f64>, // INPUT_SIZE, HIDDEN_SIZE>,
    grad_b1: Array2<f64>, // HIDDEN_SIZE>,
    grad_w2: Array2<f64>, // HIDDEN_SIZE, OUTPUT_SIZE>,
    grad_b2: Array2<f64>, // OUTPUT_SIZE>,
}

const INPUT_SIZE: usize = 784;
const HIDDEN_SIZE: usize = 50;
const OUTPUT_SIZE: usize = 10;

impl TwoLayerNet {
    pub fn new() -> TwoLayerNet {
        TwoLayerNet {
            w1: Array::random((INPUT_SIZE, HIDDEN_SIZE), Normal::new(0., 1.)) * 0.01,
            // w1: Array::from_elem((INPUT_SIZE, HIDDEN_SIZE), 0.005),
            b1: Array::zeros((1, HIDDEN_SIZE)),
            w2: Array::random((HIDDEN_SIZE, OUTPUT_SIZE), Normal::new(0., 1.)) * 0.01,
            // w2: Array::from_elem((HIDDEN_SIZE, OUTPUT_SIZE), 0.005),
            b2: Array::zeros((1, OUTPUT_SIZE)),
            grad_w1: Array::zeros((INPUT_SIZE, HIDDEN_SIZE)),
            grad_b1: Array::zeros((1, HIDDEN_SIZE)),
            grad_w2: Array::zeros((HIDDEN_SIZE, OUTPUT_SIZE)),
            grad_b2: Array::zeros((1, OUTPUT_SIZE)),
        }
    }

    fn predict(&self, input: &Array2<f64>) -> Array2<f64> {
        let mut a1 = &input.dot(&self.w1) + &self.b1;
        a1.map_inplace(|x| *x = sigmoid(*x));
        let a2 = a1.dot(&self.w2) + &self.b2;
        softmax_rowwise(&a2)
    }

    #[inline]
    fn loss(&self, input: &Array2<f64>, label: &Array2<f64>) -> f64 {
        let output = self.predict(input);
        cross_entropy_error(&output, label)
    }

    #[inline]
    fn num_gradient(&mut self, input: &Array2<f64>, label: &Array2<f64>) {
        let pself = self as *const TwoLayerNet;
        let f = || unsafe {
            (*pself).loss(input, label)
        };
        numerical_gradient_ndarray(&f, &mut self.w1, &mut self.grad_w1);
        println!("    grad_w1");
        numerical_gradient_ndarray(&f, &mut self.b1, &mut self.grad_b1);
        println!("    grad_b1");
        numerical_gradient_ndarray(&f, &mut self.w2, &mut self.grad_w2);
        println!("    grad_w2");
        numerical_gradient_ndarray(&f, &mut self.b2, &mut self.grad_b2);
        println!("    grad_b2");
    }
}

pub fn run() {
    let data_nums = 60_000;
    let image_data = load_image("./data/mnist/train-images-idx3-ubyte", data_nums).unwrap();
    let images = Array::from_shape_vec((data_nums, INPUT_SIZE), image_data).unwrap();
    println!("train image data loaded. {:?}", images.shape());
    let labels_data = label_as_onehot(&load_label("./data/mnist/train-labels-idx1-ubyte", data_nums).unwrap());
    let labels = Array::from_shape_vec((data_nums, OUTPUT_SIZE), labels_data).unwrap();
    println!("train image label loaded. {:?}", labels.shape());

    // panic!();
    let mut train_losses = vec![];

    // hyper parameters
    let iter_nums = 10_000;
    // let train_size = images.nrows();
    let batch_size = 100;
    let learning_rate = 0.1;

    let mut net = TwoLayerNet::new();

    let mut rng = thread_rng();

    for i in 0..iter_nums {
        println!("iteration {} started.", i);
        let mut batch_mask = sample(&mut rng, 0..data_nums, batch_size);
        batch_mask.sort();
        // println!("  batch_mask generated {:?}", batch_mask);
        let input = pick_rows(&images, &batch_mask);
        let label = pick_rows(&labels, &batch_mask);

        println!("  batch masked");

        net.num_gradient(&input, &label);

        println!("  num grad");

        // let subw1 = net.grad_w1 * learning_rate;
        net.w1 -= &(&net.grad_w1 * learning_rate);
        net.w1 -= &(&net.grad_w1 * learning_rate);
        net.b1 -= &(&net.grad_b1 * learning_rate);
        net.w2 -= &(&net.grad_w2 * learning_rate);
        net.b2 -= &(&net.grad_b2 * learning_rate);

        let loss = net.loss(&input, &label);
        println!("  loss: {}", loss);
        train_losses.push(loss);
        println!("iteration {} finished.", i);
    }
    println!("{:?}", train_losses);
}
