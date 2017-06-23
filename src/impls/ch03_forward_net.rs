use ndarray::{Array2};
use functions::{sigmoid};

struct ForwardNet {
    w1: Array2<f64>,
    b1: Array2<f64>,
    w2: Array2<f64>,
    b2: Array2<f64>,
    w3: Array2<f64>,
    b3: Array2<f64>,
}

impl ForwardNet {
    fn new() -> ForwardNet {
        ForwardNet {
            w1: array![[0.1, 0.3, 0.5],
                       [0.2, 0.4, 0.6]],
            b1: array![[0.1, 0.2, 0.3]],
            w2: array![[0.1, 0.4],
                       [0.2, 0.5],
                       [0.3, 0.6]],
            b2: array![[0.1, 0.2]],
            w3: array![[0.1, 0.3],
                       [0.2, 0.4]],
            b3: array![[0.1, 0.2]],
        }
    }

    fn forward(&self, input: &Array2<f64>) -> Array2<f64> {
        let a1 = input.dot(&self.w1) + &self.b1;
        println!("a1: {:?}", a1.shape());
        let z1 = a1.map(|&x| sigmoid(x));
        println!("b1: {:?}", self.b1.shape());
        let a2 = z1.dot(&self.w2) + &self.b2;
        let z2 = a2.map(|&x| sigmoid(x));
        let a3 = z2.dot(&self.w3) + &self.b3;
        a3
    }
}

pub fn run() {
    let nw = ForwardNet::new();
    let input = array![[1.0, 0.5]];
    let output = nw.forward(&input);

    println!("{:?}", output);
}
