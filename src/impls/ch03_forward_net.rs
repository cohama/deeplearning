extern crate nalgebra;

use self::nalgebra::*;
use functions::{sigmoid};

pub struct ForwardNet {
    w1: Matrix2x3<f64>,
    b1: RowVector3<f64>,
    w2: Matrix3x2<f64>,
    b2: RowVector2<f64>,
    w3: Matrix2<f64>,
    b3: RowVector2<f64>,
}

impl ForwardNet {
    fn new() -> ForwardNet {
        ForwardNet {
            w1: Matrix2x3::from_row_slice(&[0.1, 0.3, 0.5,
                                            0.2, 0.4, 0.6]),
            b1: RowVector3::new(0.1, 0.2, 0.3),
            w2: Matrix3x2::from_row_slice(&[0.1, 0.4,
                                            0.2, 0.5,
                                            0.3, 0.6]),
            b2: RowVector2::new(0.1, 0.2),
            w3: Matrix2::from_row_slice(&[0.1, 0.3,
                                          0.2, 0.4]),
            b3: RowVector2::new(0.1, 0.2),
        }
    }

    fn forward(&self, input: &RowVector2<f64>) -> RowVector2<f64> {
        let a1 = input * self.w1 + self.b1;
        println!("a1: {:?}", a1);
        let z1 = a1.map(sigmoid);
        println!("z1: {:?}", z1);
        let a2 = z1 * self.w2 + self.b2;
        println!("a2: {:?}", a2);
        let z2 = a2.map(sigmoid);
        let a3 = z2 * self.w3 + self.b3;
        println!("a3: {:?}", a3);
        a3
    }
}

pub fn run() {
    let nw = ForwardNet::new();
    let input = RowVector2::new(1.0, 0.5);
    let output = nw.forward(&input);

    println!("{:?}", output);
}
