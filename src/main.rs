extern crate nalgebra;
extern crate deeplearning;

use nalgebra::*;

use deeplearning::functions::*;
use deeplearning::impls::ch03_forward_net as ch03;

fn main() {
    println!("{:?}", sigmoid(0.3));
    ch03::run();
}

