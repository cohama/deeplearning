extern crate nalgebra;
extern crate typenum;
extern crate num_traits;

use functions::{sigmoid, softmax_v};
use self::nalgebra::{DimName, Matrix, MatrixNM, RowVectorN, U10, U50};
use self::typenum::{U784};
use mnist::{cross_entropy_error};
use self::num_traits::identities::Zero;

type InputSize = U784;
type HiddenSize = U50;
type OutputSize = U10;

#[allow(dead_code)]
pub struct TwoLayerNet {
    w1: MatrixNM<f64, InputSize, HiddenSize>,
    b1: RowVectorN<f64, HiddenSize>,
    w2: MatrixNM<f64, HiddenSize, OutputSize>,
    b2: RowVectorN<f64, OutputSize>,
}

impl TwoLayerNet {
    pub fn new() -> TwoLayerNet {
        TwoLayerNet {
            w1: MatrixNM::new_random(),
            b1: RowVectorN::zero(),
            w2: MatrixNM::new_random(),
            b2: RowVectorN::zero(),
        }
    }

    pub fn predict(&self, input: &RowVectorN<f64, InputSize>) -> RowVectorN<f64, OutputSize> {
        let a1 = input * &self.w1 + &self.b1;
        let z1 = a1.map(sigmoid);
        let a2 = z1 * &self.w2 + &self.b2;
        let y = softmax_v(a2.as_slice());
        RowVectorN::from_iterator(y.into_iter())
        // RowDVector::from_data(y)
    }

    pub fn loss(&self, input: &RowVectorN<f64, InputSize>, label: &RowVectorN<f64, OutputSize>) -> f64 {
        let output = self.predict(input);
        cross_entropy_error(&output, &label)
    }

    fn num_gradient<R, C, S, F>(f: F, x: &Matrix<f64, R, C, S>)
        where R: DimName,
              C: DimName,
              S: nalgebra::storage::OwnedStorage<f64, R, C>,
              S::Alloc: nalgebra::allocator::OwnedAllocator<f64, R, C, S>,
              F: Fn(f64) -> f64
    {
        let h = 1e-5;
    }
}
