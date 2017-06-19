extern crate nalgebra;
extern crate typenum;


pub fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

fn f64max(xs: &[f64]) -> f64 {
    let mut max = ::std::f64::MIN;
    for &x in xs {
        if x > max {max = x}
    }
    max
}

pub fn softmax(xs: &[f64], ys: &mut [f64]) {
    // p69
    // softmax := exp(ak) / (Σexp(ai))
    //         => exp(ak - C) / (Σexp(ai - C))
    let c = f64max(xs);
    let expa: f64 = xs.iter().map(|x| (x - c).exp()).sum();
    for (x, y) in xs.iter().zip(ys.iter_mut()) {
        *y = (x - c).exp() / expa;
    }
}

pub fn softmax_v(xs: &[f64]) -> Vec<f64> {
    let mut v = vec![0.0; xs.len()];
    softmax(&xs, v.as_mut_slice());
    v
}

pub fn max_index(xs: &[f64]) -> usize {
    xs.iter().enumerate().fold((0, ::std::f64::MIN), |(maxi, max), (i, &x)| {
        if x > max { (i, x) } else { (maxi, max) }
    }).0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_index() {
        assert_eq!(max_index(&[]), 0);
        assert_eq!(max_index(&[1.0, 2.0, 1.5]), 1);
    }
}
