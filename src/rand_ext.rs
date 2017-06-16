extern crate rand;

use self::rand::distributions::{IndependentSample, Range};

pub fn random_range(low: usize, high: usize, count: usize) -> Vec<usize>
{
    let mut rng = rand::thread_rng();
    let candidates = Range::new(low, high);
    let mut indice = vec![];
    while indice.len() != count {
        let i = candidates.ind_sample(&mut rng);
        if indice.contains(&i) {
            continue;
        } else {
            indice.push(i);
        }
    }
    indice.sort();
    indice
}
