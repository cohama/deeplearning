extern crate typenum;
extern crate nalgebra;

use self::typenum::{UInt, UTerm, B0, B1};

pub type U60000 = UInt<UInt<UInt<UInt<UInt<UInt<UInt<UInt<UInt<UInt<UInt<UInt<UInt<UInt<UInt<UInt<UTerm, B1>, B1>, B1>, B0>, B1>, B0>, B1>, B0>, B0>, B1>, B1>, B0>, B0>, B0>, B0>, B0>;


#[cfg(test)]
mod tests {
    use super::*;
    use super::typenum::Unsigned;

    #[test]
    #[allow(non_snake_case)]
    fn U60000_is_60000() {
        assert_eq!(U60000::to_u32(), 60_000);
    }
}
