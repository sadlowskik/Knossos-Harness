//! A small, valid crate. Oracle's ladder should run to completion on it.

pub struct Adder {
    pub base: u32,
}

impl Adder {
    pub fn new(base: u32) -> Self {
        Adder { base }
    }

    pub fn add(&self, x: u32) -> u32 {
        self.base + x
    }
}

pub fn double(x: u32) -> u32 {
    x * 2
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adder_adds() {
        assert_eq!(Adder::new(2).add(3), 5);
    }

    #[test]
    fn double_doubles() {
        assert_eq!(double(4), 8);
    }
}
