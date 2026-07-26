//! Syntactically valid, semantically wrong.
//!
//! Deliberately passes Oracle tier 0 (it parses) and fails tier 1 (`cargo
//! check`), which is what makes it useful for testing that the ladder
//! escalates one rung and then stops.

pub fn add(a: u32, b: u32) -> u32 {
    "this is not a number"
}
