#[cfg(feature = "animation")]
mod animation;
mod bounds;
mod matrix;
mod vector;

#[cfg(feature = "animation")]
pub use animation::*;
pub use bounds::*;
pub use matrix::*;
pub use vector::*;

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
}
