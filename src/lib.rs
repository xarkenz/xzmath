//! This crate provides infrastructure for performing various calculations, with a particular
//! emphasis on games and animation.

#[cfg(feature = "animation")]
pub mod animation;
pub mod bounds;
pub mod interpolate;
pub mod matrix;
#[cfg(feature = "noise")]
pub mod noise;
pub mod quaternion;
pub mod vector;

pub use bounds::*;
pub use matrix::*;
pub use quaternion::*;
pub use vector::*;

#[cfg(test)]
mod tests {
    #[allow(unused_imports)]
    use super::*;
}
