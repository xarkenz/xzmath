use std::ops::{Add, Mul, Sub};
use num::{Float, One, Zero};

pub type InterpolateFn<T, V = T> = fn(T, V, V) -> V;

pub trait Interpolate<T, V = T> : Fn(T, V, V) -> V {}

impl<T, V> Interpolate<T, V> for T
where
    T: Fn(T, V, V) -> V,
{}

pub fn jump<T, V>(interpolant: T, value_0: V, value_1: V) -> V
where
    T: Zero + PartialOrd,
{
    if interpolant <= T::zero() { value_0 } else { value_1 }
}

pub fn lerp<T>(interpolant: T, value_0: T, value_1: T) -> T
where
    T: Copy + One + Mul<Output = T> + Add<Output = T> + Sub<Output = T>,
{
    (T::one() - interpolant) * value_0 + interpolant * value_1
}

pub fn clamped_lerp<T>(interpolant: T, value_0: T, value_1: T) -> T
where
    T: Float,
{
    let t = interpolant.clamp(T::zero(), T::one());
    lerp(t, value_0, value_1)
}

pub fn smooth_step<T>(interpolant: T, value_0: T, value_1: T) -> T
where
    T: Float,
{
    let t = interpolant.clamp(T::zero(), T::one());
    let two = T::one() + T::one();
    let three = two + T::one();
    lerp((three - two * t) * t * t, value_0, value_1)
}

pub fn sine_in_out<T>(interpolant: T, value_0: T, value_1: T) -> T
where
    T: Float,
{
    let t = interpolant.clamp(T::zero(), T::one());
    let two = T::one() + T::one();
    let pi = (T::zero() - T::one()).acos();
    lerp((T::one() - (pi * t).cos()) / two, value_0, value_1)
}

pub fn sine_in<T>(interpolant: T, value_0: T, value_1: T) -> T
where
    T: Float,
{
    let t = interpolant.clamp(T::zero(), T::one());
    let half_pi = T::zero().acos();
    lerp(T::one() - (half_pi * t).cos(), value_0, value_1)
}

pub fn sine_out<T>(interpolant: T, value_0: T, value_1: T) -> T
where
    T: Float,
{
    let t = interpolant.clamp(T::zero(), T::one());
    let half_pi = T::zero().acos();
    lerp((half_pi * t).sin(), value_0, value_1)
}
