use std::ops::*;
use num::{Float, One, Zero};
use crate::{Vector3, Vector4};

#[repr(transparent)]
pub struct Quaternion<T>(
    pub Vector4<T>,
);

impl<T> Copy for Quaternion<T>
where
    T: Copy,
{}

impl<T> Clone for Quaternion<T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T> Default for Quaternion<T>
where
    T: Default,
{
    fn default() -> Self {
        Self(Vector4::default())
    }
}

impl<T> Zero for Quaternion<T>
where
    T: Zero,
{
    fn zero() -> Self {
        Self::zero()
    }

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }
}

#[cfg(feature = "bytemuck")]
unsafe impl<T> bytemuck::Zeroable for Quaternion<T>
where
    T: bytemuck::Zeroable,
{}

#[cfg(feature = "bytemuck")]
unsafe impl<T> bytemuck::Pod for Quaternion<T>
where
    T: bytemuck::Pod,
{}

impl<T> From<Vector3<T>> for Quaternion<T>
where
    T: Zero,
{
    fn from(imag: Vector3<T>) -> Self {
        Self::from_parts(T::zero(), imag)
    }
}

impl<T> Quaternion<T> {
    pub fn from_parts(real: T, imag: Vector3<T>) -> Self {
        Self(imag.into_with_w(real))
    }

    pub fn zero() -> Self
    where
        T: Zero,
    {
        Self(Vector4::zero())
    }

    pub fn identity() -> Self
    where
        T: Zero + One,
    {
        Self::from_parts(T::one(), Vector3::zero())
    }

    pub fn rotation(axis: Vector3<T>, angle: T) -> Self
    where
        T: Float,
    {
        let (sin, cos) = (angle / (T::one() + T::one())).sin_cos();
        Self::from_parts(cos, axis.normalize().mul(sin))
    }

    pub fn into_real(self) -> T {
        self.0.into_w()
    }

    pub fn into_imag(self) -> Vector3<T> {
        self.0.into_xyz()
    }

    pub const fn real(&self) -> T
    where
        T: Copy,
    {
        self.0.w()
    }

    pub const fn imag(&self) -> Vector3<T>
    where
        T: Copy,
    {
        self.0.xyz()
    }

    pub fn into_parts(self) -> (T, Vector3<T>) {
        let (imag, real) = self.0.split_w();
        (real, imag)
    }

    pub fn map<U, F>(self, f: F) -> Quaternion<U>
    where
        F: FnMut(T) -> U,
    {
        Quaternion(self.0.map(f))
    }

    pub fn map_binary<Rhs, U, F>(self, rhs: Quaternion<Rhs>, f: F) -> Quaternion<U>
    where
        F: FnMut(T, Rhs) -> U,
    {
        Quaternion(self.0.map_binary(rhs.0, f))
    }

    pub fn mul<Rhs>(self, rhs: Rhs) -> Quaternion<<T as Mul<Rhs>>::Output>
    where
        Rhs: Copy,
        T: Mul<Rhs>,
    {
        Quaternion(self.0.mul(rhs))
    }

    pub fn pre_mul<Lhs>(self, lhs: Lhs) -> Quaternion<<Lhs as Mul<T>>::Output>
    where
        Lhs: Copy + Mul<T>,
    {
        Quaternion(self.0.pre_mul(lhs))
    }

    pub fn div<Rhs>(self, rhs: Rhs) -> Quaternion<<T as Div<Rhs>>::Output>
    where
        Rhs: Copy,
        T: Div<Rhs>,
    {
        Quaternion(self.0.div(rhs))
    }

    pub fn conj(self) -> Self
    where
        T: Neg<Output = T>,
    {
        let (real, imag) = self.into_parts();
        Self::from_parts(real, -imag)
    }

    pub fn dot<Rhs>(self, rhs: Quaternion<Rhs>) -> <T as Mul<Rhs>>::Output
    where
        T: Mul<Rhs>,
        <T as Mul<Rhs>>::Output: Zero + Add<Output = <T as Mul<Rhs>>::Output>,
    {
        self.0.dot(rhs.0)
    }

    pub fn norm_squared(self) -> <T as Mul>::Output
    where
        T: Copy + Mul,
        <T as Mul>::Output: Zero + Add<Output = <T as Mul>::Output>,
    {
        self.dot(self)
    }

    pub fn norm(self) -> <T as Mul>::Output
    where
        T: Copy + Mul,
        <T as Mul>::Output: Float,
    {
        self.norm_squared().sqrt()
    }

    pub fn normalize(self) -> Self
    where
        T: Float,
    {
        self.div(self.norm())
    }

    pub fn inverse(self) -> Self
    where
        T: Float,
    {
        self.conj().div(self.norm_squared())
    }

    pub fn cos_angle_between(self, other: Quaternion<T>) -> T
    where
        T: Float,
    {
        (self.dot(other) / (self.norm() * other.norm())).clamp(-T::one(), T::one())
    }

    pub fn angle_between(self, other: Self) -> T
    where
        T: Float,
    {
        self.cos_angle_between(other).acos()
    }

    pub fn rotate_vector(self, vector: Vector3<T>) -> Vector3<T>
    where
        T: Copy + Zero + Mul<Output = T> + Add<Output = T> + Sub<Output = T>,
    {
        let (real, imag) = self.into_parts();

        let vector_mul = real * real - imag.dot(imag);
        let dot = imag.dot(vector);
        vector.mul(vector_mul) + imag.mul(dot + dot) + imag.cross(vector).mul(real + real)
    }
}

impl<T> std::iter::Sum for Quaternion<T>
where
    T: Zero + Add<Output = T>,
{
    fn sum<I: Iterator<Item = Quaternion<T>>>(iter: I) -> Self {
        Quaternion(iter.map(|item| item.0).sum())
    }
}

impl<Lhs, Rhs> Add<Quaternion<Rhs>> for Quaternion<Lhs>
where
    Lhs: Add<Rhs>,
{
    type Output = Quaternion<<Lhs as Add<Rhs>>::Output>;

    fn add(self, rhs: Quaternion<Rhs>) -> Self::Output {
        Quaternion(self.0 + rhs.0)
    }
}

impl<Lhs, Rhs> AddAssign<Quaternion<Rhs>> for Quaternion<Lhs>
where
    Lhs: AddAssign<Rhs>,
{
    fn add_assign(&mut self, rhs: Quaternion<Rhs>) {
        self.0 += rhs.0;
    }
}

impl<Lhs, Rhs> Mul<Quaternion<Rhs>> for Quaternion<Lhs>
where
    Lhs: Copy + Mul<Rhs>,
    Rhs: Copy,
    <Lhs as Mul<Rhs>>::Output: Zero + Add<Output = <Lhs as Mul<Rhs>>::Output> + Sub<Output = <Lhs as Mul<Rhs>>::Output>,
{
    type Output = Quaternion<<Lhs as Mul<Rhs>>::Output>;

    fn mul(self, rhs: Quaternion<Rhs>) -> Self::Output {
        let (lhs_real, lhs_imag) = self.into_parts();
        let (rhs_real, rhs_imag) = rhs.into_parts();

        Quaternion::from_parts(
            lhs_real * rhs_real - lhs_imag.dot(rhs_imag),
            rhs_imag.pre_mul(lhs_real) + lhs_imag.mul(rhs_real) + lhs_imag.cross(rhs_imag),
        )
    }
}

impl<T> std::fmt::Debug for Quaternion<T>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Quaternion{:?}", self.0)
    }
}
