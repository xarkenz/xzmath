use std::iter::Sum;
use std::mem::MaybeUninit;
use std::ops::*;
use num::{Float, One, Zero};
use num::traits::NumAssign;
use crate::*;

#[repr(transparent)]
pub struct Matrix<T, const R: usize, const C: usize>(
    pub [Vector<T, R>; C],
);

pub type Matrix2x2<T> = Matrix<T, 2, 2>;
pub type Matrix2x3<T> = Matrix<T, 2, 3>;
pub type Matrix2x4<T> = Matrix<T, 2, 4>;
pub type Matrix3x2<T> = Matrix<T, 3, 2>;
pub type Matrix3x3<T> = Matrix<T, 3, 3>;
pub type Matrix3x4<T> = Matrix<T, 3, 4>;
pub type Matrix4x2<T> = Matrix<T, 4, 2>;
pub type Matrix4x3<T> = Matrix<T, 4, 3>;
pub type Matrix4x4<T> = Matrix<T, 4, 4>;

impl<T, const R: usize, const C: usize> Copy for Matrix<T, R, C>
where
    T: Copy,
{}

impl<T, const R: usize, const C: usize> Clone for Matrix<T, R, C>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T, const R: usize, const C: usize> Default for Matrix<T, R, C>
where
    T: Default,
{
    fn default() -> Self {
        // Arrays implement Default manually for every length up to 32 because [T; 0] needs to
        // implement Default even if T doesn't implement it.
        // We can just kind of ignore that caveat lol who cares.
        Self::from_fn(|_, _| Default::default())
    }
}

#[cfg(feature = "bytemuck")]
unsafe impl<T, const R: usize, const C: usize> bytemuck::Zeroable for Matrix<T, R, C>
where
    T: bytemuck::Zeroable,
{}

#[cfg(feature = "bytemuck")]
unsafe impl<T, const R: usize, const C: usize> bytemuck::Pod for Matrix<T, R, C>
where
    T: bytemuck::Pod,
{}

impl<T, const R: usize, const C: usize> Matrix<T, R, C> {
    pub fn zero() -> Self
    where
        T: Zero,
    {
        Self::from_fn(|_, _| T::zero())
    }

    pub fn set_zero(&mut self)
    where
        T: Zero,
    {
        self.0.fill_with(Vector::zero);
    }

    pub fn all_ones() -> Self
    where
        T: One,
    {
        Self::from_fn(|_, _| T::one())
    }

    pub fn set_all_ones(&mut self)
    where
        T: One,
    {
        self.0.fill_with(Vector::one);
    }

    pub fn from_rows(rows: [Vector<T, C>; R]) -> Self {
        Matrix(rows).transpose()
    }

    pub fn from_column_fn<F>(f: F) -> Self
    where
        F: FnMut(usize) -> Vector<T, R>,
    {
        Self(std::array::from_fn(f))
    }

    pub fn from_row_fn<F>(f: F) -> Self
    where
        F: FnMut(usize) -> Vector<T, C>,
    {
        Matrix(std::array::from_fn(f)).transpose()
    }

    pub fn from_fn<F>(mut f: F) -> Self
    where
        F: FnMut(usize, usize) -> T,
    {
        Self::from_column_fn(|i| Vector::from_fn(|j| f(i, j)))
    }

    pub fn as_ptr(&self) -> *const T {
        self.0.as_ptr().cast()
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.0.as_mut_ptr().cast()
    }

    pub fn map_columns<U, const N: usize, F>(self, f: F) -> Matrix<U, N, C>
    where
        F: FnMut(Vector<T, R>) -> Vector<U, N>,
    {
        Matrix(self.0.map(f))
    }

    pub fn map<U, F>(self, mut f: F) -> Matrix<U, R, C>
    where
        F: FnMut(T) -> U,
    {
        self.map_columns(|column| column.map(&mut f))
    }

    pub fn mul<Rhs>(self, rhs: Rhs) -> Matrix<<T as Mul<Rhs>>::Output, R, C>
    where
        Rhs: Copy,
        T: Mul<Rhs>,
    {
        self.map(|lhs| lhs * rhs)
    }

    pub fn div<Rhs>(self, rhs: Rhs) -> Matrix<<T as Div<Rhs>>::Output, R, C>
    where
        Rhs: Copy,
        T: Div<Rhs>,
    {
        self.map(|lhs| lhs / rhs)
    }

    pub fn transpose(self) -> Matrix<T, C, R> {
        let mut transposed: Matrix<MaybeUninit<T>, C, R> = Matrix::from_fn(|_, _| MaybeUninit::uninit());

        for (i, column) in self.0.into_iter().enumerate() {
            for (j, element) in column.0.into_iter().enumerate() {
                transposed[j][i].write(element);
            }
        }

        transposed.map(|element| unsafe { element.assume_init() })
    }
}

impl<T, const N: usize> Matrix<T, N, N> {
    pub fn identity() -> Self
    where
        T: Copy + Zero + One,
    {
        let mut matrix = Self::zero();
        for n in 0..N {
            matrix[n][n] = T::one();
        }
        matrix
    }

    pub fn set_identity(&mut self)
    where
        T: Copy + Zero + One,
    {
        self.set_zero();
        for n in 0..N {
            self[n][n] = T::one();
        }
    }

    pub fn transpose_in_place(&mut self) {
        if N <= 1 {
            return;
        }
        for i in 0 .. N - 1 {
            for j in i .. N {
                let [col_i, col_j] = self.0.get_disjoint_mut([i, j]).unwrap();
                std::mem::swap(&mut col_i[j], &mut col_j[i]);
            }
        }
    }
}

impl<T> Matrix4x4<T>
where
    T: Float + NumAssign + Sum,
{
    pub fn affine(&self) -> Matrix3x3<T> {
        Matrix([
            self[0].xyz(),
            self[1].xyz(),
            self[2].xyz(),
        ])
    }

    pub fn rotate_x(&mut self, angle: T) {
        let zero = T::zero();
        let one = T::one();
        let (sin, cos) = angle.sin_cos();
        self.mul_assign(Self([
            Vector([one, zero, zero, zero]),
            Vector([zero, cos, -sin, zero]),
            Vector([zero, sin, cos, zero]),
            Vector([zero, zero, zero, one]),
        ]));
    }

    pub fn rotate_y(&mut self, angle: T) {
        let zero = T::zero();
        let one = T::one();
        let (sin, cos) = angle.sin_cos();
        self.mul_assign(Self([
            Vector([cos, zero, sin, zero]),
            Vector([zero, one, zero, zero]),
            Vector([-sin, zero, cos, zero]),
            Vector([zero, zero, zero, one]),
        ]));
    }

    pub fn rotate_z(&mut self, angle: T) {
        let zero = T::zero();
        let one = T::one();
        let (sin, cos) = angle.sin_cos();
        self.mul_assign(Self([
            Vector([cos, sin, zero, zero]),
            Vector([-sin, cos, zero, zero]),
            Vector([zero, zero, one, zero]),
            Vector([zero, zero, zero, one]),
        ]));
    }

    pub fn translate(&mut self, amount: Vector3<T>) {
        let to_add = self[0].mul(amount.x()) + self[1].mul(amount.y()) + self[2].mul(amount.z());
        self[3] += to_add;
    }

    pub fn scale(&mut self, amount: Vector3<T>) {
        self.scale_x(amount.x());
        self.scale_y(amount.y());
        self.scale_z(amount.z());
    }

    pub fn scale_x(&mut self, amount: T) {
        self[0] = self[0].mul(amount);
    }

    pub fn scale_y(&mut self, amount: T) {
        self[1] = self[1].mul(amount);
    }

    pub fn scale_z(&mut self, amount: T) {
        self[2] = self[2].mul(amount);
    }

    pub fn inverted(&self) -> Self {
        // Borrowed from JOML (github.com/JOML-CI/JOML)
        let a = self[0][0] * self[1][1] - self[0][1] * self[1][0];
        let b = self[0][0] * self[1][2] - self[0][2] * self[1][0];
        let c = self[0][0] * self[1][3] - self[0][3] * self[1][0];
        let d = self[0][1] * self[1][2] - self[0][2] * self[1][1];
        let e = self[0][1] * self[1][3] - self[0][3] * self[1][1];
        let f = self[0][2] * self[1][3] - self[0][3] * self[1][2];
        let g = self[2][0] * self[3][1] - self[2][1] * self[3][0];
        let h = self[2][0] * self[3][2] - self[2][2] * self[3][0];
        let i = self[2][0] * self[3][3] - self[2][3] * self[3][0];
        let j = self[2][1] * self[3][2] - self[2][2] * self[3][1];
        let k = self[2][1] * self[3][3] - self[2][3] * self[3][1];
        let l = self[2][2] * self[3][3] - self[2][3] * self[3][2];

        Self([
            Vector([
                l.mul_add(self[1][1], k.mul_add(-self[1][2], j * self[1][3])),
                l.mul_add(-self[0][1], k.mul_add(self[0][2], j * -self[0][3])),
                f.mul_add(self[3][1], e.mul_add(-self[3][2], d * self[3][3])),
                f.mul_add(-self[2][1], e.mul_add(self[2][2], d * -self[2][3])),
            ]),
            Vector([
                l.mul_add(-self[1][0], i.mul_add(self[1][2], h * -self[1][3])),
                l.mul_add(self[0][0], i.mul_add(-self[0][2], h * self[0][3])),
                f.mul_add(-self[3][0], c.mul_add(self[3][2], b * -self[3][3])),
                f.mul_add(self[2][0], c.mul_add(-self[2][2], b * self[2][3])),
            ]),
            Vector([
                k.mul_add(self[1][0], i.mul_add(-self[1][1], g * self[1][3])),
                k.mul_add(-self[0][0], i.mul_add(self[0][1], g * -self[0][3])),
                e.mul_add(self[3][0], c.mul_add(-self[3][1], a * self[3][3])),
                e.mul_add(-self[2][0], c.mul_add(self[2][1], a * -self[2][3])),
            ]),
            Vector([
                j.mul_add(-self[1][0], h.mul_add(self[1][1], g * -self[1][2])),
                j.mul_add(self[0][0], h.mul_add(-self[0][1], g * self[0][2])),
                d.mul_add(-self[3][0], b.mul_add(self[3][1], a * -self[3][2])),
                d.mul_add(self[2][0], b.mul_add(-self[2][1], a * self[2][2])),
            ]),
        ])
    }

    pub fn set_look_at(&mut self, eye: Vector3<T>, center: Vector3<T>, up: Vector3<T>) {
        let forward = (eye - center).normalized();
        let right = up.cross(forward).normalized();
        let up = forward.cross(right).normalized();

        self[0].0 = [right.x(), up.x(), forward.x(), T::zero()];
        self[1].0 = [right.y(), up.y(), forward.y(), T::zero()];
        self[2].0 = [right.z(), up.z(), forward.z(), T::zero()];
        self[3].0 = [-right.dot(eye), -up.dot(eye), -forward.dot(eye), T::one()];
    }

    pub fn new_look_at(eye: Vector3<T>, center: Vector3<T>, up: Vector3<T>) -> Self {
        let mut transform = Self::zero();
        transform.set_look_at(eye, center, up);
        transform
    }

    pub fn look_at(&mut self, eye: Vector3<T>, center: Vector3<T>, up: Vector3<T>) {
        let transform = Self::new_look_at(eye, center, up);
        *self *= transform;
    }

    pub fn orthographic(&mut self, left: T, right: T, bottom: T, top: T, near: T, far: T) {
        self.translate(Vector([
            (left + right) / (left - right),
            (bottom + top) / (bottom - top),
            (near + far) / (near - far),
        ]));
        let two = T::one() + T::one();
        self.scale(Vector([
            two / (right - left),
            two / (top - bottom),
            two / (near - far),
        ]));
    }

    pub fn frustum(&mut self, left: T, right: T, bottom: T, top: T, near: T, far: T) {
        let two = T::one() + T::one();
        let mut transform = Self::zero();
        transform[0][0] = two * near / (right - left);
        transform[1][1] = two * near / (top - bottom);
        transform[0][2] = (right + left) / (right - left);
        transform[1][2] = (top + bottom) / (top - bottom);
        transform[2][2] = (far + near) / (near - far);
        transform[3][2] = two * near * far / (near - far);
        transform[2][3] = -T::one();
        *self *= transform;
    }

    pub fn perspective(&mut self, field_of_view: T, aspect_ratio: T, near: T, far: T) {
        let two = T::one() + T::one();
        let scale = (field_of_view / two).to_radians().tan() * near;
        self.frustum(
            -scale * aspect_ratio,
            scale * aspect_ratio,
            -scale,
            scale,
            near,
            far,
        );
    }
}

impl<T, const R: usize, const C: usize> PartialEq for Matrix<T, R, C>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl<T, const R: usize, const C: usize> PartialOrd for Matrix<T, R, C>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<T, const R: usize, const C: usize> Eq for Matrix<T, R, C>
where
    T: Eq,
{}

impl<T, const R: usize, const C: usize> Ord for Matrix<T, R, C>
where
    T: Ord,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl<T, const R: usize, const C: usize> Index<usize> for Matrix<T, R, C> {
    type Output = Vector<T, R>;

    fn index(&self, index: usize) -> &Self::Output {
        self.0.index(index)
    }
}

impl<T, const R: usize, const C: usize> IndexMut<usize> for Matrix<T, R, C> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

impl<T, const R: usize, const C: usize> Neg for Matrix<T, R, C>
where
    T: Neg,
{
    type Output = Matrix<<T as Neg>::Output, R, C>;

    fn neg(self) -> Self::Output {
        Matrix(self.0.map(|v| -v))
    }
}

impl<Lhs, Rhs, const R: usize, const C: usize> Mul<Vector<Rhs, C>> for Matrix<Lhs, R, C>
where
    Lhs: Mul<Rhs>,
    Rhs: Copy,
    <Lhs as Mul<Rhs>>::Output: Zero + Sum,
{
    type Output = Vector<<Lhs as Mul<Rhs>>::Output, R>;

    fn mul(self, rhs: Vector<Rhs, C>) -> Self::Output {
        std::iter::zip(self.0, rhs.0)
            .map(|(lhs, rhs)| lhs.mul(rhs))
            .sum()
    }
}

impl<Lhs, Rhs, const R: usize, const N: usize, const C: usize> Mul<Matrix<Rhs, N, C>> for Matrix<Lhs, R, N>
where
    Lhs: Copy + Mul<Rhs>,
    Rhs: Copy,
    <Lhs as Mul<Rhs>>::Output: Zero + Sum,
{
    type Output = Matrix<<Lhs as Mul<Rhs>>::Output, R, C>;

    fn mul(self, rhs: Matrix<Rhs, N, C>) -> Self::Output {
        rhs.map_columns(|rhs| self * rhs)
    }
}

impl<Lhs, Rhs, const R: usize, const C: usize> MulAssign<Matrix<Rhs, C, C>> for Matrix<Lhs, R, C>
where
    Lhs: Copy + Zero + Mul<Rhs, Output = Lhs> + Sum,
    Rhs: Copy,
{
    fn mul_assign(&mut self, rhs: Matrix<Rhs, C, C>) {
        let lhs = *self;
        for col in 0..C {
            self[col] = lhs * rhs[col];
        }
    }
}

impl<T> Mul<Vector3<T>> for Matrix4x4<T>
where
    T: Copy + Add<Output = T> + Mul<Output = T> + Div<Output = T>,
{
    type Output = Vector3<T>;

    fn mul(self, rhs: Vector3<T>) -> Self::Output {
        let factor = self[0].w() * rhs.x() + self[1].w() * rhs.y() + self[2].w() * rhs.z() + self[3].w();
        Vector([
            (self[0].x() * rhs.x() + self[1].x() * rhs.y() + self[2].x() * rhs.z() + self[3].x()) / factor,
            (self[0].y() * rhs.x() + self[1].y() * rhs.y() + self[2].y() * rhs.z() + self[3].y()) / factor,
            (self[0].z() * rhs.x() + self[1].z() * rhs.y() + self[2].z() * rhs.z() + self[3].z()) / factor,
        ])
    }
}

impl<T, const R: usize, const C: usize> std::fmt::Debug for Matrix<T, R, C>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Matrix{:?}", self.0)
    }
}
