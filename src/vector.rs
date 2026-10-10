use std::iter::Sum;
use std::ops::*;
use num::{Float, One, Zero};

#[repr(transparent)]
pub struct Vector<T, const N: usize>(
    pub [T; N],
);

impl<T, const N: usize> Vector<T, N> {
    pub fn zero() -> Self
    where
        T: Zero,
    {
        Self(std::array::from_fn(|_| T::zero()))
    }

    pub fn one() -> Self
    where
        T: One,
    {
        Self(std::array::from_fn(|_| T::one()))
    }

    pub const fn splat(value: T) -> Self
    where
        T: Copy,
    {
        Self([value; N])
    }

    pub fn as_ptr(&self) -> *const T {
        self.0.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.0.as_mut_ptr()
    }

    pub fn map<F, U>(self, f: F) -> Vector<U, N>
    where
        F: FnMut(T) -> U,
    {
        Vector(self.0.map(f))
    }

    pub fn mul<Rhs>(self, rhs: Rhs) -> Vector<<T as Mul<Rhs>>::Output, N>
    where
        Rhs: Copy,
        T: Mul<Rhs>,
    {
        self.map(|lhs| lhs * rhs)
    }

    pub fn div<Rhs>(self, rhs: Rhs) -> Vector<<T as Div<Rhs>>::Output, N>
    where
        Rhs: Copy,
        T: Div<Rhs>,
    {
        self.map(|lhs| lhs / rhs)
    }

    pub fn dot<Rhs>(self, rhs: Vector<Rhs, N>) -> <T as Mul<Rhs>>::Output
    where
        T: Mul<Rhs>,
        <T as Mul<Rhs>>::Output: Sum,
    {
        std::iter::zip(self.0, rhs.0)
            .map(|(lhs, rhs)| lhs * rhs)
            .sum()
    }
}

impl<T, const N: usize> Vector<T, N>
where
    T: Copy + Float + Sum,
{
    pub fn magnitude(&self) -> T {
        self.dot(*self).sqrt()
    }

    pub fn normalized(&self) -> Self {
        self.div(self.magnitude())
    }

    pub fn equals_delta(&self, other: Self, delta: Self) -> bool {
        std::iter::zip(&self.0, &other.0)
            .zip(&delta.0)
            .all(|((&a, &b), &d)| (a - b).abs() <= d)
    }

    pub fn lerp(&self, other: Self, t: T) -> Self {
        let mut result = Self::zero();
        for index in 0..N {
            result.0[index] = self.0[index] * (T::one() - t) + other.0[index] * t;
        }
        result
    }
}

impl<T> Vector<T, 2>
where
    T: Copy,
{
    pub const fn x(&self) -> T {
        self.0[0]
    }

    pub const fn set_x(&mut self, x: T) {
        self.0[0] = x;
    }

    pub const fn y(&self) -> T {
        self.0[1]
    }

    pub const fn set_y(&mut self, y: T) {
        self.0[1] = y;
    }

    pub const fn with_z(&self, z: T) -> Vector<T, 3> {
        Vector([self.x(), self.y(), z])
    }
}

impl<T> Vector<T, 3>
where
    T: Copy,
{
    pub const fn x(&self) -> T {
        self.0[0]
    }

    pub const fn set_x(&mut self, x: T) {
        self.0[0] = x;
    }

    pub const fn y(&self) -> T {
        self.0[1]
    }

    pub const fn set_y(&mut self, y: T) {
        self.0[1] = y;
    }

    pub const fn z(&self) -> T {
        self.0[2]
    }

    pub const fn set_z(&mut self, z: T) {
        self.0[2] = z;
    }

    pub const fn xy(&self) -> Vector<T, 2> {
        Vector([self.x(), self.y()])
    }

    pub const fn with_w(&self, w: T) -> Vector<T, 4> {
        Vector([self.x(), self.y(), self.z(), w])
    }
}

impl<T> Vector<T, 3>
where
    T: Copy + Mul,
    <T as Mul>::Output: Sub,
{
    pub fn cross(&self, rhs: Self) -> Vector<<<T as Mul>::Output as Sub>::Output, 3> {
        Vector([
            self.y() * rhs.z() - self.z() * rhs.y(),
            self.z() * rhs.x() - self.x() * rhs.z(),
            self.x() * rhs.y() - self.y() * rhs.x(),
        ])
    }
}

impl<T> Vector<T, 4>
where
    T: Copy,
{
    pub const fn x(&self) -> T {
        self.0[0]
    }

    pub const fn set_x(&mut self, x: T) {
        self.0[0] = x;
    }

    pub const fn y(&self) -> T {
        self.0[1]
    }

    pub const fn set_y(&mut self, y: T) {
        self.0[1] = y;
    }

    pub const fn z(&self) -> T {
        self.0[2]
    }

    pub const fn set_z(&mut self, z: T) {
        self.0[2] = z;
    }

    pub const fn w(&self) -> T {
        self.0[3]
    }

    pub const fn set_w(&mut self, w: T) {
        self.0[3] = w;
    }

    pub const fn xyz(&self) -> Vector<T, 3> {
        Vector([self.x(), self.y(), self.z()])
    }
}

impl<T, const N: usize> Copy for Vector<T, N>
where
    T: Copy,
{}

impl<T, const N: usize> Clone for Vector<T, N>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

impl<T, const N: usize> Default for Vector<T, N>
where
    T: Default,
{
    fn default() -> Self {
        // Arrays implement Default manually for every length up to 32 because [T; 0] needs to
        // implement Default even if T doesn't implement it.
        // We can just kind of ignore that caveat lol who cares.
        Self(std::array::from_fn(|_| Default::default()))
    }
}

unsafe impl<T, const N: usize> bytemuck::Zeroable for Vector<T, N>
where
    T: bytemuck::Zeroable,
{}

unsafe impl<T, const N: usize> bytemuck::Pod for Vector<T, N>
where
    T: bytemuck::Pod,
{}

impl<T, const N: usize> PartialEq for Vector<T, N>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl<T, const N: usize> PartialOrd for Vector<T, N>
where
    T: PartialOrd,
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}

impl<T, const N: usize> Eq for Vector<T, N>
where
    T: Eq,
{}

impl<T, const N: usize> Ord for Vector<T, N>
where
    T: Ord,
{
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

impl<T, const N: usize, I> Index<I> for Vector<T, N>
where
    [T; N]: Index<I>,
{
    type Output = <[T; N] as Index<I>>::Output;

    fn index(&self, index: I) -> &Self::Output {
        self.0.index(index)
    }
}

impl<T, const N: usize, I> IndexMut<I> for Vector<T, N>
where
    [T; N]: IndexMut<I>,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        self.0.index_mut(index)
    }
}

impl<T, const N: usize> Neg for Vector<T, N>
where
    T: Neg,
{
    type Output = Vector<<T as Neg>::Output, N>;

    fn neg(self) -> Self::Output {
        Vector(self.0.map(|x| -x))
    }
}

impl<T, const N: usize> Sum for Vector<T, N>
where
    T: Zero + Add<Output = T>,
{
    fn sum<I: Iterator<Item = Vector<T, N>>>(iter: I) -> Self {
        iter.reduce(|lhs, rhs| lhs + rhs).unwrap_or(Vector::zero())
    }
}

impl<T, const N: usize, Rhs> Add<Vector<Rhs, N>> for Vector<T, N>
where
    T: Add<Rhs>,
{
    type Output = Vector<<T as Add<Rhs>>::Output, N>;

    fn add(self, rhs: Vector<Rhs, N>) -> Self::Output {
        let mut pairs = std::iter::zip(self.0, rhs.0);
        Vector(std::array::from_fn(|_| {
            // SAFETY: The size of both input arrays and the output array is N at compile time
            let (lhs, rhs) = unsafe { pairs.next().unwrap_unchecked() };
            lhs + rhs
        }))
    }
}

impl<T, const N: usize, Rhs> AddAssign<Vector<Rhs, N>> for Vector<T, N>
where
    T: AddAssign<Rhs>,
{
    fn add_assign(&mut self, rhs: Vector<Rhs, N>) {
        for (lhs, rhs) in std::iter::zip(&mut self.0, rhs.0) {
            *lhs += rhs;
        }
    }
}

impl<T, const N: usize, Rhs> Sub<Vector<Rhs, N>> for Vector<T, N>
where
    T: Sub<Rhs>,
{
    type Output = Vector<<T as Sub<Rhs>>::Output, N>;

    fn sub(self, rhs: Vector<Rhs, N>) -> Self::Output {
        let mut pairs = std::iter::zip(self.0, rhs.0);
        Vector(std::array::from_fn(|_| {
            // SAFETY: The size of both input arrays and the output array is N at compile time
            let (lhs, rhs) = unsafe { pairs.next().unwrap_unchecked() };
            lhs - rhs
        }))
    }
}

impl<T, const N: usize, Rhs> SubAssign<Vector<Rhs, N>> for Vector<T, N>
where
    T: SubAssign<Rhs>,
{
    fn sub_assign(&mut self, rhs: Vector<Rhs, N>) {
        for (lhs, rhs) in std::iter::zip(&mut self.0, rhs.0) {
            *lhs -= rhs;
        }
    }
}

impl<T, const N: usize, Rhs> Mul<Vector<Rhs, N>> for Vector<T, N>
where
    T: Mul<Rhs>,
{
    type Output = Vector<<T as Mul<Rhs>>::Output, N>;

    fn mul(self, rhs: Vector<Rhs, N>) -> Self::Output {
        let mut pairs = std::iter::zip(self.0, rhs.0);
        Vector(std::array::from_fn(|_| {
            // SAFETY: The size of both input arrays and the output array is N at compile time
            let (lhs, rhs) = unsafe { pairs.next().unwrap_unchecked() };
            lhs * rhs
        }))
    }
}

impl<T, const N: usize, Rhs> MulAssign<Vector<Rhs, N>> for Vector<T, N>
where
    T: MulAssign<Rhs>,
{
    fn mul_assign(&mut self, rhs: Vector<Rhs, N>) {
        for (lhs, rhs) in std::iter::zip(&mut self.0, rhs.0) {
            *lhs *= rhs;
        }
    }
}

impl<T, const N: usize, Rhs> Div<Vector<Rhs, N>> for Vector<T, N>
where
    T: Div<Rhs>,
{
    type Output = Vector<<T as Div<Rhs>>::Output, N>;

    fn div(self, rhs: Vector<Rhs, N>) -> Self::Output {
        let mut pairs = std::iter::zip(self.0, rhs.0);
        Vector(std::array::from_fn(|_| {
            // SAFETY: The size of both input arrays and the output array is N at compile time
            let (lhs, rhs) = unsafe { pairs.next().unwrap_unchecked() };
            lhs / rhs
        }))
    }
}

impl<T, const N: usize, Rhs> DivAssign<Vector<Rhs, N>> for Vector<T, N>
where
    T: DivAssign<Rhs>,
{
    fn div_assign(&mut self, rhs: Vector<Rhs, N>) {
        for (lhs, rhs) in std::iter::zip(&mut self.0, rhs.0) {
            *lhs /= rhs;
        }
    }
}

impl<T, const N: usize> std::fmt::Debug for Vector<T, N>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Vector{:?}", self.0)
    }
}
