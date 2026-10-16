use std::iter::Sum;
use std::ops::*;
use num::{Float, One, Zero};

#[repr(transparent)]
pub struct Vector<T, const N: usize>(
    pub [T; N],
);

pub type Vector2<T> = Vector<T, 2>;
pub type Vector3<T> = Vector<T, 3>;
pub type Vector4<T> = Vector<T, 4>;

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
        Self::from_fn(|_| Default::default())
    }
}

impl<T, const N: usize> Zero for Vector<T, N>
where
    T: Zero,
{
    fn zero() -> Self {
        Self::zero()
    }

    fn is_zero(&self) -> bool {
        self.0.iter().all(Zero::is_zero)
    }
}

impl<T, const N: usize> One for Vector<T, N>
where
    T: One,
{
    fn one() -> Self {
        Self::one()
    }
}

#[cfg(feature = "bytemuck")]
unsafe impl<T, const N: usize> bytemuck::Zeroable for Vector<T, N>
where
    T: bytemuck::Zeroable,
{}

#[cfg(feature = "bytemuck")]
unsafe impl<T, const N: usize> bytemuck::Pod for Vector<T, N>
where
    T: bytemuck::Pod,
{}

impl<T, const N: usize> From<[T; N]> for Vector<T, N> {
    fn from(components: [T; N]) -> Self {
        Self(components)
    }
}

impl<T, const N: usize> From<Vector<T, N>> for [T; N] {
    fn from(vector: Vector<T, N>) -> Self {
        vector.0
    }
}

impl<T, const N: usize> Vector<T, N> {
    pub fn from_fn<F>(f: F) -> Self
    where
        F: FnMut(usize) -> T,
    {
        Self(std::array::from_fn(f))
    }

    pub fn zero() -> Self
    where
        T: Zero,
    {
        Self::from_fn(|_| T::zero())
    }

    pub fn one() -> Self
    where
        T: One,
    {
        Self::from_fn(|_| T::one())
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

    pub fn map<U, F>(self, f: F) -> Vector<U, N>
    where
        F: FnMut(T) -> U,
    {
        Vector(self.0.map(f))
    }

    pub fn map_binary<Rhs, U, F>(self, rhs: Vector<Rhs, N>, mut f: F) -> Vector<U, N>
    where
        F: FnMut(T, Rhs) -> U,
    {
        let mut pairs = std::iter::zip(self.0, rhs.0);
        Vector::from_fn(|_| {
            // SAFETY: The size of both inputs and the output is N at compile time
            let (lhs, rhs) = unsafe { pairs.next().unwrap_unchecked() };
            f(lhs, rhs)
        })
    }

    pub fn mul<Rhs>(self, rhs: Rhs) -> Vector<<T as Mul<Rhs>>::Output, N>
    where
        Rhs: Copy,
        T: Mul<Rhs>,
    {
        self.map(|lhs| lhs * rhs)
    }

    pub fn pre_mul<Lhs>(self, lhs: Lhs) -> Vector<<Lhs as Mul<T>>::Output, N>
    where
        Lhs: Copy + Mul<T>,
    {
        self.map(|rhs| lhs * rhs)
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

    pub fn magnitude_squared(self) -> <T as Mul>::Output
    where
        T: Copy + Mul,
        <T as Mul>::Output: Sum,
    {
        self.dot(self)
    }

    pub fn magnitude(self) -> <T as Mul>::Output
    where
        T: Copy + Mul,
        <T as Mul>::Output: Float + Sum,
    {
        self.magnitude_squared().sqrt()
    }

    pub fn normalize(self) -> Self
    where
        T: Copy + Float + Sum,
    {
        self.div(self.magnitude())
    }

    pub fn equals_delta<D>(self, other: Self, delta: Vector<D, N>) -> bool
    where
        T: PartialOrd + Sub<Output = D>,
        D: PartialOrd,
    {
        std::iter::zip(self.0, other.0)
            .zip(delta.0)
            .all(|((a, b), d)| (if a > b { a - b } else { b - a }) <= d)
    }
}

impl<T> Vector2<T>
where
    T: Copy,
{
    pub const fn x(&self) -> T {
        self.0[0]
    }

    pub const fn y(&self) -> T {
        self.0[1]
    }

    pub const fn with_z(&self, z: T) -> Vector3<T> {
        Vector([self.x(), self.y(), z])
    }

    pub fn fixed_dot<Rhs>(self, rhs: Vector2<Rhs>) -> <<T as Mul<Rhs>>::Output as Add>::Output
    where
        Rhs: Copy,
        T: Mul<Rhs>,
        <T as Mul<Rhs>>::Output: Add,
    {
        self.x() * rhs.x() + self.y() * rhs.y()
    }

    pub fn cross<Rhs>(self, rhs: Vector2<Rhs>) -> <<T as Mul<Rhs>>::Output as Sub>::Output
    where
        Rhs: Copy,
        T: Mul<Rhs>,
        <T as Mul<Rhs>>::Output: Sub,
    {
        self.x() * rhs.y() - self.y() * rhs.x()
    }
}

impl<T> Vector2<T> {
    pub fn set_x(&mut self, x: T) {
        self.0[0] = x;
    }

    pub fn set_y(&mut self, y: T) {
        self.0[1] = y;
    }

    pub fn into_x(self) -> T {
        let [x, _] = self.0;
        x
    }

    pub fn into_y(self) -> T {
        let [_, y] = self.0;
        y
    }

    pub fn into_with_z(self, z: T) -> Vector3<T> {
        let [x, y] = self.0;
        Vector([x, y, z])
    }
}

impl<T> Vector3<T>
where
    T: Copy,
{
    pub const fn x(&self) -> T {
        self.0[0]
    }

    pub const fn y(&self) -> T {
        self.0[1]
    }

    pub const fn z(&self) -> T {
        self.0[2]
    }

    pub const fn xy(&self) -> Vector2<T> {
        Vector([self.x(), self.y()])
    }

    pub const fn with_w(&self, w: T) -> Vector4<T> {
        Vector([self.x(), self.y(), self.z(), w])
    }

    pub fn fixed_dot<Rhs>(self, rhs: Vector3<Rhs>) -> <<T as Mul<Rhs>>::Output as Add>::Output
    where
        Rhs: Copy,
        T: Mul<Rhs>,
        <T as Mul<Rhs>>::Output: Add,
        <<T as Mul<Rhs>>::Output as Add>::Output: Add<<T as Mul<Rhs>>::Output, Output = <<T as Mul<Rhs>>::Output as Add>::Output>,
    {
        self.xy().fixed_dot(rhs.xy()) + self.z() * rhs.z()
    }

    pub fn cross<Rhs>(self, rhs: Vector3<Rhs>) -> Vector3<<<T as Mul<Rhs>>::Output as Sub>::Output>
    where
        Rhs: Copy,
        T: Mul<Rhs>,
        <T as Mul<Rhs>>::Output: Sub,
    {
        Vector([
            self.y() * rhs.z() - self.z() * rhs.y(),
            self.z() * rhs.x() - self.x() * rhs.z(),
            self.x() * rhs.y() - self.y() * rhs.x(),
        ])
    }
}

impl<T> Vector3<T> {
    pub fn set_x(&mut self, x: T) {
        self.0[0] = x;
    }

    pub fn set_y(&mut self, y: T) {
        self.0[1] = y;
    }

    pub fn set_z(&mut self, z: T) {
        self.0[2] = z;
    }

    pub fn into_x(self) -> T {
        let [x, _, _] = self.0;
        x
    }

    pub fn into_y(self) -> T {
        let [_, y, _] = self.0;
        y
    }

    pub fn into_z(self) -> T {
        let [_, _, z] = self.0;
        z
    }

    pub fn into_xy(self) -> Vector2<T> {
        let [x, y, _] = self.0;
        Vector([x, y])
    }

    pub fn into_with_w(self, w: T) -> Vector4<T> {
        let [x, y, z] = self.0;
        Vector([x, y, z, w])
    }

    pub fn split_z(self) -> (Vector2<T>, T) {
        let [x, y, z] = self.0;
        (Vector([x, y]), z)
    }
}

impl<T> Vector4<T>
where
    T: Copy,
{
    pub const fn x(&self) -> T {
        self.0[0]
    }

    pub const fn y(&self) -> T {
        self.0[1]
    }

    pub const fn z(&self) -> T {
        self.0[2]
    }

    pub const fn w(&self) -> T {
        self.0[3]
    }

    pub const fn xyz(&self) -> Vector3<T> {
        Vector([self.x(), self.y(), self.z()])
    }

    pub fn fixed_dot<Rhs>(self, rhs: Vector4<Rhs>) -> <<T as Mul<Rhs>>::Output as Add>::Output
    where
        Rhs: Copy,
        T: Mul<Rhs>,
        <T as Mul<Rhs>>::Output: Add,
        <<T as Mul<Rhs>>::Output as Add>::Output: Add<<T as Mul<Rhs>>::Output, Output = <<T as Mul<Rhs>>::Output as Add>::Output>,
    {
        self.xyz().fixed_dot(rhs.xyz()) + self.w() * rhs.w()
    }
}

impl<T> Vector4<T> {
    pub fn set_x(&mut self, x: T) {
        self.0[0] = x;
    }

    pub fn set_y(&mut self, y: T) {
        self.0[1] = y;
    }

    pub fn set_z(&mut self, z: T) {
        self.0[2] = z;
    }

    pub fn set_w(&mut self, w: T) {
        self.0[3] = w;
    }

    pub fn into_x(self) -> T {
        let [x, _, _, _] = self.0;
        x
    }

    pub fn into_y(self) -> T {
        let [_, y, _, _] = self.0;
        y
    }

    pub fn into_z(self) -> T {
        let [_, _, z, _] = self.0;
        z
    }

    pub fn into_w(self) -> T {
        let [_, _, _, w] = self.0;
        w
    }

    pub fn into_xyz(self) -> Vector3<T> {
        let [x, y, z, _] = self.0;
        Vector([x, y, z])
    }

    pub fn split_w(self) -> (Vector3<T>, T) {
        let [x, y, z, w] = self.0;
        (Vector([x, y, z]), w)
    }
}

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
        self.map(Neg::neg)
    }
}

impl<T, const N: usize> Sum for Vector<T, N>
where
    T: Zero + Add<Output = T>,
{
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|lhs, rhs| lhs + rhs).unwrap_or(Vector::zero())
    }
}

impl<Lhs, Rhs, const N: usize> Add<Vector<Rhs, N>> for Vector<Lhs, N>
where
    Lhs: Add<Rhs>,
{
    type Output = Vector<<Lhs as Add<Rhs>>::Output, N>;

    fn add(self, rhs: Vector<Rhs, N>) -> Self::Output {
        self.map_binary(rhs, |lhs, rhs| lhs + rhs)
    }
}

impl<Lhs, Rhs, const N: usize> AddAssign<Vector<Rhs, N>> for Vector<Lhs, N>
where
    Lhs: AddAssign<Rhs>,
{
    fn add_assign(&mut self, rhs: Vector<Rhs, N>) {
        for (lhs, rhs) in std::iter::zip(&mut self.0, rhs.0) {
            *lhs += rhs;
        }
    }
}

impl<Lhs, Rhs, const N: usize> Sub<Vector<Rhs, N>> for Vector<Lhs, N>
where
    Lhs: Sub<Rhs>,
{
    type Output = Vector<<Lhs as Sub<Rhs>>::Output, N>;

    fn sub(self, rhs: Vector<Rhs, N>) -> Self::Output {
        self.map_binary(rhs, |lhs, rhs| lhs - rhs)
    }
}

impl<Lhs, Rhs, const N: usize> SubAssign<Vector<Rhs, N>> for Vector<Lhs, N>
where
    Lhs: SubAssign<Rhs>,
{
    fn sub_assign(&mut self, rhs: Vector<Rhs, N>) {
        for (lhs, rhs) in std::iter::zip(&mut self.0, rhs.0) {
            *lhs -= rhs;
        }
    }
}

impl<Lhs, Rhs, const N: usize> Mul<Vector<Rhs, N>> for Vector<Lhs, N>
where
    Lhs: Mul<Rhs>,
{
    type Output = Vector<<Lhs as Mul<Rhs>>::Output, N>;

    fn mul(self, rhs: Vector<Rhs, N>) -> Self::Output {
        self.map_binary(rhs, |lhs, rhs| lhs * rhs)
    }
}

impl<Lhs, Rhs, const N: usize> MulAssign<Vector<Rhs, N>> for Vector<Lhs, N>
where
    Lhs: MulAssign<Rhs>,
{
    fn mul_assign(&mut self, rhs: Vector<Rhs, N>) {
        for (lhs, rhs) in std::iter::zip(&mut self.0, rhs.0) {
            *lhs *= rhs;
        }
    }
}

impl<Lhs, Rhs, const N: usize> Div<Vector<Rhs, N>> for Vector<Lhs, N>
where
    Lhs: Div<Rhs>,
{
    type Output = Vector<<Lhs as Div<Rhs>>::Output, N>;

    fn div(self, rhs: Vector<Rhs, N>) -> Self::Output {
        self.map_binary(rhs, |lhs, rhs| lhs / rhs)
    }
}

impl<Lhs, Rhs, const N: usize> DivAssign<Vector<Rhs, N>> for Vector<Lhs, N>
where
    Lhs: DivAssign<Rhs>,
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
