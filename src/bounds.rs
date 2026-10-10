use std::ops::*;
use num::{One, Zero};
use crate::*;

pub struct Bounds<T, const N: usize> {
    pub min: Vector<T, N>,
    pub max: Vector<T, N>,
}

pub type Rectangle<T> = Bounds<T, 2>;
pub type Cuboid<T> = Bounds<T, 3>;

impl<T, const N: usize> Bounds<T, N> {
    pub const fn new(min: Vector<T, N>, max: Vector<T, N>) -> Self {
        Self {
            min,
            max,
        }
    }

    pub fn from_span(min: Vector<T, N>, span: Vector<T, N>) -> Self
    where
        T: Copy + Add<Output = T>,
    {
        Self {
            min,
            max: min + span,
        }
    }

    pub fn span(&self) -> Vector<T, N>
    where
        T: Copy + Sub<Output = T>,
    {
        self.max - self.min
    }

    pub fn axis_span(&self, axis: usize) -> T
    where
        T: Copy + Sub<Output = T>,
    {
        self.max[axis] - self.min[axis]
    }

    pub fn flip(&mut self) {
        std::mem::swap(&mut self.min, &mut self.max);
    }

    pub fn flip_axis(&mut self, axis: usize) {
        std::mem::swap(&mut self.min[axis], &mut self.max[axis]);
    }

    pub fn center(&self) -> Vector<T, N>
    where
        T: Copy + One + Add<Output = T> + Div<Output = T>
    {
        (self.min + self.max).div(T::one() + T::one())
    }

    pub fn shift_by(&mut self, amount: Vector<T, N>)
    where
        T: Copy + AddAssign,
    {
        self.min += amount;
        self.max += amount;
    }

    pub fn shift_axis_by(&mut self, axis: usize, amount: T)
    where
        T: Copy + AddAssign,
    {
        self.min[axis] += amount;
        self.max[axis] += amount;
    }

    pub fn shift_min_to(&mut self, target: Vector<T, N>)
    where
        T: Copy + AddAssign + Sub<Output = T>,
    {
        self.max += target - self.min;
        self.min = target;
    }

    pub fn shift_min_axis_to(&mut self, axis: usize, target: T)
    where
        T: Copy + AddAssign + Sub<Output = T>,
    {
        self.max[axis] += target - self.min[axis];
        self.min[axis] = target;
    }

    pub fn shift_max_to(&mut self, target: Vector<T, N>)
    where
        T: Copy + AddAssign + Sub<Output = T>,
    {
        self.min += target - self.max;
        self.max = target;
    }

    pub fn shift_max_axis_to(&mut self, axis: usize, target: T)
    where
        T: Copy + AddAssign + Sub<Output = T>,
    {
        self.min[axis] += target - self.max[axis];
        self.max[axis] = target;
    }

    pub fn expand_toward(&mut self, amount: Vector<T, N>)
    where
        T: Copy + Zero + AddAssign + PartialOrd,
    {
        for axis in 0..N {
            self.expand_axis_toward(axis, amount[axis]);
        }
    }

    pub fn expand_axis_toward(&mut self, axis: usize, amount: T)
    where
        T: Copy + Zero + AddAssign + PartialOrd,
    {
        if amount >= T::zero() {
            self.max[axis] += amount;
        }
        else {
            self.min[axis] += amount;
        }
    }

    pub fn contains_inclusive(&self, point: Vector<T, N>) -> bool
    where
        T: PartialOrd,
    {
        std::iter::zip(&self.min.0, &self.max.0)
            .zip(&point.0)
            .all(|((min, max), value)| value >= min && value <= max)
    }

    pub fn contains_exclusive(&self, point: Vector<T, N>) -> bool
    where
        T: PartialOrd,
    {
        std::iter::zip(&self.min.0, &self.max.0)
            .zip(&point.0)
            .all(|((min, max), value)| value > min && value < max)
    }

    pub fn intersects_inclusive(&self, other: &Self) -> bool
    where
        T: PartialOrd,
    {
        std::iter::zip(
            std::iter::zip(&self.min.0, &self.max.0),
            std::iter::zip(&other.min.0, &other.max.0),
        )
            .all(|((self_min, self_max), (other_min, other_max))| {
                self_max > other_min && self_min < other_max
            })
    }

    pub fn intersects_exclusive(&self, other: &Self) -> bool
    where
        T: PartialOrd,
    {
        std::iter::zip(
            std::iter::zip(&self.min.0, &self.max.0),
            std::iter::zip(&other.min.0, &other.max.0),
        )
            .all(|((self_min, self_max), (other_min, other_max))| {
                self_max > other_min && self_min < other_max
            })
    }
}

impl<T> Rectangle<T> {
    pub fn min_x_max_y(&self) -> Vector2<T>
    where
        T: Copy,
    {
        Vector([self.min.x(), self.max.y()])
    }

    pub fn max_x_min_y(&self) -> Vector2<T>
    where
        T: Copy,
    {
        Vector([self.max.x(), self.min.y()])
    }

    pub fn x_span(&self) -> T
    where
        T: Copy + Sub<Output = T>,
    {
        self.axis_span(0)
    }

    pub fn y_span(&self) -> T
    where
        T: Copy + Sub<Output = T>,
    {
        self.axis_span(1)
    }

    pub fn flip_x(&mut self) {
        self.flip_axis(0);
    }

    pub fn flip_y(&mut self) {
        self.flip_axis(1);
    }

    pub fn shift_x_by(&mut self, amount: T)
    where
        T: Copy + AddAssign,
    {
        self.shift_axis_by(0, amount);
    }

    pub fn shift_y_by(&mut self, amount: T)
    where
        T: Copy + AddAssign,
    {
        self.shift_axis_by(1, amount);
    }

    pub fn shift_min_x_to(&mut self, target: T)
    where
        T: Copy + AddAssign + Sub<Output = T>,
    {
        self.shift_min_axis_to(0, target);
    }

    pub fn shift_min_y_to(&mut self, target: T)
    where
        T: Copy + AddAssign + Sub<Output = T>,
    {
        self.shift_min_axis_to(1, target);
    }

    pub fn shift_max_x_to(&mut self, target: T)
    where
        T: Copy + AddAssign + Sub<Output = T>,
    {
        self.shift_max_axis_to(0, target);
    }

    pub fn shift_max_y_to(&mut self, target: T)
    where
        T: Copy + AddAssign + Sub<Output = T>,
    {
        self.shift_max_axis_to(1, target);
    }

    pub fn expand_x_toward(&mut self, amount: T)
    where
        T: Copy + Zero + AddAssign + PartialOrd,
    {
        self.expand_axis_toward(0, amount);
    }

    pub fn expand_y_toward(&mut self, amount: T)
    where
        T: Copy + Zero + AddAssign + PartialOrd,
    {
        self.expand_axis_toward(1, amount);
    }
}

impl<T, const N: usize> Copy for Bounds<T, N>
where
    T: Copy,
{}

impl<T, const N: usize> Clone for Bounds<T, N>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Self::new(self.min.clone(), self.max.clone())
    }
}

impl<T, const N: usize> PartialEq for Bounds<T, N>
where
    T: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.min == other.min && self.max == other.max
    }
}

impl<T, const N: usize> Eq for Bounds<T, N>
where
    T: Eq,
{}

impl<T, const N: usize> std::fmt::Debug for Bounds<T, N>
where
    T: std::fmt::Debug,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Bounds")
            .field("min", &self.min)
            .field("max", &self.max)
            .finish()
    }
}
