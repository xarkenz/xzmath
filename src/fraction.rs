use std::ops::*;
use num::One;

pub struct Fraction<N, D = N>(pub N, pub D);

impl<T> From<T> for Fraction<T>
where
    T: One,
{
    fn from(value: T) -> Self {
        Self(value, T::one())
    }
}

impl<N, D> Fraction<N, D> {
    pub const fn new(numerator: N, denominator: D) -> Self {
        Self(numerator, denominator)
    }

    pub fn compute(self) -> <N as Div<D>>::Output
    where
        N: Div<D>,
    {
        self.0 / self.1
    }

    pub fn inverse(self) -> Fraction<D, N> {
        Fraction(self.1, self.0)
    }
}

impl<N, D> Neg for Fraction<N, D>
where
    N: Neg,
{
    type Output = Fraction<<N as Neg>::Output, D>;

    fn neg(self) -> Self::Output {
        Fraction(-self.0, self.1)
    }
}

impl<LN, LD, RN, RD> Mul<Fraction<RN, RD>> for Fraction<LN, LD>
where
    LN: Mul<RN>,
    LD: Mul<RD>,
{
    type Output = Fraction<
        <LN as Mul<RN>>::Output,
        <LD as Mul<RD>>::Output,
    >;

    fn mul(self, rhs: Fraction<RN, RD>) -> Self::Output {
        Fraction(self.0 * rhs.0, self.1 * rhs.1)
    }
}

impl<LN, LD, RN, RD> Div<Fraction<RN, RD>> for Fraction<LN, LD>
where
    LN: Mul<RD>,
    LD: Mul<RN>,
{
    type Output = Fraction<
        <LN as Mul<RD>>::Output,
        <LD as Mul<RN>>::Output,
    >;

    fn div(self, rhs: Fraction<RN, RD>) -> Self::Output {
        Fraction(self.0 * rhs.1, self.1 * rhs.0)
    }
}

impl<LN, LD, RN, RD> Add<Fraction<RN, RD>> for Fraction<LN, LD>
where
    LN: Mul<RD>,
    LD: Copy + Mul<RN> + Mul<RD>,
    RD: Copy,
    <LN as Mul<RD>>::Output: Add<<LD as Mul<RN>>::Output>,
{
    type Output = Fraction<
        <<LN as Mul<RD>>::Output as Add<<LD as Mul<RN>>::Output>>::Output,
        <LD as Mul<RD>>::Output,
    >;

    fn add(self, rhs: Fraction<RN, RD>) -> Self::Output {
        Fraction(self.0 * rhs.1 + self.1 * rhs.0, self.1 * rhs.1)
    }
}

impl<LN, LD, RN, RD> Sub<Fraction<RN, RD>> for Fraction<LN, LD>
where
    LN: Mul<RD>,
    LD: Copy + Mul<RN> + Mul<RD>,
    RD: Copy,
    <LN as Mul<RD>>::Output: Sub<<LD as Mul<RN>>::Output>,
{
    type Output = Fraction<
        <<LN as Mul<RD>>::Output as Sub<<LD as Mul<RN>>::Output>>::Output,
        <LD as Mul<RD>>::Output,
    >;

    fn sub(self, rhs: Fraction<RN, RD>) -> Self::Output {
        Fraction(self.0 * rhs.1 - self.1 * rhs.0, self.1 * rhs.1)
    }
}
