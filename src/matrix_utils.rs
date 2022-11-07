use core::fmt;
use na::{DMatrix, Scalar};




#[derive(Debug, Clone, Copy, PartialEq)]
struct LacedMatrixElement<T: Scalar>(T, T);

impl<T: Scalar> LacedMatrixElement<T> {
    fn new(a: T, b: T) -> Self {
        LacedMatrixElement(a, b)
    }
}

impl From<(f32, f32)> for LacedMatrixElement<f32> {
    fn from((a, b): (f32, f32)) -> Self {
        LacedMatrixElement::new(a, b)
    }
}

impl fmt::Display for LacedMatrixElement<f32> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.2}|{:.2}", self.0, self.1)
    }
}

impl std::ops::Add for LacedMatrixElement<f32> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        Self(self.0 + other.0, self.1 + other.1)
    }
}

impl num_traits::identities::Zero for LacedMatrixElement<f32> {
    fn zero() -> Self {
        LacedMatrixElement(0.0, 0.0)
    }

    fn is_zero(&self) -> bool {
        self.0 == 0.0 && self.1 == 0.0
    }

    fn set_zero(&mut self) {
        *self = num_traits::Zero::zero();
    }
}

trait Interlaceable<T: Scalar> {
    fn interlace(&self, other: &DMatrix<T>) -> DMatrix<LacedMatrixElement<T>>;
}

impl Interlaceable<f32> for DMatrix<f32> {
    fn interlace(&self, other: &DMatrix<f32>) -> DMatrix<LacedMatrixElement<f32>> {
        assert!(self.nrows() == other.nrows() && self.ncols() == other.ncols());
    
        // For each element in each matrix, create a tuple of the two elements 
        // and put them in a new matrix
        let mut result = DMatrix::<LacedMatrixElement<f32>>::zeros(self.nrows(), self.ncols());
        for i in 0..self.nrows() {
            for j in 0..self.ncols() {
                result[(i, j)] = LacedMatrixElement::new(self[(i, j)], other[(i, j)]);
            }
        }
    
        result
    }
}