
pub struct Matrix<T> {
    shape: (usize, usize),

}

impl Matrix<f32> {
    fn new(shape: (usize, usize)) -> Self {
        Self {
            shape,
        }
    }
}

impl Default for Matrix<f32> {
    fn default() -> Self {
        Self::new((1, 1))
    }
}