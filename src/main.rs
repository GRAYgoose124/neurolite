
use na::{DMatrix};

mod matrix_utils;

struct ResizableNetwork<T> {
    input: Vec<T>,

    activations: DMatrix<T>,
    weights: DMatrix<T>,

    output: Vec<T>,
}

impl ResizableNetwork<f32> {
    pub fn new(input_size: usize, hidden_size: (usize, usize), output_size: usize) -> Self {
        let input = vec![0.0; input_size];
        let activations = DMatrix::<f32>::zeros(hidden_size.0, hidden_size.1);
        let weights = DMatrix::<f32>::zeros(hidden_size.0, hidden_size.1);
        let output = vec![0.0; output_size];

        ResizableNetwork {
            input,
            activations,
            weights,
            output,
        }
    }
}

impl Default for ResizableNetwork<f32> {
    fn default() -> Self {
        Self::new(2, (2, 2), 1)
    }
}






#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn d_sigmoid(x: f32) -> f32 {
    sigmoid(x) * (1.0 - sigmoid(x))
}




fn main() {
    let mut net = ResizableNetwork::default();

    println!("{net}");
}
