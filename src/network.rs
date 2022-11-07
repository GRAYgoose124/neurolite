use std::fmt;

use na::{DMatrix};


pub struct ResizableNetwork<T> {
    pub input: Vec<T>,
    pub activations: DMatrix<T>,
    pub output: Vec<T>,

    /// The weights of the network.
    /// 
    /// The weights are stored a Vec of DMatrix, where each DMatrix is
    /// is a full connection from the previous layer to the next layer.
    /// 
    /// The first DMatrix is the weights from the input layer to the first hidden layer.
    /// The last DMatrix is the weights from the last hidden layer to the output layer.
    /// 
    /// The length of the Vec is the number of hidden layers + 2. (For input + output)
    pub weights: Vec<DMatrix<T>>,

}

impl ResizableNetwork<f32> {
    pub fn new(input_size: usize, hidden_size: (usize, usize), output_size: usize) -> Self {
        let input = vec![0.0; input_size];
        let activations = DMatrix::<f32>::zeros(hidden_size.0, hidden_size.1);
        let output = vec![0.0; output_size];

        let mut weights = Vec::<DMatrix<f32>>::new();
        // Create the input weights (in x h0)
        weights.push(DMatrix::<f32>::zeros(input_size, hidden_size.0));
        // Create the hidden weights (fully-connected)
        for _ in 0..hidden_size.1 { // (hn x (hn + 1)
            weights.push(DMatrix::<f32>::zeros(hidden_size.0, hidden_size.0));
        }
        // Create the output weights // hn x out
        weights.push(DMatrix::<f32>::zeros(hidden_size.1, output_size));

        ResizableNetwork {
            // input -> activations(hidden_size) -> output
            input,
            activations,
            output,
            // length = input + activations + output
            // weights[0] == input weights, weights[-1] == output, weights[1..-1] == hidden weights
            weights,
        }
    }
}

trait FeedForwardBackProp<T> {
    fn activate(&mut self);
}

impl FeedForwardBackProp<f32> for ResizableNetwork<f32> {
    fn activate(&mut self) {

    }
}


impl Default for ResizableNetwork<f32> {
    fn default() -> Self {
        Self::new(4, (2, 2), 1)
    }
}

impl fmt::Display for ResizableNetwork<f32> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "input: {:?}\toutput: {:?}\nactivations: {}\n", self.input, self.output, self.activations)?;
        
        let model_iter = self.input.iter().chain(self.activations.iter().chain(self.output.iter()));
        for (i, (m, w)) in model_iter.zip(self.weights.iter()).enumerate() {
            write!(f, "layer {}: {:?}\nweights: {}", i, m, w)?;

        }

        Ok(())
    }
}
