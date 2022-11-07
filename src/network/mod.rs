use std::fmt;

use self::layer::Layer;

mod node;
mod layer;

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn d_sigmoid(x: f32) -> f32 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

pub struct FatNet {
    input: Layer,
    hidden: Vec<Layer>,
    output: Layer,
}

impl FatNet {
    pub fn new(input_width: usize, hidden_shape: (usize, usize), output_width: usize) -> Self {
        // Create Layers.
        let mut input = Layer::new(input_width);
        let mut hidden = vec![Layer::new(hidden_shape.1); hidden_shape.0];
        let mut output = Layer::new(output_width);

        // Generate weights between layers
        // input -> hidden
        for node in &mut input.nodes {
            // node.weights = vec![0.0; hidden_shape.1];
            node.weights = vec![1.0; hidden_shape.1];
        }

        // hidden -> hidden, except the last layer
        let last_idx = hidden.len() - 1;
        for layer in &mut hidden[..last_idx] {
            for node in &mut layer.nodes {
                // node.weights = vec![0.0; hidden_shape.1];
                // lets use random weights for now
                // node.weights = vec![rand::random(); hidden_shape.1];
                node.weights = vec![1.0; hidden_shape.1];
            }
        }

        // last hidden -> output
        for node in &mut hidden[last_idx].nodes {
            // node.weights = vec![0.0; output_width];
            // lets use random weights for now
            node.weights = vec![1.0; output_width];
        }

        Self {
            input,
            hidden,
            output,
        }
    }
    
    // Take the input and feed it forward through the network.
    pub fn feedforward(&mut self, input: Vec<f32>) {
        // Set the input layer activations.
        for (node, &input) in self.input.nodes.iter_mut().zip(input.iter()) {
            node.activation = input;
        }

        // Feed the input to the first hidden layer.
        self.input.feedforward(&mut self.hidden[0]);

        // Feed the hidden layers to each other.
        for (layer, next) in self.hidden.clone().iter().zip(self.hidden.iter_mut().skip(1)) {
            layer.feedforward(next);
        }

        // Feed the last hidden layer to the output layer.
        self.hidden.last().unwrap().feedforward(&mut self.output);
    }

    pub fn backprop(&mut self, target: Vec<f32>) {
        self.update_output_error(target);

        // get the error term for each node in the hidden layer
        for (layer, next) in self.hidden.clone().iter_mut().zip(self.hidden.iter().skip(1)) {
            for (node, next) in layer.nodes.iter_mut().zip(next.nodes.iter()) {
                node.error = next.error * node.activation * (1.0 - node.activation);
            }
        }
    }

    pub fn get_output(&self) -> Vec<f32> {
        self.output.nodes.iter().map(|node| node.activation).collect()
    }

    pub fn update_output_error(&mut self, target: Vec<f32>)  {
        // https://inst.eecs.berkeley.edu/~cs182/sp06/notes/backprop.pdf
        // 6 - Derivation (output only error)
        for (node, &target) in self.output.nodes.iter_mut().zip(target.iter()) {
            node.error = (target - node.activation) * node.activation * (1.0 - node.activation);
        }
    }

    pub fn update_weights(&mut self, alpha: f32) {
        //  update the weights for each node in the hidden layer using the error term 
        // from the previous layer (next for backprop)
        let binding = self.hidden.clone();
        let next_layers = binding.iter().skip(1);
        for (layer, next) in self.hidden.iter_mut().zip(next_layers) {
            // update the weights of each layer node using
            // alpha * prev_error * this_activation
            let binding = layer.nodes.clone();
            for (node, next) in layer.nodes.iter_mut().zip(next.nodes.iter()) {
                for (weight, prev_activation) in node.weights.iter_mut().zip(binding.iter().map(|node| node.activation)) {
                    println!("weight: {}, prev_activation: {}, next.error: {}", weight, prev_activation, next.error);
                    *weight += alpha * next.error * prev_activation;
                    println!("weight: {}", weight);

                }
            }
        }
        
    }
}

fn update_weight(a: &mut Layer, b: Layer) {
    let alpha = 0.001;
    let eta = 1.0;
    // update the weights in layer a from layer b
    // using (alpha * error_term * a.activation) + 0.0 * a.delta
    for (node_a, node_b) in a.nodes.iter_mut().zip(b.nodes.iter()) {
        for (weight_a, weight_b) in node_a.weights.iter_mut().zip(node_b.weights.iter()) {
            *weight_a += alpha * node_b.error * node_a.activation;
        }
    }
    
}

impl Default for FatNet {
    fn default() -> Self {
        Self::new(4, (2, 2), 1)
    }
}

impl fmt::Display for FatNet {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "FatNet(\n")?;
        write!(f, "   Input:\n\t{:?}\n", self.input)?;
        write!(f, "   Hidden:\n")?;
        for layer in &self.hidden {
            write!(f, "\t{:?}\n", layer)?;
        }
        write!(f, "   Output:\n\t{:?}\n", self.output)?;
        write!(f, ")")
    }
}