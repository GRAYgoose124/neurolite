use std::{fmt, ops::{Index, IndexMut}};

#[derive(Clone)]
struct Node {
    activation: f32,
    bias: f32,

    weights: Vec<f32>,
}

impl Node {
    fn new() -> Self {
        Self {
            activation: 0.0,
            bias: 0.0,
            weights: Vec::new(),
        }
    }
}

impl Default for Node {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for Node {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.activation)
    }
} 

#[derive(Clone)]
struct Layer {
    nodes: Vec<Node>,
}

impl Layer {
    fn new (width: usize) -> Self {
        Self {
            nodes: vec![Node::default(); width],
        }
    }

    fn feedforward(&self, next: &mut Layer) {
        for i in 0..next.len() {
            let mut sum = 0.0;
            // Multiply each node of this layer by the corresponding weight
            // linking it to the current node of the next layer, and sum the results.
            for node in self.nodes.iter() {
                sum += node.activation * node.weights[i];
            }
            // Add the node bias to the sum.
            sum += next[i].bias;
            
            // Set the node activation to the normalized sum.
            next[i].activation = sigmoid(sum);
        }
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }
}

impl Index<usize> for Layer {
    type Output = Node;

    fn index(&self, index: usize) -> &Self::Output {
        &self.nodes[index]
    }
}

impl IndexMut<usize> for Layer {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.nodes[index]
    }
}

impl fmt::Debug for Layer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[ ")?;
        for node in &self.nodes {
            write!(f, "{:?} ", node)?;
        }
        write!(f, "]")
    }
}

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
    fn new(input_width: usize, hidden_shape: (usize, usize), output_width: usize) -> Self {
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
