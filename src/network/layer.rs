use std::{fmt, ops::{Index, IndexMut}};

use super::{node::Node, sigmoid};


#[derive(Clone)]
pub struct Layer {
    pub nodes: Vec<Node>,
}

impl Layer {
    pub fn new (width: usize) -> Self {
        Self {
            nodes: vec![Node::default(); width],
        }
    }

    pub fn feedforward(&self, next: &mut Layer) {
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

    pub fn len(&self) -> usize {
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