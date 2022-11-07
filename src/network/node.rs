use std::fmt;

#[derive(Clone)]
pub struct Node {
    pub activation: f32,
    pub bias: f32,
    pub error: f32,

    pub weights: Vec<f32>,
}

impl Node {
    fn new() -> Self {
        Self {
            activation: 0.0,
            bias: 0.0,
            error: 0.0,
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