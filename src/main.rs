use crate::network::ResizableNetwork;

mod matrix_utils;
mod network;


#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[inline]
fn d_sigmoid(x: f32) -> f32 {
    sigmoid(x) * (1.0 - sigmoid(x))
}


fn main() {
    let mut net = ResizableNetwork::default();
    
    println!("{net}");
}
