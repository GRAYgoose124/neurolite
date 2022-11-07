use neurolite::network::FatNet;


const EPOCHS: usize = 1000;
const LEARNING_RATE: f32 = 0.01;




fn main() {
    let xor_table = [
    (vec![0.0, 0.0], vec![0.0]),
    (vec![0.0, 1.0], vec![1.0]),
    (vec![1.0, 0.0], vec![1.0]),
    (vec![1.0, 1.0], vec![0.0])];


    let mut net = FatNet::new(2, (1, 2), 1);
    
    println!("Before training: ");
    for (input, target) in &xor_table {
        net.feedforward(input.to_vec());
        println!(" {target:?} <- {:?}", net.get_output());
    }

    for _ in 0..EPOCHS {
        for (input, target) in &xor_table {
            net.feedforward(input.to_vec());
            net.backprop(target.to_vec());
            net.update_weights(LEARNING_RATE);
        }
    }

    println!("After training:");
    for (input, target) in &xor_table {
        net.feedforward(input.to_vec());
        println!(" {target:?} <- {:?}", net.get_output());
    }
}
