use neurolite::network::FatNet;






fn main() {
    let mut net = FatNet::default();

    println!("{net}");

    net.feedforward(vec![1.0, 1.0, 1.0, 1.0]);

    println!("{net}");
}
