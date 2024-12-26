use ndarray::{Array2, Array1};
use rand::Rng;

// sigmoid activation function
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// sigmoid derivative function (for backpropogation)
fn sigmoid_derivative(x: f64) -> f64 {
    sigmoid(x) * (1.0 - sigmoid(x))
}

// neural network
struct NeuralNetwork {
    weights: Array2<f64>,  // weight matrix
    biases: Array1<f64>,   // bias vector
}

impl NeuralNetwork {

    fn new(input_size: usize, output_size: usize) -> Self {
        let mut rng = rand::thread_rng();

        let weights = Array2::from_shape_fn((input_size, output_size), |_| rng.gen_range(-1.0..1.0));
        let biases = Array1::from_shape_fn(output_size, |_| rng.gen_range(-1.0..1.0));

        NeuralNetwork { weights, biases }
    }

    fn outer_product(vec1: &Array1<f64>, vec2: &Array1<f64>) -> Array2<f64> {
        let rows = vec1.len();
        let cols = vec2.len();
        let mut result = Array2::<f64>::zeros((rows, cols));

        for i in 0..rows {
            for j in 0..cols {
                result[(i, j)] = vec1[i] * vec2[j];
            }
        }

        result
    }

    fn forward(&self, input: &Array1<f64>) -> Array1<f64> {
        let z = input.dot(&self.weights) + &self.biases;
        z.mapv(sigmoid)
    }

    fn train(&mut self, inputs: &[Array1<f64>], targets: &[Array1<f64>], epochs: usize, learning_rate: f64) {
        for _ in 0..epochs {
            for (input, target) in inputs.iter().zip(targets.iter()) {
                let output = self.forward(input);
                let error = target - &output;

                // gradient descent
                let delta = error.mapv(sigmoid_derivative);
                let outer = Self::outer_product(input, &delta);
                self.weights = &self.weights + &(outer * learning_rate);
                self.biases = &self.biases + &(delta * learning_rate);
            }
        }
    }


}

fn main() {
    // Example dataset: XOR problem
    let inputs = vec![
        Array1::from(vec![0.0, 0.0]),
        Array1::from(vec![0.0, 1.0]),
        Array1::from(vec![1.0, 0.0]),
        Array1::from(vec![1.0, 1.0]),
    ];

    let targets = vec![
        Array1::from(vec![0.0]),
        Array1::from(vec![1.0]),
        Array1::from(vec![1.0]),
        Array1::from(vec![0.0]),
    ];

    // Initialize the neural network
    let mut nn = NeuralNetwork::new(2, 1);

    // Train the network
    nn.train(&inputs, &targets, 5000, 0.1);

    // Test the network
    for input in &inputs {
        let output = nn.forward(input);
        println!("Input: {:?}, Output: {:?}", input, output);
    }
}
