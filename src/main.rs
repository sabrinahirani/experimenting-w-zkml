use ndarray::{Array1, Array2};
use rand::Rng;

use csv::ReaderBuilder;
use std::fs::File;

use rand::prelude::SliceRandom;
use std::error::Error;

#[allow(non_snake_case)]

// artificial neural network
struct ANN {
    input_dim: usize,
    hidden_dim: usize,
    output_dim: usize,
    w_0: Array2<f64>,
    w_1: Array2<f64>,
    b_0: Array1<f64>,
    b_1: Array1<f64>,
}

impl ANN {
    fn new(input_dim: usize, hidden_dim: usize, output_dim: usize) -> Self {
        let mut rng = rand::thread_rng();

        // initialize weights
        let w_0 = Array2::from_shape_fn((input_dim, hidden_dim), |_| rng.gen_range(-1.0..1.0));
        let w_1 = Array2::from_shape_fn((hidden_dim, output_dim), |_| rng.gen_range(-1.0..1.0));

        // initialize biases
        let b_0 = Array1::from_shape_fn(hidden_dim, |_| rng.gen_range(-1.0..1.0));
        let b_1 = Array1::from_shape_fn(output_dim, |_| rng.gen_range(-1.0..1.0));

        ANN { input_dim, hidden_dim, output_dim, w_0, w_1, b_0, b_1 }
    }

    // sigmoid activation function
    fn sigmoid(x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    // sigmoid derivative function (for backpropogation)
    fn sigmoid_dx(x: f64) -> f64 {
        Self::sigmoid(x) * (1.0 - Self::sigmoid(x))
    }

    fn mse_loss(y_pred: &Array1<f64>, y: &Array1<f64>) -> f64 {
        let err = y_pred - y;
        let sum_sq = err.mapv(|x| x.powi(2)).sum();
        sum_sq / (y_pred.len() as f64)
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

    fn forward(&self, x: &Array1<f64>) -> (Array1<f64>, Array1<f64>) {

        // hidden
        let x_in_0 = x.dot(&self.w_0) + &self.b_0;
        let x_out_0 = x_in_0.mapv(Self::sigmoid);

        // output
        let x_in_1 = x_out_0.dot(&self.w_1) + &self.b_1;
        let x_out_1 = x_in_1.mapv(Self::sigmoid);

        (x_out_0, x_out_1)
    }

    fn train(&mut self, X: &[Array1<f64>], Y: &[Array1<f64>], epochs: usize, learning_rate: f64) {
        for epoch in 0..epochs {

            let mut total_loss = 0.0;
            for (x, y) in X.iter().zip(Y.iter()) {

                // forward pass
                let (x_out_0, x_out_1) = self.forward(x);

                // backpropagation

                let err_out_1 = y - &x_out_1;
                let delta_out_1 = err_out_1 * x_out_1.mapv(Self::sigmoid_dx);

                let err_out_0 = delta_out_1.dot(&self.w_1.t());
                let delta_out_0 = err_out_0 * x_out_0.mapv(Self::sigmoid_dx);

                // update weights and biases using outer_product
                self.w_1 = &self.w_1 + &(Self::outer_product(&x_out_0, &delta_out_1) * learning_rate);
                self.b_1 = &self.b_1 + &(delta_out_1 * learning_rate);

                self.w_0 = &self.w_0 + &(Self::outer_product(&x, &delta_out_0) * learning_rate);
                self.b_0 = &self.b_0 + &(delta_out_0 * learning_rate);

                // compute loss
                total_loss += ANN::mse_loss(&x_out_1, y);

            }

            if epoch % 100 == 0 {
                println!("Epoch {}: Loss = {}", epoch, total_loss / X.len() as f64);
            }

        }
    }

    fn test(&self, X_test: &[Array1<f64>], Y_test: &[Array1<f64>]) -> f64 {
        let mut total_loss = 0.0;

        for (x, y) in X_test.iter().zip(Y_test.iter()) {
            let (_, x_out_1) = self.forward(x);
            total_loss += ANN::mse_loss(&x_out_1, y);
        }

        total_loss / X_test.len() as f64
    }
}

fn load_data_from_csv(file_path: &str) -> Result<(Vec<Array1<f64>>, Vec<f64>), Box<dyn Error>> {

    let file = File::open(file_path)?;
    let mut rdr = ReaderBuilder::new().has_headers(true).from_reader(file);

    let mut inputs = Vec::new();
    let mut targets = Vec::new();

    for result in rdr.records() {
        let record = result?;

        let features: Vec<f64> = record.iter().take(3).map(|x| x.parse().unwrap()).collect();
        let target: f64 = record[3].parse().unwrap();

        inputs.push(Array1::from(features));
        targets.push(target);
    }

    Ok((inputs, targets))
}

fn main() -> Result<(), Box<dyn Error>> {
    let data_file_path = "./housing.csv";
    let (mut X, mut Y) = load_data_from_csv(data_file_path)?;

    // Convert Y (f64) into Array1<f64> for compatibility with the train and test methods
    let Y: Vec<Array1<f64>> = Y.into_iter().map(|y| Array1::from(vec![y])).collect();

    // Shuffle the data
    let mut rng = rand::thread_rng();
    let len = X.len();
    let indices: Vec<usize> = (0..len).collect();
    let shuffled_indices = indices.choose_multiple(&mut rng, len).cloned().collect::<Vec<usize>>();

    // Shuffle the data
    X = shuffled_indices.iter().map(|&i| X[i].clone()).collect();
    let Y = shuffled_indices.iter().map(|&i| Y[i].clone()).collect::<Vec<Array1<f64>>>();

    // Split the data into training and test sets (90% train, 10% test)
    let train_size = (len as f64 * 0.9) as usize;
    let (X_train, X_test) = X.split_at(train_size);
    let (Y_train, Y_test) = Y.split_at(train_size);

    // Initialize and train the model
    let mut ann = ANN::new(3, 10, 1);
    ann.train(X_train, Y_train, 1000, 0.01);

    // Evaluate the model on the test set
    let test_loss = ann.test(X_test, Y_test);
    println!("Test MSE Loss: {}", test_loss);

    Ok(())
}

