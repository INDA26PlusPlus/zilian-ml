use ndarray::prelude::*;
use ndarray_rand::{RandomExt, rand_distr::Uniform};

pub struct Model {
    // List of weights and biases
    weights: Vec<Array2<f32>>,
    biases:  Vec<Array1<f32>>,
    layers: usize,
}

// Forward values
pub struct Cache {
    // List of nodes saved for backward_propagation
    // Includes x (input), a1, a2, ..., output
    pub nodes: Vec<Array2<f32>>
}

impl Model {
    pub fn new(sizes: [usize; 4]) -> Self {

        // Set random weight/biases
        let dist = Uniform::new(-0.5, 0.5).unwrap();
        let mut weights = Vec::new();
        let mut biases = Vec::new();
        let layers = sizes.len() - 1;

        for l in 0..layers {
            weights.push(Array2::random((sizes[l], sizes[l + 1]), dist));
            biases.push(Array1::random(sizes[l + 1], dist));
        }

        return Self { weights, biases, layers};
    }

    // Forward pass:
    // Input to prediction
    pub fn forward_propagation(&self, x: &Array2<f32>) -> Cache {

        // Current images raw data
        let mut a = x.clone();
        let mut nodes = Vec::new();
        nodes.push(x.clone()); // Raw data as first layer to cache


        for layer in 0..self.layers {

            // z = a * w + b
            let mut z = a.dot(&self.weights[layer]) + &self.biases[layer];
            if layer == self.layers-1 {
                softmax(&mut z); // last layer
            } 
            else {
                sigmoid(&mut z); // activation
            }
            a = z.clone(); // new input nodes

            nodes.push(z);
        }

        return Cache { nodes };
    }

    // Backward pass
    pub fn backward_propagation(&mut self, x: &Array2<f32>, y_label: &Array2<f32>, cache: &Cache, learning_rate: f32) {

        // Number of images
        let m = x.nrows() as f32;

        // Output error
        let mut dz = cache.nodes.last().unwrap() - y_label;

        for layer in (0..self.layers).rev() {

            // Activation going into current layer so:
            // output <- layer-1 <- ... <- 0 <- x
            let a_prev = &cache.nodes[layer];
            // how error changes depending on weight
            let dw = a_prev.t().dot(&dz) / m;
            // how error changes depending on bias
            let db = dz.sum_axis(Axis(0)) / m;

            // Propagate back (cant if 0)
            if layer > 0 {
                let mut a_deriv = cache.nodes[layer].clone();
                deriv_sigmoid(&mut a_deriv);

                // Each nodes previous error (so we know the error going back and how much it matters)
                dz = dz.dot(&self.weights[layer].t()) * a_deriv;
            }

            self.weights[layer] -= &(dw * learning_rate);
            self.biases[layer] -= &(db * learning_rate);
        }
    }
}

// Activation function
pub fn sigmoid(z: &mut Array2<f32>) {
    z.mapv_inplace(|x| 1.0 / (1.0 + (-x).exp()));
}

pub fn deriv_sigmoid(z:&mut Array2<f32>){
    z.mapv_inplace(|x| x * (1.0 - x));
}

// Each number (row) into probabilities
pub fn softmax(z: &mut Array2<f32>) {
    for mut row in z.axis_iter_mut(Axis(0)) {
        // Subtract max for numerical stability
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        row.mapv_inplace(|x| (x - max).exp());
        let sum = row.sum();
        row.mapv_inplace(|x| x / sum);
    }
}

// Gets accuracy across 
pub fn get_accuracy(predictions: &Array2<f32>, labels: &Array2<f32>) -> f32 {

    let mut correct_guesses = 0;

    // Go through each row
    for i in 0..predictions.nrows() {
        // Find biggest value in array
        let mut best = 0;
        for j in 0..10 {
            if predictions[[i, j]] > predictions[[i, best]] {
                best = j;
            }
        }
        if best == labels[[i, 0]] as usize {
            correct_guesses += 1;
        }
    }
    return correct_guesses as f32 / predictions.nrows() as f32;
}
