use ndarray::prelude::*;
use ndarray_rand::{RandomExt, rand_distr::Uniform};

pub struct Model {
    w1: Array2<f32>,
    b1: Array1<f32>,
    w2: Array2<f32>,
    b2: Array1<f32>, 
}

// Forward values
pub struct Cache {
    pub _z1: Array2<f32>,
    pub a1: Array2<f32>,
    pub _z2: Array2<f32>,
    pub a2: Array2<f32>, // output probabilities
}

impl Model {
    pub fn new() -> Self {

        // Set random weight/biases
        let dist = Uniform::new(-0.5, 0.5).unwrap();

        // Hidden layer 1
        let w1 = Array2::random((784, 10), dist);
        let b1 = Array1::random(10, dist);

        // Output layer
        let w2 = Array2::random((10, 10), dist);
        let b2 = Array1::random(10, dist);

        return Self { w1, b1, w2, b2 };
    }

    pub fn forward_propagation(&self, x: &Array2<f32>) -> Cache {
        // Hidden layer matrix multiplication with weight + bias
        let _z1 = x.dot(&self.w1) + &self.b1;
        let mut a1 = _z1.clone();
        sigmoid(&mut a1);

        // Output matrix multiplication with weight + bias
        let _z2 = &a1.dot(&self.w2) + &self.b2;
        let mut a2 = _z2.clone();
        softmax(&mut a2);

        return Cache { _z1, a1, _z2, a2 };
    }

    pub fn backward_propagation(&mut self, x: &Array2<f32>, y_label: &Array2<f32>, cache: &Cache, learning_rate: f32) {

        // Each row is a new sample
        let m = x.nrows() as f32;

        // Output layer
        let dz2 = &cache.a2 - y_label;          // Output error
        let dw2 = cache.a1.t().dot(&dz2) / m;   // Average weight error
        let db2 = dz2.sum_axis(Axis(0)) / m;    // Average bias error

        // Hidden layer
        let mut a1_deriv = cache.a1.clone();
        deriv_sigmoid(&mut a1_deriv);   // 
        let dz1 = dz2.dot(&self.w2.t()) * &a1_deriv;    // Node error
        let dw1 = x.t().dot(&dz1) / m;                  // Average weight error
        let db1 = dz1.sum_axis(Axis(0)) / m;            // Average basis error

        // Step in the direction that decreases error most
        self.w1 -= &(dw1 * learning_rate);
        self.b1 -= &(db1 * learning_rate);
        self.w2 -= &(dw2 * learning_rate);
        self.b2 -= &(db2 * learning_rate);
    }
}

// Activation
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
