mod data;
mod model;

use ndarray_rand::rand::seq::SliceRandom;
use ndarray::*;

fn main() {
    // Loads data
    let data = data::Data::new();
    // Displays the number index 1 in train data
    // data.display_number(&data.train_data, &data.train_labels, 100);

    // Untrained model, so random ~10% accuracy

    /*
    let model = model::Model::new();
    let cache = model.forward_propagation(&data.test_data);

    println!("First image probabilities: {:.3}", cache.a2.row(0));

    let acc = model::get_accuracy(&cache.a2, &data.test_labels);
    println!("Accuracy: {:.1}%", acc * 100.0);
    */
 
    // Training loop
    let sizes = [784, 128, 128, 10];
    let mut model = model::Model::new(sizes);

    // ===
    let y_train = data.y_label();
    let mut indices: Vec<usize> = (0..data.train_data.nrows()).collect();
    let mut learning_rate = 0.1;
    for iteration in 0..51 {

        // ==== Stochastic Gradient Decent, divide the data into chunks to speed up training ====   
        indices.shuffle(&mut ndarray_rand::rand::rng());

        for group in indices.chunks(64) {
            let x_group = data.train_data.select(Axis(0), group);
            let y_group = y_train.select(Axis(0), group);

            let cache = model.forward_propagation(&x_group);
            model.backward_propagation(&x_group, &y_group, &cache, learning_rate);
        }
        learning_rate *= 0.9;
        // Get accuracy on test data
        if iteration % 10 == 0 {
            let test_cache = model.forward_propagation(&data.test_data);
            let acc = model::get_accuracy(test_cache.nodes.last().unwrap(), &data.test_labels);

            println!("Iteration [{}]", iteration);
            println!("Accuracy: {:.1}%", acc * 100.0);
        }
    }
}
