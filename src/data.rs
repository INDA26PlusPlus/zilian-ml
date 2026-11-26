// Largely just taken from example from mnist documentation
use mnist::*;
use ndarray::prelude::*;

pub struct Data {
    pub train_data: Array2<f32>,
    pub train_labels: Array2<f32>,

    pub test_data: Array2<f32>,
    pub test_labels: Array2<f32>,
}

impl Data {

    pub fn new() -> Self {

        // Deconstruct the returned Mnist struct.
        let Mnist {
            trn_img,
            trn_lbl,
            tst_img,
            tst_lbl,
            ..
        } = MnistBuilder::new()
            .label_format_digit()
            .training_set_length(50_000)
            .validation_set_length(10_000)
            .test_set_length(10_000)
            .finalize();

        // Convert data to Array2 and labels
        let train_data = Array2::from_shape_vec((50_000, 784), trn_img)
            .expect("Error converting images to Array3 struct")
            .map(|x| *x as f32 / 255.0);

        let train_labels: Array2<f32> = Array2::from_shape_vec((50_000, 1), trn_lbl)
            .expect("Error converting training labels to Array2 struct")
            .map(|x| *x as f32);

        let test_data = Array2::from_shape_vec((10_000, 784), tst_img)
            .expect("Error converting images to Array3 struct")
            .map(|x| *x as f32 / 255.0);

        let test_labels: Array2<f32> = Array2::from_shape_vec((10_000, 1), tst_lbl)
            .expect("Error converting testing labels to Array2 struct")
            .map(|x| *x as f32);
        
        Self {
            train_data,
            train_labels,

            test_data,
            test_labels,
        }
        

    }


    pub fn _display_number(&self, num_data: &Array2<f32>, num_label: &Array2<f32>, image_num: usize) {
        let luminance:[char; 11] = [' ', '.','-',':',';','=','!','*','#','$', '@' ];

        // Get row of numbers per image
        for row in num_data.slice(s![image_num, ..]).rows() {

            let mut col_index = 0;
            for &value in row {

                // Get luminance for each number
                let luminance_index = (value * 10.0) as usize;
                print!("{}", luminance[luminance_index]);

                // Increment the column counter and start new row if needed
                col_index += 1;

                if (col_index % 28) == 0 {
                    println!()
                }
            }
        }
        println!("The digit is a {:?}",num_label.slice(s![image_num, ..]) );
    }

    // Set 1 in each row that the columns number is supposed to be
    pub fn y_label(&self) -> Array2<f32> {
        let mut y = Array2::zeros((50_000, 10));
        for (i, &digit) in self.train_labels.iter().enumerate() {
            y[[i, digit as usize]] = 1.0;
        }
        return y;
    }
}




