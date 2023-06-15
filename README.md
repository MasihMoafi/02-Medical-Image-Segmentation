## Medical Image Segmentation with U-Net Architecture

This repository contains code for performing medical image segmentation using the U-Net architecture. The aim is to rectify the challenge of having a small dataset by utilizing the power of U-Net for segmentation tasks.

# Dataset

The dataset used for this project can be found at [Kaggle: USSIM and SEGM](https://www.kaggle.com/datasets/ignaciorlando/ussimandsegm). Please download the dataset and adjust accordingly.

# Code

The code consists of two main parts:

    Data Preparation: The code loads and preprocesses the training images and masks. It performs necessary resizing, normalization, and encoding of the masks. The data is split into a training set and a validation set.

    U-Net Model: The code defines the U-Net model architecture using TensorFlow and Keras. It includes the encoding and decoding stages with convolutional and upsampling blocks. The model is compiled and trained on the prepared dataset.

Due to the high computational requirements, the results are not printed in the notebook. However, the code is reliable and can be executed on a suitable computing environment.

# Usage

To use this code, follow these steps:

    Download and organize the dataset as mentioned above.
    Install the required dependencies, including TensorFlow, Keras, PIL, NumPy, and matplotlib.
    Open the Jupyter Notebook or Python script and execute the code sequentially.
    Monitor the training progress and evaluate the model performance.

Note: Adjust the file paths in the code to match the location of the dataset on your system.

# Results

The results of the image segmentation will be obtained after training the U-Net model. You can analyze the results by visualizing the segmented images and evaluating the model's performance metrics.
Acknowledgments

The implementation of the U-Net architecture is inspired by the original paper "U-Net: Convolutional Networks for Biomedical Image Segmentation" by Olaf Ronneberger, Philipp Fischer, and Thomas Brox.

License

This is a personal project for portfolio purposes and does not come with a specific license. You are free to modify and use the code in this repository for personal or educational purposes. If you find it useful, kindly give credit by referencing this repository.
