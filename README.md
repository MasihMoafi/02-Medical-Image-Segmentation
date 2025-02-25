# Ultrasound Nerve Segmentation

This repository contains code for performing binary segmentation of nerve structures in ultrasound images using a U-Net model with a ResNet34 encoder backbone. The model leverages the [segmentation_models_pytorch](https://github.com/qubvel/segmentation_models.pytorch) library.

## Overview

The project aims to segment nerve structures from ultrasound images. The segmentation model is based on the U-Net architecture, enhanced by a ResNet34 encoder pre-trained on ImageNet. The loss function used is the Dice Loss (binary mode), which is well-suited for segmentation tasks with imbalanced classes.

## Model Details

- **Architecture:** U-Net
- **Encoder Backbone:** ResNet34 (pre-trained on ImageNet)
- **Input:** Grayscale ultrasound images (1 channel)
- **Output:** Binary segmentation mask (nerve vs. background)
- **Loss Function:** Dice Loss (binary)

## Data Preparation

- **Image Size:** 128x128 pixels
- **Data Augmentation:** 
  - Resizing
  - Horizontal and vertical flips
  - Random rotations and shifts/scales
  - Gaussian noise addition
  - Normalization
- **Dataset:** The images and corresponding masks are expected to be in the training directory with `.tif` format. Masks are normalized to [0,1] and reshaped to have a channel dimension.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- albumentations
- segmentation_models_pytorch
- numpy
- scikit-learn
- matplotlib
- Pillow

Install dependencies via pip:

```
pip install torch torchvision albumentations segmentation-models-pytorch numpy scikit-learn matplotlib Pillow
Training
The code includes a complete training loop with validation and early stopping:

Epochs: 50 (adjustable)
Batch Size: 16
Learning Rate: 1e-4 (with ReduceLROnPlateau scheduler)
Metrics: Dice coefficient and pixel accuracy are computed per epoch.
The best model (based on validation Dice) is saved as best_unet_nerve_segmentation.pth, and the final model is saved as final_unet_nerve_segmentation.pth.
```
## Running the Code
Set Up the Data:

Place your training images and masks in the designated training directory (e.g., /kaggle/input/ultrasound-nerve-segmentation/train).
Ensure that each image has a corresponding mask with _mask appended to the filename.
Execute the Script:

Run the Python script to start training:
```
python train.py
```
## Monitoring Training:

The training loop prints the loss, Dice coefficient, and pixel accuracy for both training and validation datasets.
After training, plots of loss, Dice coefficient, and accuracy over epochs will be displayed.
Visualizing Results
The script also includes functionality to visualize a few examples from the validation set. For each example, it displays:

The input image.
The ground truth mask.
The predicted mask from the model.
## Final Remarks
This project demonstrates the application of a U-Net with a ResNet34 encoder for ultrasound nerve segmentation. The combination of effective data augmentation, a robust loss function, and a pre-trained encoder allows for improved performance on binary segmentation tasks.

For further details or questions, please refer to the repository documentation or open an issue.

Happy coding!
