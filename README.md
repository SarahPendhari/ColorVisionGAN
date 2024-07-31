# ColorVisionGAN: Colorization of Black and White Images using GANs

This repository contains a project focused on using Generative Adversarial Networks (GANs) to simulate and restore colors in black and white images. The project leverages the COCO dataset to train a GAN model for effective colorization.

## Table of Contents
- [Project Overview](#project-overview)
- [Getting Started](#getting-started)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Visualizations](#visualizations)
- [Metrics](#metrics)
- [Future Work](#future-work)
- [Acknowledgements](#acknowledgements)
- [License](#license)

## Project Overview

This project uses a basic GAN architecture to colorize black and white images. The goal is to generate color images from grayscale inputs, which can be useful in various applications such as historical image restoration and improving accessibility for color vision deficient individuals.

## Getting Started

To get started with this project, follow these instructions:

### Prerequisites

Ensure you have the following libraries installed:
- TensorFlow 2.x
- NumPy
- Matplotlib
- scikit-image
- Keras

You can install the necessary packages using pip:

```bash
pip install tensorflow numpy matplotlib scikit-image keras
```
Dataset
The project uses the COCO dataset for training. You need to download and preprocess the dataset before training the model.

Running the Code
Clone the repository:

```bash
Copy code
git clone https://github.com/SarahPendhari/ColorVisionGAN
cd colorization-gan
```
Preprocess the dataset:

Use the provided script to preprocess images from the COCO dataset and save them in the desired format.

Train the GAN:
```bash
Run the training script:
```

Copy code
```bash
python train.py
```
Generate Colorized Images:

Use the trained model to generate colorized images from grayscale inputs:

```bash
Copy code
python generate.py
```
## Model Architecture
The project utilizes a basic GAN structure comprising:

Generator: Generates color images from grayscale inputs.
Discriminator: Distinguishes between real color images and generated color images.
Training
The model is trained using the COCO dataset with the following hyperparameters:

Epochs: 100
Batch Size: 1
Learning Rate: 0.0002
Loss Functions
Generator Loss: Combination of Binary Cross-Entropy and L1 Loss
Discriminator Loss: Binary Cross-Entropy
Evaluation
Model performance is evaluated using the following metrics:

FID Score (Fréchet Inception Distance): Measures the similarity between real and generated image distributions.
Inception Score: Evaluates the quality and diversity of generated images.
Visualizations
During training, the following visualizations are generated:

Sample Outputs: Visual comparison of generated colorized images versus grayscale inputs.
Loss Curves: Plots showing the evolution of generator and discriminator losses over epochs.
Example Output

## Metrics
The model's performance is assessed using various metrics:

FID Score: Measures the distance between the distributions of real and generated images.
Inception Score: Assesses the quality and variety of generated images.
Future Work
Improve Model Architecture: Explore more advanced GAN architectures such as CycleGAN or Pix2Pix for better colorization results.
Enhanced Evaluation Metrics: Implement additional metrics to evaluate colorization quality.
Expand Dataset: Use more diverse datasets for training to enhance model robustness.
Acknowledgements
COCO Dataset for providing the dataset.
TensorFlow and Keras for the deep learning frameworks used.
## License
This project is licensed under the MIT License - see the LICENSE file for details.