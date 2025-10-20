# 🧠 Generative AI: AutoEncoders & GANs

Generative AI enables machines to create new data resembling existing samples. we will explore two foundational approaches are AutoEncoders and Generative Adversarial Networks (GANs).

## 🚀 Features

### AutoEncoders

This project explores autoencoders through three practical examples:
- **Basic Autoencoder**: Compresses and reconstructs Fashion MNIST images using a dense neural network.
- **Image Denoising Autoencoder**: Removes random noise from images with a convolutional autoencoder.
- **Anomaly Detection Autoencoder**: Detects abnormal ECG signals by training on normal data and flagging high reconstruction errors as anomalies.

### GANs (Generative Adversarial Networks)

The GAN section focuses on the pix2pix conditional GAN for image-to-image translation:
- **pix2pix (cGAN)**: Learns to map input images (e.g., architectural labels) to output images (e.g., building facades) using paired datasets.
- **U-Net Generator**: Employs skip connections for detailed image synthesis.
- **PatchGAN Discriminator**: Classifies image patches for realistic outputs.
- **Applications**: Label-to-photo translation, colorization, style transfer, and more.

## 💡 Business Value of Autoencoders & GANs

Autoencoders are powerful for efficient data compression, image quality enhancement, and anomaly detection. The autoencoder in this project learns how to compress and reconstruct images from the MNIST dataset, which could be scaled up to reduce storage costs for large-scale image datasets. The denoising autoencoder can be applied to cleaning up noisy input datasets in a variety of applications. The anomaly autoencoder can be used for real time monitoring in clinical settings to support early diagnosis and risk management for patients.

## 📝 Project Overview

This repository demonstrates the power of generative models through hands-on experiments with **Autoencoders** and **Generative Adversarial Networks (GANs)**. The project covers a range of applications, from image compression and denoising to anomaly detection and image-to-image translation.

### Autoencoders

Autoencoders are neural networks that learn to compress data into a lower-dimensional latent space and then reconstruct the original input. This project explores three main autoencoder applications:

#### 1. MNIST Encode/Decode
- **Description:** Compress and reconstruct handwritten digit images from the MNIST dataset.
- **Visualization:**
    ![MNIST Encode/Decode 1](assets/mnist_diff_heatmaps.png)
    ![MNIST Encode/Decode 2](assets/mnist_latent_analysis.png)

#### 2. Flower Image Denoising
- **Description:** Remove noise from flower images using a convolutional autoencoder.
- **Visualization:**
  - *Noisy vs. Denoised Images*
    ![Flower Denoising 1](assets/flowers_results.png)

#### 3. ECG Anomaly Detection
- **Description:** Detect abnormal heartbeats by training an autoencoder on normal ECG signals and flagging high reconstruction errors.
- **Visualization:**
  - *Normal vs. Anomalous Reconstructions*
    ![ECG Anomaly 1](assets/ecg_anomaly_1.png)
    ![ECG Anomaly 2](assets/ecg_anomaly_2.png)

### Generative Adversarial Networks (GANs)

GANs consist of a generator and a discriminator competing in a zero-sum game. This project implements the **pix2pix** conditional GAN for image-to-image translation.

#### GANs Architecture & Results
- **Description:** Translate label images to realistic building facades using paired datasets.
- **Visualization:**
  - *GAN Architecture Diagram*
    ![GAN Architecture 1](assets/gans_architecture_1.png)
    ![GAN Architecture 2](assets/gans_architecture_2.png)

---

*Replace the image placeholders above with your own

## 📂 Folder Structure

```
Week_23_Autoencoders_GANS
├── Docs/                   # Additional documentation
│
├── notebooks/              # Jupyter notebooks for experiments
│   ├── autoencoder.ipynb   # Autoencoder examples (basic, denoising, anomaly detection)
│   └── pix2pix.ipynb       # pix2pix GAN for image-to-image translation
├── .gitignore              # File to mark ignored files for git
├── environment.yml         # Conda environment for generative AI
├── Instructions.md         # Student instructions and guide
├── README.md               # Project overview and documentation
└── requirements.txt        # Python dependencies
```

## 🚀 Quick Start

For setting up the environment please refer to the [instructions file.](Instructions.md)

### Step [1]
Check autoencoder notebook at [notebooks/autoencoder.ipynb](./notebooks/autoencoder.ipynb)
Follow each cell carefully and understand it.

### Step [2]
Check the GANs notebook at [notebooks/pix2pix.ipynb](./notebooks/pix2pix.ipynb)
Follow each cell carefully to understand the GANs archeticture.


### Step By Step Instructions

To practically implement and understand AutoEncoders and GANs, follow the instructions in the [Step By Step](./step_by_step.md) file.

### 🧬 Autoencoder Notebook 

This notebook demonstrates a basic autoencoder architecture applied to the MNIST dataset. The autoencoder consists of an **encoder** that compresses input images into a low-dimensional latent space, and a **decoder** that reconstructs the images from these compressed representations.
For autoencoders, model performance is primarily evaluated using:
- **Reconstruction Loss**: Measures how well the output matches the input. Common metrics include:
	- Mean Squared Error (MSE): Used for image reconstruction and denoising tasks.
	- Mean Absolute Error (MAE): Sometimes used for anomaly detection.
- **Anomaly Detection**: In the ECG example, reconstruction error (MSE) is used to flag abnormal signals.

### Key Steps

1. **Load and Preprocess MNIST Data**
   - Visualize sample images from the dataset.
   ![Alt text](assets/mnist_sample_images.png)
	*Sample MNIST images.*
2. **Build the Autoencoder Model**
   - Define the encoder and decoder using dense layers.
   - Compile and train the model on MNIST images.

3. **Latent Space Analysis**
   - Project images into the latent space.
   - Visualize the distribution of latent vectors.
   ![Alt text](assets/mnist_latent_correlation.png)
    *Correlation matrix in latent space between digit classes.*
4. **Reconstruction and Difference Analysis**
   - Compare original and reconstructed images.
   - Visualize the difference between input and output.
   ![Alt text](assets/mnist_reconstruction_differences.png)
   * Original images (row 1) followed by their reconstructions, difference heatmaps (row 2), and difference histograms (row 3). Bright spots on heatmaps indicate regions with the lowest reconstruction quality. Difference histograms show distribution of pixel-wise differences between original and reconstructed images.
5. **Evaluate Performance**
   - Plot reconstruction loss curves.
   - Discuss results and limitations.
   ![Alt text](assets/mnist_training_curves.png)

---

*Replace the placeholders above with your own images and analysis


### GANs (pix2pix)

For GANs, especially pix2pix, the following metrics are used:

- **Generator Loss**: Combination of:
	- Adversarial Loss (Binary Crossentropy): Measures how well the generator fools the discriminator.
	- L1 Loss (Mean Absolute Error): Encourages the generated image to be similar to the target image.
- **Discriminator Loss**: Binary Crossentropy between real and generated images.
- **Visual Inspection**: Generated images are visually compared to ground truth for qualitative assessment.
- **TensorBoard Logs**: Training losses (gen_total_loss, gen_gan_loss, gen_l1_loss, disc_loss) are tracked and visualized.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request


## 📚 Documentation

For detailed setup and learning instructions, see:
- [Student Instructions](Instructions.md) - Complete learning guide
- [Environment Setup](Docs/3.Setup_Environment.md) - Development environment
- [GitHub Setup](Docs/1.Setup_Github.md) - Version control setup
