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

## 📊 Model Performance Results

### Autoencoders

For autoencoders, model performance is primarily evaluated using:

- **Reconstruction Loss**: Measures how well the output matches the input. Common metrics include:
	- Mean Squared Error (MSE): Used for image reconstruction and denoising tasks.
	- Mean Absolute Error (MAE): Sometimes used for anomaly detection.
- **Anomaly Detection**: In the ECG example, reconstruction error (MSE) is used to flag abnormal signals.

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
