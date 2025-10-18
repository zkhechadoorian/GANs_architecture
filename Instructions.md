# 🧠 Generative AI: AutoEncoders & GANs — Instructions

Welcome! This guide will help you set up, run, and understand the Generative AI project focused on AutoEncoders and GANs to showcasing your work and building your online presence, **needed to get hired in the industry**.

## 🎯 Learning Objectives

This project explores two foundational generative AI approaches:

- **AutoEncoders**: For compression, denoising, and anomaly detection. AutoEncoders base concepts is used in everyday Generative AI, so it's important to understand them.
- **GANs (pix2pix)**: Used for image-to-image translation with conditional GANs. GANs were the leading approach for generating realistic images before diffusion models became popular.

## 🚀 Quick Start

Before diving into the full project, we recommend starting with our simplified Party-Time jupyter notebook in Google Colab. This **condensed version** introduces the **main concepts** and workflow without the complexity of the complete implementation.
Once you're comfortable with the fundamentals, **return here for the comprehensive walkthrough**.

**📓 [Access Party-Time Notebook](https://drive.google.com/file/d/1ATz4v7_5jR4I62GnIY258Keh9VhTI6XC/view?usp=drive_link)** - *A beginner-friendly introduction to get you started*

## 🚀 Getting Started

Follow this comprehensive step-by-step workflow to complete the project. Each step includes both execution instructions and understanding of what you're accomplishing.

### Step 1: Environment Setup

**Step 1: Fork the Repository**

1. Sign in to your GitHub account
2. Navigate to https://github.com/compu-flair/Week_23_Autoencoders_GANS
3. Click the `Fork` button in the top-right corner
4. Click `Create fork` to make a copy of the project in your GitHub account

**Step 2: Clone Your Fork to Your Local Machine**

1. On your forked repository page, click the green `Code` button
2. Select the `SSH` tab (if you see "You don't have any public SSH keys," follow the [SSH Setup Guide](./Docs/2.Add_SSH_to_GitHub.md))
3. Copy the provided SSH URL

```bash
# Clone the repository
git clone <url-to-your-forked-repo-from-steps-above>
cd Week_23_Autoencoders_GANS

# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate generative-env

# (Optional) Update the environment if you change dependencies
conda env update -f environment.yml --prune

# Add conda kernel to jupyter notebook
conda install ipykernel
python -m ipykernel install --user --name generative-env --display-name "generative-env"
```

**Additional Setup Steps:**

* **VSCode Python Interpreter Setup:**
  - **Windows/Linux:** Press `Ctrl+Shift+P`
  - **Mac:** Press `Cmd+Shift+P`

- Select "Python: Select Interpreter", then choose the "generative-env" interpreter.

* Once you open the Jupyter Notebook, it should automatically use the "generative-env" kernel. If not, please restart VSCode. And if not successful, then on the top right corner of the notebook, you can manually select the kernel by clicking on it and choosing "generative-env". You most likely will find it in the Jupyter kernel list.

🌟 **Alternative: Python Virtual Environment Setup**
Instead of using Conda, you can opt for a Python virtual environment. It's lightweight and more production-friendly, though you might encounter dependency conflicts. For detailed instructions, refer to the [📄 Setup Environment Guide](Docs/3.Setup_Environment.md).

### Step 2: Autoencoders

#### What are Autoencoders?

Autoencoders are neural networks designed to learn efficient representations of data, typically for dimensionality reduction or denoising. They consist of two main parts:

- **Encoder**: Compresses the input into a lower-dimensional latent space.
- **Decoder**: Reconstructs the original input from the compressed representation.

Autoencoders are widely used for:

- Data compression
- Noise reduction in images
- Detecting anomalies by comparing input and reconstruction

In this step, you'll explore how autoencoders work by running and modifying the provided notebook. Try changing the architecture, experimenting with different datasets, or adding noise to see how the autoencoder performs.

1. **Open the Jupyter Notebook:**
   In the left panel of the VSCode, click on the explorer tab, navigate to the notebook file `notebooks/autoencoder.ipynb`, click and open it.
2. **Go through the notebook, read, understand, and run cells one by one.**

### Step 3: GANs

#### What are GANs?

GANs (Generative Adversarial Networks) are a class of neural networks used to generate new, realistic data samples, such as images. They consist of two main components:

- **Generator**: Creates fake data samples from random noise.
- **Discriminator**: Tries to distinguish between real data and fake data produced by the generator.

GANs are widely used for:

- Generating realistic images
- Image-to-image translation (e.g., pix2pix)
- Data augmentation
- Creating art, faces, and more

In this step, you'll explore how GANs work by running and modifying the provided notebook. Try changing the architecture, experimenting with different datasets, or tuning training parameters to see how the GAN performs.

1. **Open the Jupyter Notebook:**
   In the left panel of VSCode, click on the explorer tab, navigate to the notebook file `notebooks/pix2pix_gan.ipynb`, click and open it.
2. **Go through the notebook, read, understand, and run cells one by one.**

## 🔄 Potential Changes

Consider adding these generative AI-focused extensions:

1. **Experiment with Latent Space Manipulation**: Explore how modifying the latent vectors in autoencoders or GANs affects the generated outputs. Try interpolating between samples or visualizing latent space clusters.
2. **Conditional Generation**: Implement conditional autoencoders or conditional GANs (cGANs) to generate outputs based on specific labels or attributes.
3. **Data Augmentation with Generative Models**: Use your trained autoencoder or GAN to create new synthetic data samples for training other models or improving dataset diversity.
4. **Transfer Learning**: Fine-tune pre-trained generative models on your own dataset to achieve better results or adapt to a specific domain.
5. **Model Architecture Exploration**: Experiment with different architectures (e.g., variational autoencoders, CycleGAN, StyleGAN) and compare their performance on your data.

### 🏆 Change the Project and Showcase Your Skills on GitHub

Follow these steps to make your own improvements to the project and demonstrate your learning:

1. **Create a New Branch for Your Work**

   - It's best practice to make changes on a new branch:
     ```bash
     git checkout -b my-feature-branch
     ```
2. **Make Your Changes**

   - Edit code, add features, improve documentation, or experiment with new models.
   - Commit your changes regularly:
     ```bash
     git add .
     git commit -m "Describe your change"
     ```

   Follow commit [conventions](https://www.conventionalcommits.org/en/v1.0.0/) in your change message.
3. **Push Your Changes to GitHub**

   - Push your branch to your fork:
     ```bash
     git push origin my-feature-branch
     ```
4. **Showcase Your Work**

   - **Update the [ReadMe file](README.md)** to describe the new features you added.
   - Add your deployed Streamlit app url to the ReadMe file.
   - Screen record the Streamlit app
     - walk the viewer through overall app
     - demonstrate your own updates
   - Turn the recorded video to gif image and add to your ReadMe file.
   - For a detailed guide on how to write a job-winning ReadMe, read [this file](./Docs/6.How_to_Prepare_ReadMe.md).
5. **(Optional) Create a Pull Request**

   - If you think your changes could help others, open a pull request to the original repo to contribute back. For a step-by-step guide, see [How to Make a Pull Request](./Docs/5.How_to_Make_a_Pull_Request.md).
   - **This will improve your GitHub presence, which is publicly trackable by future employers**

### Present Your Project to Potential Hiring Managers

- Make a LinkedIn post about your project, and the value you added.
- See a detailed guide on how to write a catchy LinkedIn post [here](./Docs/7.Present_project_on_LinkedIn.md).
