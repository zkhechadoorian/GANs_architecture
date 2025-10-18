# Practice Creating Project Step-by-Step Instructions

This document provides step-by-step instructions for creating a project. Follow each step carefully to ensure successful completion.

You're required to follow and think the steps below to create the project. Please try to implement each step on your own before looking at the provided code snippets. 

Click reveal to see the code snippets for each step.
<details>
  <summary>Reveal</summary>
  Correct, you shall press on this button when you are stuck or want to see the solution.
</details>

## Step 1: AutoEncoders

We will start by exploring AutoEncoders.
In `notebooks` folder, create a new notebook named `autoencoder.ipynb`.

### Basic AutoEncoder


1. We will be using Tensorflow, dataframes, matplotlib, numpy, metrics from scikit-learn libraries. We will also need to import keras layers, losses and Model to create our AutoEncoder.

<details>
    <summary>Reveal</summary>
    
```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
```
</details>

2. We need to load the dataset. We will be using the Fashion MNIST dataset which is available in keras datasets. Please load the dataset and unpack it into training and testing sets. Then check the shape of the data.

Note: The dataset contains grayscale images of 28x28 pixels. You will need to normalize the pixel values.

<details>
    <summary>Reveal</summary>
    
```python

(x_train, _), (x_test, _) = fashion_mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

print (x_train.shape)
print (x_test.shape)
```
</details>

3. Next, we need to build our AutoEncoder model. The model is sequantial and consists of an encoder and a decoder. The encoder compresses the input data into a lower-dimensional representation, while the decoder reconstructs the original data from this compressed representation.

Note: The input shape is (28, 28, 1) since the images are grayscale.

Note: You can add a flatten layer to convert the 2D images into 1D vectors before passing them to the dense layers.


<details>
    <summary>Reveal</summary>
    
```python
# Define an Autoencoder class inheriting from tf.keras.Model
class Autoencoder(Model):
  def __init__(self, latent_dim, shape):
    super(Autoencoder, self).__init__()  # Initialize the base class
    self.latent_dim = latent_dim         # Store the size of the latent space
    self.shape = shape                   # Store the original input shape

    # Encoder: flattens input and encodes to latent_dim
    self.encoder = tf.keras.Sequential([
      layers.Flatten(),                  # Flatten input to 1D
      layers.Dense(latent_dim, activation='relu'),  # Dense layer for encoding
    ])

    # Decoder: reconstructs original shape from latent vector
    self.decoder = tf.keras.Sequential([
      layers.Dense(tf.math.reduce_prod(shape).numpy(), activation='sigmoid'),  # Dense layer to expand back to original size
      layers.Reshape(shape)               # Reshape output to original input shape
    ])

  # Forward pass: encode then decode
  def call(self, x):
    encoded = self.encoder(x)             # Encode input
    decoded = self.decoder(encoded)       # Decode latent vector
    return decoded     
```
</details>

4. Now, we need to compile the model and train it. We will use the Adam optimizer. Guess the loss function that is suitable for this task.

<details>
    <summary>Reveal</summary>
    
```python
# Set the shape and latent dimension for the autoencoder
shape = x_test.shape[1:]                  # Get shape of input images (e.g., (28, 28))
latent_dim = 64                           # Set size of latent space

# Instantiate the Autoencoder model
autoencoder = Autoencoder(latent_dim, shape)
autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())
autoencoder.fit(x_train, x_train,
                epochs=10,
                shuffle=True,
                validation_data=(x_test, x_test))
```
</details>

5. Finally, we will visualize the results. We will take some test images, pass them through the autoencoder and display the original and reconstructed images side by side.

<details>
    <summary>Reveal</summary>
    
```python
encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()


n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
  # display original
  ax = plt.subplot(2, n, i + 1)
  plt.imshow(x_test[i])
  plt.title("original")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)

  # display reconstruction
  ax = plt.subplot(2, n, i + 1 + n)
  plt.imshow(decoded_imgs[i])
  plt.title("reconstructed")
  plt.gray()
  ax.get_xaxis().set_visible(False)
  ax.get_yaxis().set_visible(False)
plt.show()
```
</details>

### Image Denoising with AutoEncoder

We will add gaussian noise to the images and train the autoencoder to remove the noise. We will be using MNIST dataset for this task as well.

1. Load the MNIST dataset and normalize the pixel values. We also be using Conv2D layers in our model, so we need to add a channel dimension to the images.

<details>
    <summary>Reveal</summary>
    
```python
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.

# Add a channel dimension to the images for compatibility with Conv2D layers
x_train = x_train[..., tf.newaxis]
x_test = x_test[..., tf.newaxis]

print (x_train.shape)
```
</details>

2. Next, we will add gaussian noise to the train and test images. Add 20% noise to the images.

Hint: You can use the equavilant of `np.random.normal` to generate gaussian noise in Tensorflow.

<details>
    <summary>Reveal</summary>
    
```python
noise_factor = 0.2  # Set the amount of noise to add

# Add random Gaussian noise to the training images
x_train_noisy = x_train + noise_factor * tf.random.normal(shape=x_train.shape)
# Add random Gaussian noise to the test images
x_test_noisy = x_test + noise_factor * tf.random.normal(shape=x_test.shape)
```
</details>

3. Remember we already scalled our images to be between 0 and 1. After adding noise, some pixel values might be outside this range. We need to clip the values to ensure they remain between 0 and 1.

<details>
    <summary>Reveal</summary>
    
```python
# Clip the noisy training images to be between 0 and 1
x_train_noisy = tf.clip_by_value(x_train_noisy, clip_value_min=0., clip_value_max=1.)
# Clip the noisy test images to be between 0 and 1
x_test_noisy = tf.clip_by_value(x_test_noisy, clip_value_min=0., clip_value_max=1.)
```
</details>

4. Aren't you curious to see how the noisy images look like? Let's visualize some of the noisy images. Check only 10 images of the test dataset.

<details>
    <summary>Reveal</summary>
    
```python

n = 10  # Number of images to display
plt.figure(figsize=(20, 4))
for i in range(n):
    ax = plt.subplot(1, n, i + 1)  # Create a subplot for each image
    plt.imshow(x_test_noisy[i].numpy().squeeze(), cmap='gray')
    plt.title("noisy")
    plt.gray()
plt.show()
```
</details>

5. Now, we will build our denoising autoencoder model. We will use Conv2D and Conv2DTranspose layers for the encoder and decoder respectively. The encoder will consist of two Conv2D layers with ReLU activation and max pooling. The decoder will consist of two Conv2DTranspose layers with ReLU activation and a final Conv2D layer with sigmoid activation to reconstruct the image.

<details>
    <summary>Reveal</summary>
    
```python
# Define a convolutional autoencoder model for image denoising
class Denoise(Model):
  def __init__(self):
    super(Denoise, self).__init__()  # Initialize the base Model class

    # Define the encoder as a Sequential model
    self.encoder = tf.keras.Sequential([
      layers.Input(shape=(28, 28, 1)),  # Input layer for 28x28 grayscale images
      layers.Conv2D(16, (3, 3), activation='relu', padding='same', strides=2),  # Downsample with 16 filters
      layers.Conv2D(8, (3, 3), activation='relu', padding='same', strides=2)    # Further downsample with 8 filters
    ])

    # Define the decoder as a Sequential model
    self.decoder = tf.keras.Sequential([
      layers.Conv2DTranspose(8, kernel_size=3, strides=2, activation='relu', padding='same'),   # Upsample with 8 filters
      layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', padding='same'),  # Further upsample with 16 filters
      layers.Conv2D(1, kernel_size=(3, 3), activation='sigmoid', padding='same')                # Output layer to reconstruct the image
    ])

  # Define the forward pass
  def call(self, x):
    encoded = self.encoder(x)   # Pass input through encoder
    decoded = self.decoder(encoded)  # Pass encoded output through decoder
    return decoded             # Return the reconstructed image
```
</details>

6. Now, we need to compile and train the model. We will use the Adam optimizer and Mean Squared Error loss function. We will train the model for 10 epochs.

<details>
    <summary>Reveal</summary>
    
```python
autoencoder = Denoise()

autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

autoencoder.fit(x_train_noisy, x_train,
                epochs=10,
                shuffle=True,
                validation_data=(x_test_noisy, x_test))
```
</details>

7. Let's view our encoder/decoder parameters and summary.

<details>
    <summary>Reveal</summary>
    
```python

autoencoder.encoder.summary()
autoencoder.decoder.summary()
```
</details>

8. Finally, we will visualize the results. We will take some noisy test images, pass them through the autoencoder and display the noisy and denoised images side by side.

<details>
    <summary>Reveal</summary>
    
```python

encoded_imgs = autoencoder.encoder(x_test_noisy).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

n = 10
plt.figure(figsize=(20, 4))
for i in range(n):

    # display original + noise
    ax = plt.subplot(2, n, i + 1)
    plt.title("original + noise")
    plt.imshow(tf.squeeze(x_test_noisy[i]))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    bx = plt.subplot(2, n, i + n + 1)
    plt.title("reconstructed")
    plt.imshow(tf.squeeze(decoded_imgs[i]))
    plt.gray()
    bx.get_xaxis().set_visible(False)
    bx.get_yaxis().set_visible(False)
plt.show()
```
</details>

### ECG Data for Anomaly Detection with AutoEncoder

We will use the ECG5000 dataset from UCR archive for this task. The dataset contains ECG signals with normal and abnormal heartbeats. We will train an autoencoder on the normal heartbeats and use it to detect anomalies.

1. Load the ECG5000 dataset. The dataset is in a csv format and can be retrieved from thi url `http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv`. Don't forget to set header to None while loading the dataset and check the first 5 rows of the dataset.

<details>
    <summary>Reveal</summary>
    
```python
# Download the dataset
dataframe = pd.read_csv('http://storage.googleapis.com/download.tensorflow.org/data/ecg.csv', header=None)
raw_data = dataframe.values
dataframe.head()
```
</details>

2. The last column of the dataset contains the labels. The label 1 represents normal heartbeats, while labels 2, 3, 4, and 5 represent different types of anomalies. We need to seperate the features and labels. Then please test split the data into training and testing sets. 

<details>
    <summary>Reveal</summary>
    
```python
# The last element contains the labels
labels = raw_data[:, -1]

# The other data points are the electrocadriogram data
data = raw_data[:, 0:-1]

train_data, test_data, train_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=21
)
```
</details>

3. We need to normalize the data. We will use Min-Max scaling to scale the data between 0 and 1. Can you do it manually using tensorflow?

Note: We need to make sure all of our data are tensorflow float32. You can cast them to tensors using `tf.cast` function.

<details>
    <summary>Reveal</summary>
    
```python
min_val = tf.reduce_min(train_data)
max_val = tf.reduce_max(train_data)

train_data = (train_data - min_val) / (max_val - min_val)
test_data = (test_data - min_val) / (max_val - min_val)

train_data = tf.cast(train_data, tf.float32)
test_data = tf.cast(test_data, tf.float32)
```
</details>

4. We will only use the normal heartbeats (label 1) for training the autoencoder. Please filter the training data to only include samples with label 1. and keep anomalous data for testing.

<details>
    <summary>Reveal</summary>
    
```python
train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)

normal_train_data = train_data[train_labels]
normal_test_data = test_data[test_labels]

anomalous_train_data = train_data[~train_labels]
anomalous_test_data = test_data[~test_labels]
```
</details>


5. Plot the first 5 normal and anomalous ECG signals from the training set.

<details>
    <summary>Reveal</summary>
    
```python
plt.grid()
plt.plot(np.arange(140), normal_train_data[0])
plt.title("A Normal ECG")
plt.show()


plt.grid()
plt.plot(np.arange(140), anomalous_train_data[0])
plt.title("An Anomalous ECG")
plt.show()
```
</details>

6. Now, we will build our autoencoder model. The model will consist of an encoder and a decoder. both of them will have 3 dense layers.

<details>
    <summary>Reveal</summary>
    
```python
class AnomalyDetector(Model):
  def __init__(self):
    super(AnomalyDetector, self).__init__()
    self.encoder = tf.keras.Sequential([
      layers.Dense(32, activation="relu"),
      layers.Dense(16, activation="relu"),
      layers.Dense(8, activation="relu")])

    self.decoder = tf.keras.Sequential([
      layers.Dense(16, activation="relu"),
      layers.Dense(32, activation="relu"),
      layers.Dense(140, activation="sigmoid")])

  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded
```
</details>

7. Now, we need to compile and train the model. We will use the Adam optimizer and Mean Squared Error loss function. We will train the model for 20 epochs.

<details>
    <summary>Reveal</summary>
    
```python
autoencoder = AnomalyDetector()
autoencoder.compile(optimizer='adam', loss='mae')
history = autoencoder.fit(normal_train_data, normal_train_data,
          epochs=20,
          batch_size=512,
          validation_data=(test_data, test_data),
          shuffle=True)
```
</details>

8. Let's plot our training and validation loss.

<details>
    <summary>Reveal</summary>
    
```python
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
```
</details>

9. Now, we will use the trained autoencoder to reconstruct the test data. We will then calculate the reconstruction error for each sample in the test set. Plot the reconstruction error for both normal and anomalous samples, you can use fill_between to highlight the area under the curve.

<details>
    <summary>Reveal</summary>
    
```python

encoded_data = autoencoder.encoder(normal_test_data).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()

plt.plot(normal_test_data[0], 'b')
plt.plot(decoded_data[0], 'r')
plt.fill_between(np.arange(140), decoded_data[0], normal_test_data[0], color='lightcoral')
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.show()


encoded_data = autoencoder.encoder(anomalous_test_data).numpy()
decoded_data = autoencoder.decoder(encoded_data).numpy()

plt.plot(anomalous_test_data[0], 'b')
plt.plot(decoded_data[0], 'r')
plt.fill_between(np.arange(140), decoded_data[0], anomalous_test_data[0], color='lightcoral')
plt.legend(labels=["Input", "Reconstruction", "Error"])
plt.show()

```
</details>


10.  Finally, we will set a threshold for the reconstruction error to classify samples as normal or anomalous.

<details>
    <summary>Reveal</summary>
    
```python
reconstructions = autoencoder.predict(normal_train_data)
train_loss = tf.keras.losses.mae(reconstructions, normal_train_data)

threshold = np.mean(train_loss) + np.std(train_loss)
print("Threshold: ", threshold)
```
</details>

11. We need to simpilify prediction, please create a function that takes data, trained model and threashold as inputs and returns the predictions if the data is ourlier or not.

<details>
    <summary>Reveal</summary>
    
```python
def predict(model, data, threshold):
    reconstructions = model.predict(data)
    loss = tf.keras.losses.mae(reconstructions, data)
    return tf.math.less(loss, threshold)
```
</details>

12. Create a function called `print_stats` that takes true labels and predicted labels as inputs and prints accuracy, precision and recall scores.

<details>
    <summary>Reveal</summary>
    
```python
def print_stats(predictions, labels):
  # Print the accuracy of the predictions compared to the true labels
  print("Accuracy = {}".format(accuracy_score(labels, predictions)))
  # Print the precision of the predictions
  print("Precision = {}".format(precision_score(labels, predictions)))
  # Print the recall of the predictions
  print("Recall = {}".format(recall_score(labels, predictions)))
```
</details>

13. Give it a try!

<details>
    <summary>Reveal</summary>
    
```python
preds = predict(autoencoder, test_data, threshold)
print_stats(preds, test_labels)
```
</details>

## Step 2: Generative Adversarial Networks (GANs)

We will explore Generative Adversarial Networks (GANs) and apply pix2pix GAN on the facades dataset.

1. In `notebooks` folder, create a new notebook named `pix2pix_gan.ipynb`.

2. We will be using Tensorflow, dataframes, matplotlib, datetime, os, time, and Display from IPython libraries. We will also need to import keras layers, losses and Model to create our GAN.

<details>
    <summary>Reveal</summary>
    
```python
import tensorflow as tf

import os
import pathlib
import time
import datetime

from matplotlib import pyplot as plt
from IPython import display
```
</details>

3. We need to load the facades dataset. The dataset is available in tensorflow datasets. Please load the dataset and split it into training and testing sets. 
Note: dataset name is `facades`. and you can retrieve it from this link `http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/{dataset_name}.tar.gz`. You can also use other datasets if you want.


<details>
    <summary>Reveal</summary>
    
```python

dataset_name = "facades" #@param ["cityscapes", "edges2handbags", "edges2shoes", "facades", "maps", "night2day"]


_URL = f'http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/{dataset_name}.tar.gz'

path_to_zip = tf.keras.utils.get_file(
    fname=f"{dataset_name}.tar.gz",
    origin=_URL,
    extract=True)

path_to_zip  = pathlib.Path(path_to_zip)

PATH = path_to_zip/dataset_name
```
</details>


4. Let's read one sample image and check it's shape to understand how our model input and output will look like.

<details>
    <summary>Reveal</summary>
    
```python
sample_img_path = str(PATH / os.path.join('train', '1.jpg'))
print(sample_img_path)
sample_image = tf.io.read_file(sample_img_path)
sample_image = tf.io.decode_jpeg(sample_image)
print(sample_image.shape)
```
</details>

5. But how does it look like? Let's visualize it.


<details>
    <summary>Reveal</summary>
    
```python

plt.figure()
plt.imshow(sample_image)
```
</details>


6. Now we need to create a helper function to load image path, read it using tesorflow read_file, decode the image, split the images into input and target images, and return the input_image and real_image.

<details>
    <summary>Reveal</summary>
    
```python
def load(image_file):
  # Read and decode an image file to a uint8 tensor
  image = tf.io.read_file(image_file)
  image = tf.io.decode_jpeg(image)

  # Split each image tensor into two tensors:
  # - one with a real building facade image
  # - one with an architecture label image 
  w = tf.shape(image)[1]
  w = w // 2
  input_image = image[:, w:, :]
  real_image = image[:, :w, :]

  # Convert both images to float32 tensors
  input_image = tf.cast(input_image, tf.float32)
  real_image = tf.cast(real_image, tf.float32)

  return input_image, real_image
```
</details>

7. Let's test our function by passing the sample image path to it and visualizing the input and target images using matplotlib.

<details>
    <summary>Reveal</summary>
    
```python
inp, re = load(str(PATH / 'train/100.jpg'))
# Casting to int for matplotlib to display the images
plt.figure()
plt.imshow(inp / 255.0)
plt.figure()
plt.imshow(re / 255.0)
```
</details>

8. We need to preprocess the images before passing them to the model. We will create three functions for this purpose. The first function will resize the images to a given height and width. The second function will randomly jitter crop the images to our constant sizes. The third function will normalize the images to be between -1 and 1.

Note: Before that we need to define our constant image height and width to 256, batch size to 1 and buffer size to 400.

<details>
    <summary>Reveal</summary>
    
```python

# The facade training set consist of 400 images
BUFFER_SIZE = 400
# The batch size of 1 produced better results for the U-Net in the original pix2pix experiment
BATCH_SIZE = 1
# Each image is 256x256 in size
IMG_WIDTH = 256
IMG_HEIGHT = 256

def resize(input_image, real_image, height, width):
  input_image = tf.image.resize(input_image, [height, width],
                                method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  real_image = tf.image.resize(real_image, [height, width],
                               method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

  return input_image, real_image

def random_crop(input_image, real_image):
  stacked_image = tf.stack([input_image, real_image], axis=0)
  cropped_image = tf.image.random_crop(
      stacked_image, size=[2, IMG_HEIGHT, IMG_WIDTH, 3])

  return cropped_image[0], cropped_image[1]

# Normalizing the images to [-1, 1]
def normalize(input_image, real_image):
  input_image = (input_image / 127.5) - 1
  real_image = (real_image / 127.5) - 1

  return input_image, real_image
```
</details>

9. There's a concept called jittering which is used to augment the dataset. We will create a function that resizes the images to 286x286, randomly crops them to 256x256, and randomly flips them horizontally. 

Note: wapre it with `tf.function` to speed up the execution.


<details>
    <summary>Reveal</summary>
    
```python
@tf.function()
def random_jitter(input_image, real_image):
  # Resizing to 286x286
  input_image, real_image = resize(input_image, real_image, 286, 286)

  # Random cropping back to 256x256
  input_image, real_image = random_crop(input_image, real_image)

  if tf.random.uniform(()) > 0.5:
    # Random mirroring
    input_image = tf.image.flip_left_right(input_image)
    real_image = tf.image.flip_left_right(real_image)

  return input_image, real_image
```
</details>

10. Curiouse to check how the jittering looks like? Let's visualize the input and target images after applying jittering and normalization.


<details>
    <summary>Reveal</summary>


```python

plt.figure(figsize=(6, 6))
for i in range(4):
  rj_inp, rj_re = random_jitter(inp, re)
  plt.subplot(2, 2, i + 1)
  plt.imshow(rj_inp / 255.0)
  plt.axis('off')
plt.show()
```
</details>

11. We have an issue, jittering is only applied for training dataset. Therefore we need to create two functions to load and preprocess the training and testing images. The training function will apply jittering and normalization, while the testing function will only apply resizing and normalization.


<details>
    <summary>Reveal</summary>
    
```python

def load_image_train(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = random_jitter(input_image, real_image)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image

def load_image_test(image_file):
  input_image, real_image = load(image_file)
  input_image, real_image = resize(input_image, real_image,
                                   IMG_HEIGHT, IMG_WIDTH)
  input_image, real_image = normalize(input_image, real_image)

  return input_image, real_image
```
</details>

12. Now, we need to create the training and testing datasets. We will use the `tf.data` API to create the datasets. We will map the training dataset to the `load_image_train` function and the testing dataset to the `load_image_test` function. We will also shuffle and batch the training dataset.

Hint: Use Dataset.list_files to get the list of image files in the training and testing directories. Then you can use map function.

<details>
    <summary>Reveal</summary>
    
```python
train_dataset = tf.data.Dataset.list_files(str(PATH / 'train/*.jpg'))
train_dataset = train_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)

try:
  test_dataset = tf.data.Dataset.list_files(str(PATH / 'test/*.jpg'))
except tf.errors.InvalidArgumentError:
  test_dataset = tf.data.Dataset.list_files(str(PATH / 'val/*.jpg'))
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)
```

</details>

13. We are done with data preprocessing. Now, we will build the generator model. The generator is a U-Net model that consists of an encoder and a decoder. The encoder consists of downsampling layers, while the decoder consists of upsampling layers. We will also use skip connections between the encoder and decoder.

Can you create a downsample and upsample function that returns a sequential model with the layers Conv2D, batch normalization, LeakyReLU for downsample and Conv2DTranspose, batch normalization, Dropout (optional), ReLU for upsample?


<details>
    <summary>Reveal</summary>
    
```python

def downsample(filters, size, apply_batchnorm=True):
  # Initialize the weights with a normal distribution
  initializer = tf.random_normal_initializer(0., 0.02)

  # Create a Sequential model
  result = tf.keras.Sequential()
  # Add a Conv2D layer with the given number of filters and kernel size
  result.add(
      tf.keras.layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))

  # Optionally add BatchNormalization for faster and more stable training
  if apply_batchnorm:
    result.add(tf.keras.layers.BatchNormalization())

  # Add LeakyReLU activation for non-linearity
  result.add(tf.keras.layers.LeakyReLU())

  return result


def upsample(filters, size, apply_dropout=False):
  # Initialize the weights with a normal distribution
  initializer = tf.random_normal_initializer(0., 0.02)

  # Create a Sequential model for the upsampling block
  result = tf.keras.Sequential()
  # Add a Conv2DTranspose layer to upsample the input
  result.add(
    tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
                                    padding='same',
                                    kernel_initializer=initializer,
                                    use_bias=False))

  # Add BatchNormalization for faster and more stable training
  result.add(tf.keras.layers.BatchNormalization())

  # Optionally add Dropout for regularization (only in first 3 blocks of decoder)
  if apply_dropout:
      result.add(tf.keras.layers.Dropout(0.5))

  # Add ReLU activation for non-linearity
  result.add(tf.keras.layers.ReLU())

  return result
```
</details>

14. Can you test them and check their shapes?

<details>
    <summary>Reveal</summary>
    
```python
down_model = downsample(3, 4)
down_result = down_model(tf.expand_dims(inp, 0))
print (down_result.shape)

up_model = upsample(3, 4)
up_result = up_model(down_result)
print (up_result.shape)
```
</details>

15. Perfect! Now, we will build the generator model using the downsample and upsample functions. The generator will consist of 8 downsampling layers and 8 upsampling layers. We will also use skip connections between the encoder and decoder.

Note: It's tricky, you can check the original paper for the details of the architecture. [Link](https://arxiv.org/pdf/1611.07004)


<details>
    <summary>Reveal</summary>
    
```python
def Generator():
  # Define the input layer with shape (256, 256, 3)
  inputs = tf.keras.layers.Input(shape=[256, 256, 3])

  # Create the encoder (downsampling stack) using downsample blocks
  down_stack = [
    downsample(64, 4, apply_batchnorm=False),  # First block, no batchnorm
    downsample(128, 4),  # Second block
    downsample(256, 4),  # Third block
    downsample(512, 4),  # Fourth block
    downsample(512, 4),  # Fifth block
    downsample(512, 4),  # Sixth block
    downsample(512, 4),  # Seventh block
    downsample(512, 4),  # Eighth block
  ]

  # Create the decoder (upsampling stack) using upsample blocks
  up_stack = [
    upsample(512, 4, apply_dropout=True),  # First block, with dropout
    upsample(512, 4, apply_dropout=True),  # Second block, with dropout
    upsample(512, 4, apply_dropout=True),  # Third block, with dropout
    upsample(512, 4),  # Fourth block
    upsample(256, 4),  # Fifth block
    upsample(128, 4),  # Sixth block
    upsample(64, 4),   # Seventh block
  ]

  # Initialize the weights for the last layer
  initializer = tf.random_normal_initializer(0., 0.02)
  # Define the last layer to get the output image with tanh activation
  last = tf.keras.layers.Conv2DTranspose(
      OUTPUT_CHANNELS, 4,
      strides=2,
      padding='same',
      kernel_initializer=initializer,
      activation='tanh')  # Output shape: (batch_size, 256, 256, 3)

  x = inputs  # Start with the input

  # Downsampling through the encoder, saving skip connections
  skips = []
  for down in down_stack:
    x = down(x)      # Apply downsampling block
    skips.append(x)  # Save output for skip connection

  # Reverse all but the last skip for use in upsampling
  skips = reversed(skips[:-1])

  # Upsampling and adding skip connections from encoder
  for up, skip in zip(up_stack, skips):
    x = up(x)                              # Apply upsampling block
    x = tf.keras.layers.Concatenate()([x, skip])  # Add skip connection

  x = last(x)  # Apply the last layer to get the final output

  # Return the Keras Model
  return tf.keras.Model(inputs=inputs, outputs=x)
```
</details>

16. Let's create the generator model and plot it's archeticture.

<details>
    <summary>Reveal</summary>
    
```python
generator = Generator()
tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)
```
</details>

17. Can you try passing a sample input image to the generator and check plot the generated image?

<details>
    <summary>Reveal</summary>
    
```python
gen_output = generator(inp[tf.newaxis, ...], training=False)
plt.imshow(gen_output[0, ...])
```
</details>

18. We have built the generator model. But we need to build a special loss function for the generator. The generator loss consists of two parts: the adversarial loss and the L1 loss. The adversarial loss is calculated using binary crossentropy between the real and generated images. The L1 loss is calculated using mean absolute error between the real and generated images. The total generator loss is the sum of the adversarial loss and the L1 loss multiplied by a lambda factor.
Note: Set the lambda factor to 100.


<details>
    <summary>Reveal</summary>
    
```python

LAMBDA = 100
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
def generator_loss(disc_generated_output, gen_output, target):
  # GAN loss: how well the generator fools the discriminator
  gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

  # L1 loss: mean absolute error between generated image and target image
  l1_loss = tf.reduce_mean(tf.abs(target - gen_output))

  # Total generator loss: GAN loss + weighted L1 loss
  total_gen_loss = gan_loss + (LAMBDA * l1_loss)

  # Return all losses for logging and optimization
  return total_gen_loss, gan_loss, l1_loss
```
</details>

19. We also need to build the discriminator model. The discriminator is a PatchGAN model that consists of downsampling layers. The input to the discriminator is the concatenation of the input image and the target image (real or generated). The output is a feature map where each value represents whether the corresponding patch in the input image is real or fake.

Note: For details of the architecture, you can check the original paper. [Link](https://arxiv.org/pdf/1611.07004)

<details>
    <summary>Reveal</summary>
    
```python
def Discriminator():
  initializer = tf.random_normal_initializer(0., 0.02)

  # Input layers for the input image and the target image
  inp = tf.keras.layers.Input(shape=[256, 256, 3], name='input_image')
  tar = tf.keras.layers.Input(shape=[256, 256, 3], name='target_image')

  # Concatenate the input and target images along the channel axis
  # Shape: (batch_size, 256, 256, 6)
  x = tf.keras.layers.concatenate([inp, tar])

  # First downsampling block: Conv2D -> (optional) BatchNorm -> LeakyReLU
  # Reduces spatial size, increases channels
  down1 = downsample(64, 4, False)(x)  # (batch_size, 128, 128, 64)

  # Second downsampling block
  down2 = downsample(128, 4)(down1)    # (batch_size, 64, 64, 128)

  # Third downsampling block
  down3 = downsample(256, 4)(down2)    # (batch_size, 32, 32, 256)

  # Zero padding to increase spatial dimensions
  zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (batch_size, 34, 34, 256)

  # Convolution to extract features, stride=1 keeps spatial size
  conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                kernel_initializer=initializer,
                                use_bias=False)(zero_pad1)  # (batch_size, 31, 31, 512)

  # Batch normalization for stable training
  batchnorm1 = tf.keras.layers.BatchNormalization()(conv)

  # LeakyReLU activation for non-linearity
  # LeakyReLU is used instead of ReLU to avoid the "dying ReLU" problem,
  # where neurons can become inactive and only output zero. LeakyReLU allows
  # a small, non-zero gradient when the unit is not active, which helps gradients
  # flow through the network and improves training stability for GANs.
  leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)

  # Zero padding before the last layer
  zero_pad2 = tf.keras.layers.ZeroPadding2D()(leaky_relu)  # (batch_size, 33, 33, 512)

  # Final convolution: outputs a single-channel prediction map
  # Each value represents real/fake for a patch
  last = tf.keras.layers.Conv2D(1, 4, strides=1,
                                kernel_initializer=initializer)(zero_pad2)  # (batch_size, 30, 30, 1)

  # Return the Keras Model
  return tf.keras.Model(inputs=[inp, tar], outputs=last)
```
</details>

20. Let's create the discriminator model and plot it's archeticture.

<details>
    <summary>Reveal</summary>
    
```python
discriminator = Discriminator()
tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)
```
</details>

21. Aren't you curious to see how the discriminator classifies real and generated images? Let's test it by passing a sample input image and the corresponding target image (real) to the discriminator. Then, we will pass the same input image and the generated image from the generator to the discriminator.

<details>
    <summary>Reveal</summary>
    
```python
disc_out = discriminator([inp[tf.newaxis, ...], gen_output], training=False)
plt.imshow(disc_out[0, ..., -1], vmin=-20, vmax=20, cmap='RdBu_r')
plt.colorbar()
```
</details>

22. We need to create a loss function for the discriminator. The discriminator loss is calculated using binary crossentropy between the real and generated images.

<details>
    <summary>Reveal</summary>
    
```python
def discriminator_loss(disc_real_output, disc_generated_output):
  # Calculate loss for real images (should be classified as real/ones)
  real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

  # Calculate loss for generated images (should be classified as fake/zeros)
  generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

  # Total discriminator loss is the sum of real and generated losses
  total_disc_loss = real_loss + generated_loss

  return total_disc_loss
```
</details>

23. We are almost done. Now, we need to create optimizers for the generator and discriminator. We will use the Adam optimizer with a learning rate of 2e-4 and beta_1 of 0.5. We also need to create checkpoints to save the model weights during training.

<details>
    <summary>Reveal</summary>
    
```python
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)


# Directory where checkpoints will be saved during training
checkpoint_dir = './training_checkpoints'

# Prefix for checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

# Create a TensorFlow checkpoint object to manage saving and restoring models and optimizers
checkpoint = tf.train.Checkpoint(
    generator_optimizer=generator_optimizer,      # Save generator optimizer state
    discriminator_optimizer=discriminator_optimizer,  # Save discriminator optimizer state
    generator=generator,                         # Save generator model weights
    discriminator=discriminator                  # Save discriminator model weights
)
```
</details>

24. Before going into our training, we need to create a function to generate and save images during training by the generator. This will help us visualize the progress of the generator over time.

Note: You can test it on the test dataset.

<details>
    <summary>Reveal</summary>
    
```python
def generate_images(model, test_input, tar):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(15, 15))

  display_list = [test_input[0], tar[0], prediction[0]]
  title = ['Input Image', 'Ground Truth', 'Predicted Image']

  for i in range(3):
    plt.subplot(1, 3, i+1)
    plt.title(title[i])
    # Getting the pixel values in the [0, 1] range to plot.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.show()

for example_input, example_target in test_dataset.take(1):
  generate_images(generator, example_input, example_target)

```
</details>

25.  One last thing before we write our training logic, we need to store our logs. We will use TensorBoard to visualize the logs. We will create a summary writer.

<details>
    <summary>Reveal</summary>
    
```python
# Set the directory for TensorBoard logs
log_dir = "logs/"

# Create a summary writer for TensorBoard.
# The logs will be saved in a subdirectory named with the current date and time.
# This allows you to visualize training metrics in TensorBoard.
summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
```
</details>

26. Now, we are ready to write our training logic. We will create a `train_step` function that takes input and target image and step params as inputs. The function will calculate the generator and discriminator losses, compute the gradients, and apply the gradients to the optimizers. We will also log the losses to TensorBoard.

Note: Wrap it with `tf.function` to speed up the execution.


<details>
    <summary>Reveal</summary>
    
```python
@tf.function
def train_step(input_image, target, step):
  # Record operations for automatic differentiation for generator and discriminator
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    # Generate an output image from the input using the generator (forward pass)
    gen_output = generator(input_image, training=True)

    # Get discriminator's output for real image pairs (input and ground truth)
    disc_real_output = discriminator([input_image, target], training=True)
    # Get discriminator's output for fake image pairs (input and generated output)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    # Compute generator losses: total loss, GAN loss, and L1 loss
    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(
        disc_generated_output, gen_output, target)
    # Compute discriminator loss (real vs. fake)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  # Calculate gradients of generator loss w.r.t. generator's trainable variables
  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  # Calculate gradients of discriminator loss w.r.t. discriminator's trainable variables
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  # Apply gradients to update generator weights
  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  # Apply gradients to update discriminator weights
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))

  # Write loss values to TensorBoard for visualization
  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=step//1000)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step//1000)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step//1000)
    tf.summary.scalar('disc_loss', disc_loss, step=step//1000)
```
</details>

27.  Finally, we will create a `fit` function that takes training dataset, testing dataset and number of steps. The function will iterate over the epochs and steps, calling the `train_step` function for each batch of data. We will also save the model checkpoints and generate images at regular intervals.

Note: You can use the `generate_images` function to visualize the progress of the generator every 1000 steps and save the model checkpoints every 5000 steps.

<details>
    <summary>Reveal</summary>
    
```python
def fit(train_ds, test_ds, steps):
  # Get one example input and target from the test dataset for visualization
  example_input, example_target = next(iter(test_ds.take(1)))
  # Record the start time for timing training steps
  start = time.time()

  # Iterate over the training dataset for the specified number of steps
  for step, (input_image, target) in train_ds.repeat().take(steps).enumerate():
    # Every 1000 steps, clear the output and display progress
    if (step) % 1000 == 0:
      # Clear previous output in the notebook for a cleaner display
      display.clear_output(wait=True)

      # If not the first step, print the time taken for the last 1000 steps
      if step != 0:
        print(f'Time taken for 1000 steps: {time.time()-start:.2f} sec\n')

      # Reset the timer for the next 1000 steps
      start = time.time()

      # Generate and display images using the generator for visual progress
      generate_images(generator, example_input, example_target)
      # Print the current step in thousands (k)
      print(f"Step: {step//1000}k")

    # Perform one training step (update generator and discriminator)
    train_step(input_image, target, step)

    # Print a dot every 10 steps to indicate progress
    if (step+1) % 10 == 0:
      print('.', end='', flush=True)

    # Save a checkpoint every 5000 steps to preserve model state
    if (step + 1) % 5000 == 0:
      checkpoint.save(file_prefix=checkpoint_prefix)
```
</details>


28. We are almost done, can you spin up the tensorboard server in the jupyter notebook?

<details>
    <summary>Reveal</summary>
    
```python
%load_ext tensorboard
%tensorboard --logdir logs
```
</details>

29. Let's train our model for 40000 steps.

<details>
    <summary>Reveal</summary>
    
```python
fit(train_dataset, test_dataset, steps=1000)
```
</details>

30. Now, can you restore the latest checkpoint and visually test the generator?

<details>
    <summary>Reveal</summary>

```python
# Restoring the latest checkpoint in checkpoint_dir
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

# Run the trained model on a few examples from the test set
for inp, tar in test_dataset.take(5):
  generate_images(generator, inp, tar)
```
</details>


Congratulations! You have successfully implemented an autoencoders for generating MNIST dataset, image denoisining, anomaly detection and a pix2pix GAN for image-to-image translation. You can further experiment with different architectures, hyperparameters, and datasets to improve the performance of your models. Happy coding!