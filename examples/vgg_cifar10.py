# System imports
import os
import sys
import numpy as np
from sklearn.metrics import accuracy_score

# TensorFlow imports
from tensorflow.keras.datasets import cifar10

# Akida models imports
from akida_models import ds_cnn_cifar10, vgg_cifar10

# CNN2SNN
from cnn2snn import convert

from hoplite import Hoplite

# Load CIFAR10 dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Reshape x-data
x_train = x_train.reshape(50000, 32, 32, 3)
x_test = x_test.reshape(10000, 32, 32, 3)
input_shape = (32, 32, 3)

# Set aside raw test data for use with Akida Execution Engine later
raw_x_test = x_test.astype("uint8")

# Rescale x-data
a = 255
b = 0
input_scaling = (a, b)
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")
x_train = (x_train - b) / a
x_test = (x_test - b) / a

# Instantiate the quantized model
model_keras = vgg_cifar10(
    input_shape,
    weights="cifar10",
    weight_quantization=2,
    activ_quantization=2,
    input_weight_quantization=2,
)

num_images = 10000

hoplite = Hoplite(model_keras, "vgg_cifar10.csv", zero_sensitivity=10 ** -6)

# Check Model performance
potentials_keras = model_keras.predict(x_test[:num_images])
for x in x_test:
    hoplite.analyze_raw(x)

hoplite.output()
