from tensorflow.keras.applications.vgg16 import VGG16
from hoplite import Hoplite, vgg16_preprocess

model = VGG16(include_top=True, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=1000)
h = Hoplite(model, vgg16_preprocess, "output.csv")

# TODO h.analyze some stuff

# TODO h.output file
