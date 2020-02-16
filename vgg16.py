#!./bin/python
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from hoplite import Hoplite
import numpy as np

import cProfile

# returns input ready to be processed
def vgg16_preprocess(path):
    img = image.load_img(path, target_size=(224, 224))
    return preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))


model = VGG16(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
)
h = Hoplite(model, vgg16_preprocess, "output.csv", 10 ** -6, 400)

# h.analyze("test.png")
cProfile.run('h.analyze("test.png")')
h.output()
