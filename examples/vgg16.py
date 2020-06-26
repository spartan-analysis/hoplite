#!./bin/python
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
from hoplite import Hoplite
import numpy as np
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-o", dest="out", help="output file name")
parser.add_argument("-m", dest="max", help="max number of images")
parser.add_argument("-d", dest="dir", help="directory to analyze")

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

# needs outfile_name, max # images, and dir name
args = parser.parse_args()

h = Hoplite(
    model,
    args.out,
    preprocess=vgg16_preprocess,
    zero_sensitivity=10 ** -6,
    max_number=int(args.max),
)
h.analyze_dir(args.dir)
h.output()
