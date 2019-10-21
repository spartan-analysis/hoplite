from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
from sparsity import calculate_sparsity


tiny_imgnet = "./tiny-imagenet-200/test/images/"
img_path = tiny_imgnet + "test_0.JPEG"
img = image.load_img(img_path, target_size=(224, 224))
x = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))

model = VGG16(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
)

sparsities = calculate_sparsity(model, x)

print(sparsities)

plt.bar(range(len(sparsities)), sparsities)
plt.show()
