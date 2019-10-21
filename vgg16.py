from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import matplotlib.pyplot as plt
from sparsity import calculate_sparsity


tiny_imgnet = "./tiny-imagenet-200/test/images/"
test_number = 10

model = VGG16(
    include_top=True,
    weights="imagenet",
    input_tensor=None,
    input_shape=None,
    pooling=None,
    classes=1000,
)

sparsities = []

for i in range(test_number):
    img_path = tiny_imgnet + "test_{}.JPEG".format(i)
    img = image.load_img(img_path, target_size=(224, 224))
    x = preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))
    test_sparsities = calculate_sparsity(model, x)

    index = 0
    for j in test_sparsities:
        if index < len(sparsities):
            sparsities[index] = sparsities[index] + j
        else:
            sparsities.append(j)
        index = index + 1

sparsities = [x / test_number for x in sparsities]

print(sparsities)

fig = plt.figure()
plt.bar(range(len(sparsities)), sparsities)
plt.title("Sparsity by CONV Layer")
fig.savefig("plot.png")
plt.show()
