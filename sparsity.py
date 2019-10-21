from tensorflow.keras.models import Model
import numpy as np


def calculate_sparsity(model, input):
    sparsities = []

    # only convolution layers
    conv_layers = [k.name for k in model.layers if "conv" in k.name]

    for layer in conv_layers:
        layer_model = Model(inputs=model.inputs, outputs=model.get_layer(layer).output)

        output = layer_model.predict(input)
        zeroes = output.size - np.count_nonzero(output)

        sparsities.append(zeroes)

    return sparsities
