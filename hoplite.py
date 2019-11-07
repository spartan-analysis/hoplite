from tensorflow.keras.models import Model
import numpy as np
# import csv


# def calculate_sparsity(model, input): sparsities = []
#    # only convolution layers
#    conv_layers = [k.name for k in model.layers if "conv" in k.name]
#
#    for layer in conv_layers:
#        layer_model = Model(inputs=model.inputs,
#                            outputs=model.get_layer(layer).output)
#
#        output = layer_model.predict(input)
#        zeroes = (output.size - np.count_nonzero(output))
#
#        sparsities.append(zeroes)
#
#    return sparsities

class Hoplite:

    # the average sparsity of the whole feature map cube
    average_sparsity = 0

    # histograms of consecutive 0s in rows, columns, and channels
    # ex:   row_hist[0] is the # of rows with all nonzeroes
    #       row_hist[1] is the # of rows with 1 consecutive zero
    row_hist = []
    col_hist = []
    chan_hist = []

    # histograms of the number of zeros in vectors of size 4
    vec4_row_hist = []
    vec4_col_hist = []
    vec4_chan_hist = []

    # histograms of the number of zeros in vectors of size 8
    vec8_row_hist = []
    vec8_col_hist = []
    vec8_chan_hist = []

    # histograms of the number of zeros in vectors of size 16
    vec16_row_hist = []
    vec16_col_hist = []
    vec16_chan_hist = []

    # histograms of the number of zeros in vectors of size 32
    vec32_row_hist = []
    vec32_col_hist = []
    vec32_chan_hist = []

    def __init__(self, model):
        self.model = model
        self.conv_layers = [k.name for k in model.layers if "conv" in k.name]

    def addInput(self, input):
        for layer in self.conv_layers:
            layer_model = Model(inputs=self.model.inputs,
                                outputs=self.model.get_layer(layer).output)

            output = layer_model.predict(input)
            zeroes = (output.size - np.count_nonzero(output))
