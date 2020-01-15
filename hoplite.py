from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import csv
import os


class LayerData:
    """ Stores Data about Layers """

    def __init__(self, name, dimensions):
        self.name = name
        self.dimensions = dimensions

        self.average_sparsity = 0

        self.row_hist = []
        self.col_hist = []
        self.chan_hist = []

        self.vec4_row_hist = []
        self.vec4_col_hist = []
        self.vec4_chan_hist = []

        self.vec8_row_hist = []
        self.vec8_col_hist = []
        self.vec8_chan_hist = []

        self.vec16_row_hist = []
        self.vec16_col_hist = []
        self.vec16_chan_hist = []

        self.vec32_row_hist = []
        self.vec32_col_hist = []
        self.vec32_chan_hist = []

    def average(self, other):
        if self.name is not other.name:
            print("warning: averaging different layers???")

        self.average_sparsity = (self.average_sparsity + other.average_sparsity) / 2

        self.row_hist = np.mean(
            np.array([self.row_hist, other.row_hist]), axis=0
        ).tolist()
        self.col_hist = np.mean(
            np.array([self.col_hist, other.col_hist]), axis=0
        ).tolist()
        self.chan_hist = np.mean(
            np.array([self.chan_hist, other.chan_hist]), axis=0
        ).tolist()

        self.vec4_row_hist = np.mean(
            np.array([self.vec4_row_hist, other.vec4_row_hist]), axis=0
        ).tolist()
        self.vec4_col_hist = np.mean(
            np.array([self.vec4_col_hist, other.vec4_col_hist]), axis=0
        ).tolist()
        self.vec4_chan_hist = np.mean(
            np.array([self.vec4_chan_hist, other.vec4_chan_hist]), axis=0
        ).tolist()

        self.vec8_row_hist = np.mean(
            np.array([self.vec8_row_hist, other.vec8_row_hist]), axis=0
        ).tolist()
        self.vec8_col_hist = np.mean(
            np.array([self.vec8_col_hist, other.vec8_col_hist]), axis=0
        ).tolist()
        self.vec8_chan_hist = np.mean(
            np.array([self.vec8_chan_hist, other.vec8_chan_hist]), axis=0
        ).tolist()

        self.vec16_row_hist = np.mean(
            np.array([self.vec16_row_hist, other.vec16_row_hist]), axis=0
        ).tolist()
        self.vec16_col_hist = np.mean(
            np.array([self.vec16_col_hist, other.vec16_col_hist]), axis=0
        ).tolist()
        self.vec16_chan_hist = np.mean(
            np.array([self.vec16_chan_hist, other.vec16_chan_hist]), axis=0
        ).tolist()

        self.vec32_row_hist = np.mean(
            np.array([self.vec32_row_hist, other.vec32_row_hist]), axis=0
        ).tolist()
        self.vec32_col_hist = np.mean(
            np.array([self.vec32_col_hist, other.vec32_col_hist]), axis=0
        ).tolist()
        self.vec32_chan_hist = np.mean(
            np.array([self.vec32_chan_hist, other.vec32_chan_hist]), axis=0
        ).tolist()


# example preprocess for vgg16 TODO improve this structure
# returns input ready to be processed
def vgg16_preprocess(path):
    img = image.load_img(path, target_size=(224, 244))
    return preprocess_input(np.expand_dims(image.img_to_array(img), axis=0))


# TODO change the data gathering to count small values as zeroes as well
# instead of just exact 0s
class Hoplite:
    """Hoplite Sparsity Analyzer"""

    # preprocess is a function
    def __init__(self, model, preprocess, output_filename):
        self.model = model
        # relevant layers are conv and input
        self.layers = [
            k.name for k in self.model.layers if "conv" in k.name or "input" in k.name
        ]
        self.output_filename = output_filename
        self.preprocess = preprocess

        self.input_layer_data = 0
        self.conv_layers_data = {}

    def compute_average_sparsity(self, output):
        return output.size - np.count_nonzero(output)

    @staticmethod
    def consec_1d(arr, hist):
        all_nonzeroes = True
        count = 0
        for a in range(len(arr)):
            end = a == (len(arr) - 1)
            if arr[a] == 0:
                all_nonzeroes = False
                count += 1
                if end:
                    hist[count] += 1
            else:
                if count != 0:
                    hist[count] += 1
                    count = 0
                if end and all_nonzeroes:
                    hist[0] += 1

    @staticmethod
    def consec_row(output):
        row_hist = [0] * (len(output[0][0]) + 1)
        print(len(output))
        np.apply_along_axis(Hoplite.consec_1d, 2, output, row_hist)
        return row_hist

    @staticmethod
    def consec_col(output):
        col_hist = [0] * (len(output[0]))
        np.apply_along_axis(Hoplite.consec_1d, 1, output, col_hist)
        return col_hist

    @staticmethod
    def consec_chan(output):
        chan_hist = [0] * (len(output) + 1)
        np.apply_along_axis(Hoplite.consec_1d, 0, output, chan_hist)
        return chan_hist

    @staticmethod
    def vec_1d(arr, vec_size, hist):
        if len(arr) < vec_size:
            return

        chunks = np.array_split(arr, vec_size)
        for chunk in chunks:
            zeroes = len(chunk) - np.count_nonzero(arr)
            hist[zeroes] += 1

    @staticmethod
    def vec_3d_row(output, vec_size):
        vec_row_hist = [0] * (vec_size + 1)
        np.apply_along_axis(Hoplite.vec_1d, 2, output, vec_size, vec_row_hist)
        return vec_row_hist

    @staticmethod
    def vec_3d_col(output, vec_size):
        vec_col_hist = [0] * (vec_size + 1)
        np.apply_along_axis(Hoplite.vec_1d, 1, output, vec_size, vec_col_hist)
        return vec_col_hist

    @staticmethod
    def vec_3d_chan(output, vec_size):
        vec_chan_hist = [0] * (vec_size + 1)
        np.apply_along_axis(Hoplite.vec_1d, 0, output, vec_size, vec_chan_hist)
        return vec_chan_hist

    def analyze(self, filename):
        x = self.preprocess(filename)

        for layer in self.layers:
            layer_model = Model(
                inputs=self.model.inputs, outputs=self.model.get_layer(layer).output
            )
            output = layer_model.predict(x)

            # for input layers
            if self.input_layer_data == 0 and "input" in layer:
                name = self.layers[0]
                self.input_layer_data = LayerData(
                    name, self.model.layers[0].output_shape[0][1:]
                )

                self.input_layer_data.average_sparsity = Hoplite.compute_average_sparsity(
                    output
                )

            # for conv layers
            temp = LayerData(layer, self.model.get_layer(layer).output_shape[1:])
            temp.average_sparsity = Hoplite.compute_average_sparsity(x)
            temp.row_hist = Hoplite.consec_row(output)
            temp.col_hist = Hoplite.consec_col(output)
            temp.chan_hist = Hoplite.consec_chan(output)

            temp.vec4_row_hist = self.vec_3d_row(output, 4)
            temp.vec4_col_hist = self.vec_3d_col(output, 4)
            temp.vec4_chan_hist = self.vec_3d_chan(output, 4)

            temp.vec8_row_hist = self.vec_3d_row(output, 8)
            temp.vec8_col_hist = self.vec_3d_col(output, 8)
            temp.vec8_chan_hist = self.vec_3d_chan(output, 8)

            temp.vec16_row_hist = self.vec_3d_row(output, 16)
            temp.vec16_col_hist = self.vec_3d_col(output, 16)
            temp.vec16_chan_hist = self.vec_3d_chan(output, 16)

            temp.vec32_row_hist = self.vec_3d_row(output, 32)
            temp.vec32_col_hist = self.vec_3d_col(output, 32)
            temp.vec32_chan_hist = self.vec_3d_chan(output, 32)

            if self.conv_layers_data[layer] is None:
                self.conv_layers_data[layer] = temp
            else:
                self.conv_layers_data[layer].average(temp)

    def analyze_dir(self, dir_name):
        for (dirpath, dirnames, filenames) in os.walk(dir_name):
            for filename in filenames:
                self.analyze(filename)

    def output(self):
        with open(self.output_filename, "w", newline="") as csv_out:
            writer = csv.writer(csv_out, delimiter=",")

            # input layer
            writer.writerow(["layer=", self.input_layer_data.name])
            writer.writerow(["dimensions=", self.input_layer_data.dimensions])
            writer.writerow(["average=", self.input_layer_data.average_sparsity])

            # output conv layers
            for layer in self.layers[1:]:
                writer.writerow(["layer=", layer])
                writer.writerow(
                    ["dimensions=", self.conv_layers_data[layer].dimensions]
                )
                writer.writerow(
                    ["average=", self.conv_layers_data[layer].average_sparsity]
                )

                writer.writerow(["row_hist=", self.conv_layers_data[layer].row_hist])
                writer.writerow(["col_hist=", self.conv_layers_data[layer].col_hist])
                writer.writerow(["chan_hist=", self.conv_layers_data[layer].chan_hist])

                writer.writerow(["vector=4"])
                writer.writerow(
                    ["vec4_row_hist=", self.conv_layers_data[layer].vec4_row_hist]
                )
                writer.writerow(
                    ["vec4_col_hist=", self.conv_layers_data[layer].vec4_col_hist]
                )
                writer.writerow(
                    ["vec4_chan_hist=", self.conv_layers_data[layer].vec4_chan_hist]
                )

                writer.writerow(["vector=8"])
                writer.writerow(
                    ["vec8_row_hist=", self.conv_layers_data[layer].vec8_row_hist]
                )
                writer.writerow(
                    ["vec8_col_hist=", self.conv_layers_data[layer].vec8_col_hist]
                )
                writer.writerow(
                    ["vec8_chan_hist=", self.conv_layers_data[layer].vec8_chan_hist]
                )

                writer.writerow(["vector=16"])
                writer.writerow(
                    ["vec16_row_hist=", self.conv_layers_data[layer].vec16_row_hist]
                )
                writer.writerow(
                    ["vec16_col_hist=", self.conv_layers_data[layer].vec16_col_hist]
                )
                writer.writerow(
                    ["vec16_chan_hist=", self.conv_layers_data[layer].vec16_chan_hist]
                )

                writer.writerow(["vector=32"])
                writer.writerow(
                    ["vec32_row_hist=", self.conv_layers_data[layer].vec32_row_hist]
                )
                writer.writerow(
                    ["vec32_col_hist=", self.conv_layers_data[layer].vec32_col_hist]
                )
                writer.writerow(
                    ["vec32_chan_hist=", self.conv_layers_data[layer].vec32_chan_hist]
                )
