from tensorflow.keras.models import Model
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

def average_layer_data(data_list):
    temp = LayerData(data_list[0].name, data_list[0].dimensions)

    temp.average_sparsity = np.mean([x.average_sparsity for x in data_list])

    temp.row_hist = np.mean(
        np.array([x.row_hist for x in data_list]), axis=0
    ).tolist()
    temp.col_hist = np.mean(
        np.array([x.col_hist for x in data_list]), axis=0
    ).tolist()
    temp.chan_hist = np.mean(
        np.array([x.chan_hist for x in data_list]), axis=0
    ).tolist()

    temp.vec4_row_hist = np.mean(
        np.array([x.vec4_row_hist for x in data_list]), axis=0
    ).tolist()
    temp.vec4_col_hist = np.mean(
        np.array([x.vec4_col_hist for x in data_list]), axis=0
    ).tolist()
    temp.vec4_chan_hist = np.mean(
        np.array([x.vec4_chan_hist for x in data_list]), axis=0
    ).tolist()

    temp.vec8_row_hist = np.mean(
        np.array([x.vec8_row_hist for x in data_list]), axis=0
    ).tolist()
    temp.vec8_col_hist = np.mean(
        np.array([x.vec8_col_hist for x in data_list]), axis=0
    ).tolist()
    temp.vec8_chan_hist = np.mean(
        np.array([x.vec8_chan_hist for x in data_list]), axis=0
    ).tolist()


    temp.vec16_row_hist = np.mean(
        np.array([x.vec16_row_hist for x in data_list]), axis=0
    ).tolist()
    temp.vec16_col_hist = np.mean(
        np.array([x.vec16_col_hist for x in data_list]), axis=0
    ).tolist()
    temp.vec16_chan_hist = np.mean(
        np.array([x.vec16_chan_hist for x in data_list]), axis=0
    ).tolist()


    temp.vec32_row_hist = np.mean(
        np.array([x.vec32_row_hist for x in data_list]), axis=0
    ).tolist()
    temp.vec32_col_hist = np.mean(
        np.array([x.vec32_col_hist for x in data_list]), axis=0
    ).tolist()
    temp.vec32_chan_hist = np.mean(
        np.array([x.vec32_chan_hist for x in data_list]), axis=0
    ).tolist()

    return temp

class Hoplite:
    """Hoplite Sparsity Analyzer"""

    # preprocess is a function
    def __init__(
        self, model, preprocess, output_filename, zero_sensitivity=0, max_number=None
    ):
        self.model = model
        # relevant layers are conv and input
        self.layers = [
            k.name for k in self.model.layers if "conv" in k.name or "input" in k.name
        ]
        self.output_filename = output_filename
        self.preprocess = preprocess

        self.input_layer_data = 0
        self.conv_layers_data = {}
        self.zero_sensitivity = zero_sensitivity
        self.counter = 0
        self.max_number = max_number

    def equals_zero(self, number):
        return abs(number) <= self.zero_sensitivity

    def compute_average_sparsity(self, output):
        count = 0
        for chan in output:
            for col in chan:
                for number in col:
                    if self.equals_zero(number):
                        count += 1
        return float(count) / output.size

    def consec_1d(self, arr, hist):
        all_nonzeroes = True
        count = 0
        for a in range(len(arr)):
            end = a == (len(arr) - 1)
            if self.equals_zero(arr[a]):
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

    def consec_row(self, output):
        row_hist = [0] * (len(output[0][0]) + 1)
        np.apply_along_axis(self.consec_1d, 2, output, row_hist)
        return row_hist

    def consec_col(self, output):
        col_hist = [0] * (len(output[0]) + 1)
        np.apply_along_axis(self.consec_1d, 1, output, col_hist)
        return col_hist

    def consec_chan(self, output):
        chan_hist = [0] * (len(output) + 1)
        np.apply_along_axis(self.consec_1d, 0, output, chan_hist)
        return chan_hist

    @staticmethod
    def chunk_array(lst, n):
        for i in range(0, len(lst), n):
            yield lst[i : i + n]

    def vec_1d(self, arr, vec_size, hist):
        if len(arr) < vec_size:
            return

        chunks = Hoplite.chunk_array(arr, vec_size)
        for chunk in chunks:
            zeroes = 0
            for num in chunk:
                if self.equals_zero(num):
                    zeroes += 1
            hist[zeroes] += 1

    def vec_3d_row(self, output, vec_size):
        vec_row_hist = [0] * (vec_size + 5)
        np.apply_along_axis(self.vec_1d, 2, output, vec_size, vec_row_hist)
        return vec_row_hist

    def vec_3d_col(self, output, vec_size):
        vec_col_hist = [0] * (vec_size + 5)
        np.apply_along_axis(self.vec_1d, 1, output, vec_size, vec_col_hist)
        return vec_col_hist

    def vec_3d_chan(self, output, vec_size):
        vec_chan_hist = [0] * (vec_size + 5)
        np.apply_along_axis(self.vec_1d, 0, output, vec_size, vec_chan_hist)
        return vec_chan_hist

    def analyze_raw(self, data):
        if self.max_number is not None and self.counter > self.max_number:
            return  # don't analyze more than max number

        x = data

        for layer in self.layers:
            layer_model = Model(
                inputs=self.model.inputs, outputs=self.model.get_layer(layer).output
            )
            output = layer_model.predict(x)[0]

            # for input layers
            if self.input_layer_data == 0 and "input" in layer:
                self.input_layer_data = LayerData(
                    layer, self.model.layers[0].output_shape[0][1:]
                )
                self.input_layer_data.average_sparsity = self.compute_average_sparsity(
                    output
                )

            # for conv layers
            temp = LayerData(layer, self.model.get_layer(layer).output_shape[1:])
            temp.average_sparsity = self.compute_average_sparsity(output)
            temp.row_hist = self.consec_row(output)
            temp.col_hist = self.consec_col(output)
            temp.chan_hist = self.consec_chan(output)

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

            if "input" not in layer:
                # if self.conv_layers_data[layer] is None:
                if layer not in self.conv_layers_data:
                    self.conv_layers_data[layer] = [temp] # TODO MAKE INTO LIST INSTEAD OF JUST A THING
                else:
                    self.conv_layers_data[layer].append(temp)

        self.counter += 1

    def analyze(self, filename):
        print("analysing {}".format(filename))
        x = self.preprocess(filename)
        self.analyze_raw(x)


    def analyze_dir(self, dir_name):
        if self.max_number is not None and self.counter >= self.max_number:
            return
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
                current = average_layer_data(self.conv_layers_data[layer])
                writer.writerow(["layer=", layer])
                writer.writerow(
                    ["dimensions=", current.dimensions]
                )
                writer.writerow(
                    ["average=", current.average_sparsity]
                )

                writer.writerow(["row_hist=", current.row_hist])
                writer.writerow(["col_hist=", current.col_hist])
                writer.writerow(["chan_hist=", current.chan_hist])

                writer.writerow(["vector=4"])
                writer.writerow(
                    ["vec4_row_hist=", current.vec4_row_hist]
                )
                writer.writerow(
                    ["vec4_col_hist=", current.vec4_col_hist]
                )
                writer.writerow(
                    ["vec4_chan_hist=", current.vec4_chan_hist]
                )

                writer.writerow(["vector=8"])
                writer.writerow(
                    ["vec8_row_hist=", current.vec8_row_hist]
                )
                writer.writerow(
                    ["vec8_col_hist=", current.vec8_col_hist]
                )
                writer.writerow(
                    ["vec8_chan_hist=", current.vec8_chan_hist]
                )

                writer.writerow(["vector=16"])
                writer.writerow(
                    ["vec16_row_hist=", current.vec16_row_hist]
                )
                writer.writerow(
                    ["vec16_col_hist=", current.vec16_col_hist]
                )
                writer.writerow(
                    ["vec16_chan_hist=", current.vec16_chan_hist]
                )

                writer.writerow(["vector=32"])
                writer.writerow(
                    ["vec32_row_hist=", current.vec32_row_hist]
                )
                writer.writerow(
                    ["vec32_col_hist=", current.vec32_col_hist]
                )
                writer.writerow(
                    ["vec32_chan_hist=", current.vec32_chan_hist]
                )
