# hoplite
General sparsity tests on various neural nets.

## Set Up

Requirements:
 - be able to install tensorflow 2.0

```bash
# clone the repo
git clone https://github.com/spartan-analysis/hoplite

cd hoplite

# optional, I recommend creating a virtual environment of some sort
virtualenv env
source env/bin/activate

pip install -r requirements.txt
# congrats! you have set up hoplite!
```

## Usage

The Hoplite class is primarily used through only a few functions.

```
h = Hoplite(model, output_filename, preprocess=preprocess_func, zero_sensitivity=0, max_number=500, layers=["input_1", "conv_1" ...])

h.analyze(img_path) # analyzes a single file
h.analyze_dir(img_dir_path) # analyzes an entire dir of files
h.output() # saves output csv to given output file name
```

### Hoplite constructor
 - model: any tensorflow.Keras model (NOTE: must use channel_last)
 - output_filename: the name of the file the output CSV will be saved to
 - preprocess (optional): a function that takes a path of an input file and returns the values ready to pass into the model
 - zero_sensitivity (optional): any value that has an absolute value less than this value will be considered a 0 for the sparsity statistics
 - max_number (optional): maximum number of files to analyze, Hoplite will stop accepting `analyze` functions after this max is reached
 - layers (optional): a list of str names of layers to be analyzed for sparsity
