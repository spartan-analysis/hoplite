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

./bin/pip install -r requirements.txt
# congrats! you have set up hoplite!
```

## Usage

The Hoplite class is primarily used through only a few functions after it is constructed via `h = Hoplite(model, output_filename)`.

`h.analyze(img_path_)` or `h.analyze_dir(dir_path)` will both analyze the sparsity of the respective paths.


`h.output()` will write out the gathered data to the output csv.
