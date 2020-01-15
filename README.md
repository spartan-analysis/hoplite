# hoplite
general sparsity tests on various neural nets

## Set Up

Requirements:
 - virtualenv
 - be able to install tensorflow 2.0 (for whatever reason I wasn't able to get arch linux working yet, but will work on it soon)

 ```bash
 git clone https://github.com/spartan-analysis/hoplite # clone the repo

cd hoplite

virtualenv . # some versions of virtualenv need you to specify python 3.0, if so, do so

./bin/pip install -r requirements.txt #congrats! you have set up hoplite... more documentation will be coming in the future.```

## Usage

The Hoplite class is primarily used through only a few functions after it is constructed via `h = Hoplite(model, output_filename)`.

`h.analyze(img_path_)` or `h.analyze_dir(dir_path)` will both analyze the sparsity of the respective paths.


`h.output()` will write out the gathered data to the output csv.
