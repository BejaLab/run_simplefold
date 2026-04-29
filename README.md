run\_simplefold
==============

A [SimpleFold](https://github.com/apple/ml-simplefold) wrapper.

Installation
------------

In an environmental with `python3.10`:

```
pip install \
    git+https://github.com/apple/ml-simplefold \
    git+https://github.com/BejaLab/run_simplefold
```

How to use
----------

```
#
# Initialize the model files and the database
simplefold_init -D data_dir -m simplefold_360M
#
# Run the inference using a set of seeds
simplefold_run -D data_dir -m simplefold_360M -i path/to/input.fasta -O path/to/run/ -g 0,1 -s 234,9,103,60,5,79,232,229,45,51 -l path/to/log.txt
#
# Select the best structure per protein
simplefold_select -I path/to/run/ -O path/to/select/
```

simplefold\_init
----------------

```
usage: simplefold_init [-h] -D DATA_DIR [-m {simplefold_100M,simplefold_360M,simplefold_700M,simplefold_1.1B,simplefold_1.6B,simplefold_3B}]

SimpleFold wrapper: Initialize

options:
  -h, --help            show this help message and exit
  -D DATA_DIR, --data-dir DATA_DIR
                        Base directory for data
  -m {simplefold_100M,simplefold_360M,simplefold_700M,simplefold_1.1B,simplefold_1.6B,simplefold_3B}, --model {simplefold_100M,simplefold_360M,simplefold_700M,simplefold_1.1B,simplefold_1.6B,simplefold_3B}
                        Model name (optional)
```

simplefold\_run
---------------

```
usage: simplefold_run [-h] -i INPUT -O OUTPUT -D DATA_DIR -m {simplefold_100M,simplefold_360M,simplefold_700M,simplefold_1.1B,simplefold_1.6B,simplefold_3B} [-g GPUS] [-s SEED]
                      [-b BATCH] [-l LOG] [--tau TAU] [--steps STEPS]

SimpleFold wrapper: Run

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Path to input sequences
  -O OUTPUT, --output OUTPUT
                        Output directory
  -D DATA_DIR, --data-dir DATA_DIR
                        Base directory for data
  -m simplefold_{100M,360M,700M,1.1B,1.6B,3B}, --model simplefold_{100M,360M,700M,1.1B,1.6B,3B}
                        Model name to use for inference
  -g GPUS, --gpus GPUS  GPU indices to use
  -s SEED, --seed SEED  Seed
  -b BATCH, --batch BATCH
                        Batch size
  -l LOG, --log LOG     Raw log file
  --tau TAU
  --steps STEPS
```

simplefold\_select
------------------

```
usage: simplefold_select [-h] -I INPUT -O OUTPUT [-l] [--no-seed-suffix]

SimpleFold wrapper: Select

options:
  -h, --help            show this help message and exit
  -I INPUT, --input INPUT
                        Directory containing 'run' outputs
  -O OUTPUT, --output OUTPUT
                        Output directory for the best models
  -l, --soft-link       Soft link instead of hard copy
  --no-seed-suffix      The input files do not contain the seed as the suffix
```
