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
usage: simplefold_run [-h] -i INPUT -O OUTPUT -D DATA_DIR -m {simplefold_100M,simplefold_360M,simplefold_700M,simplefold_1.1B,simplefold_1.6B,simplefold_3B} [-g GPUS] [-s SEEDS]
                      [-b BATCH] [-l LOG] [--tau TAU] [--steps STEPS] [--desc] [--no-cache] [--max-wait MAX_WAIT]

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
  -g GPUS, --gpus GPUS  GPUs to use
  -s SEEDS, --seeds SEEDS
                        Seeds
  -b BATCH, --batch BATCH
                        Batch size
  -l LOG, --log LOG     Raw log file
  --tau TAU
  --steps STEPS
  --desc                Consider the whole fasta header is the record name
  --no-cache            Do not use cache
  --max-wait MAX_WAIT   Seconds to wait for a busy GPU before giving up on it [43200, i.e. 12 h]
```

### GPU locking

Each batch locks its GPU, so that runs started at the same time, from any data directory, by any
user, take turns on a GPU instead of running out of memory: a batch on a GPU that is busy waits
for it, with a message naming the PID that locked it. The lock is a `flock` on the GPU's device
node (e.g. `/dev/nvidia0`), the same one taken by
[run\_alphafold3](https://github.com/BejaLab/run_alphafold3), so SimpleFold and AlphaFold3 runs
share the GPUs, and any other command can take part with `flock /dev/nvidia0 <command>`. The lock
is released between batches, and a run waiting for the GPU gets it before the run that released
it takes it again, so that a short run does not wait for a long one to finish. After `--max-wait`
seconds of waiting, the run gives up on the GPU and leaves its batches to the other GPUs; sequences
left when every GPU has been given up on fail. SimpleFold inherits the lock: the GPU stays locked
while it runs, even if the wrapper dies. GPU indices follow `nvidia-smi` (`CUDA_DEVICE_ORDER=PCI_BUS_ID`).

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
