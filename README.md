# ae-asplos25-iks

## Cycle-approximate simulator

Directory: `cycle_approx_iks`. Details on the simulator are provided in the README of this directory.

## GPU implementation:
Directory: `gpu_faiss`.

For comparison with IKS, we run exact search on up to 8 GPUs for various corpus sizes. See `gpu_faiss`. Run `./setup.sh` to install Conda and FAISS with GPU support. After this, run with `python main.py`. The maximum number of GPUs can be set with `--max_gpus`.

## CPU implementation

Directory: `ae-aslpos25-iks-faiss`.

The default parameters of FAISS are suboptimal on many current systems. To address this, we use Intel MKL as the backend, and increase the MKL block size to 16384. A source file for a program generating the values for Fig.9 is located in the `ae_asplos25` directory. Run `./ae_setup.sh /path/to/mkl/latest/lib` to run Cmake with the relevant options, and run `./ae_fig_9.sh` to build and run the program for figure 9.
