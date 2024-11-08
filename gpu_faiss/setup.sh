#!/bin/bash

wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

./Miniconda3-latest-Linux-x86_64.sh

conda install -c pytorch -c nvidia -c rapidsai -c conda-forge faiss-gpu-raft=1.9.0 pytorch pytorch-cuda numpy

