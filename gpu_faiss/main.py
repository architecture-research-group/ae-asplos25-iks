import argparse
import numpy as np
import faiss
import time
import torch

d=768
dtype = np.float16
nb_50 = 32_552_083 # Vectors for 50 GB


parser = argparse.ArgumentParser()

parser.add_argument("--max_gpus", required=False, default=1)

opts = parser.parse_args()

resources = [faiss.StandardGpuResources() for i in range(opts.max_gpus)]


co = faiss.GpuMultipleClonerOptions()
co.useFloat16=True
co.shard=True




for nq in [1,16]:
    for lg_gpus in range(4): #Up to 3, i.e., 2**3 = 8
        if 2**lg_gpus > opts.max_gpus:
            break

        for size in [1,2,4,8,16, 50,200,512,1024,2048]:
            xb = np.random.randn(round(size*nb_50/50), d).astype(dtype)
            index_flat = faiss.IndexFlatIP(d)
            index_flat.add(xb)
            gpu_index_flat = faiss.index_cpu_to_gpu_multiple_py(resources, index_flat, co)
            xq = np.random.randn(nq,d).astype(np.float32) #Faiss does not support float16 here
            gpu_index_flat.search(xq,32) #Do once as a warmup
            s = time.time()
            gpu_index_flat.search(xq,32)
            e = time.time()
            elapsed = e-s

            print(f"Total: {elapsed}s")
            corpus_size = size
            print(f"Corpus: {corpus_size} GB")
