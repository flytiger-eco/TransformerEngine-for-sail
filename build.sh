#!/bin/bash

# pip install pybind11 nvidia-mathdx
export NVTE_BUILD_THREADS_PER_JOB=4
export MAX_JOBS=48

export NVTE_WITH_NCCL_EP=0 #需要sm90及以上
export NVTE_CUDA_ARCHS="80;89"
export TORCH_CUDA_ARCH_LIST=8.0 
export NVTE_SKIP_SUBMODULE_CHECKS_DURING_BUILD=1 
export NVTE_FRAMEWORK=pytorch 

pip3 install --no-build-isolation -e . --no-deps
