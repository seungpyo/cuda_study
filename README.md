# cuda_study

This repo contains some example codes that I wrote to study CUDA environment.

## hello.cu

A typcial hello world example.

## cuutils.h

Some utility functins are in this header file. `CUUTIL_DEBUG` is the bolierplate CUDA error checking macro, which is heavily used in this repo.

## matmul_tiled.cu

Tiled matrix multiplcation example. Running this code will show the comparison result between naive matrix multiplication and tiled matrix multiplication. What a beautiful technique!
