###########not working, need help
# -*- coding: utf-8 -*-

""" 
Multiplies two square matrices together using a *single* block of threads and 
global memory only. Each thread computes one element of the resulting matrix.
"""

import numpy as np
from pycuda import driver, compiler, gpuarray, tools
import pycuda.cumath as cumath
import python_speech_features as sf
from sklearn.metrics.pairwise import euclidean_distances
from scipy.io import wavfile as wv
from array import *

# -- initialize the device
import pycuda.autoinit as pcit
# driver.init()
kernel_code_template = """
#include<stdio.h>
 __global__ void EuclidianDistances(const double *A,double *C , int n)
{
    // SIZE is equal to 13
    __shared__ double accum;
    int bx = blockIdx.x;  // n
    int by = blockIdx.y;  // m
    int tx = threadIdx.x; // 0 to 12

    if(by<=bx)
    {
        if(tx==0) accum=0;
        __syncthreads();

        double sA = A [bx * %(SIZE)s + tx];
        double sB = A [by * %(SIZE)s + tx];
        accum+=(sA-sB)*(sA-sB);
        if(bx>1830 & by ==0)
        //printf("%%lf %%lf %%lf %%d %%d %%d\\n",sA,sB,accum,bx,by, tx);
        __syncthreads();

        if(tx==0) 
        {
            C[bx * n + by]=accum;
            C[by * n + bx]=accum;
        }
    }
}                                                                                                                                                                                               
"""

# define the (square) matrix size
#  note that we'll only use *one* block of threads here
#  as a consequence this number (squared) can't exceed max_threads,
#  see http://documen.tician.de/pycuda/util.html#pycuda.tools.DeviceData
#  for more information on how to get this number for your device


# create two random square matrices
fs,s=wv.read("/home/yash/Dropbox/Music Recommendation/Outputs/faint_chota.wav")
mf=sf.mfcc(s,samplerate=fs)

# compute reference on the CPU to verify GPU computation
# c_cpu = np.dot(a_cpu, b_cpu)

# transfer host (CPU) memory to device (GPU) memory 
def eucl_gpu(mf):
    a_cpu = np.asarray(mf)
    SIZE = a_cpu.shape[1]
    a_gpu = gpuarray.to_gpu(a_cpu.flatten()) 
    print("before")
    print(a_cpu)
    print("To gpu")
    print(a_gpu.get())
    print(a_cpu.shape)
    c_gpu = gpuarray.zeros((a_cpu.shape[0], a_cpu.shape[0]), np.double)
    kernel_code = kernel_code_template % {
        'SIZE': SIZE 
        }
    print(SIZE)

    # compile the kernel code 
    mod = compiler.SourceModule(kernel_code)

    # get the kernel function from the compiled module
    distance = mod.get_function("EuclidianDistances")

    # call the kernel on the card
    distance(
        # inputs
        a_gpu,
        # output
        c_gpu, 
        # number of blocks
        np.int64(a_cpu.shape[0]),
        # number of blocks
        grid = (a_cpu.shape[0],a_cpu.shape[0]),
        #number of threads
        block=(SIZE,1,1)
        )
    driver.Context.synchronize()
    # print("expected")
    # kk=np.matrix(euclidean_distances(mf))
    print("actual")
    pc=c_gpu.get()
    driver.Context.synchronize()
    return np.sqrt(pc)

print(eucl_gpu(mf))