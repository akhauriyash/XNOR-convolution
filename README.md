# XNOR convolution
Attempting to implement convolution in CUDA following XNOR-net strategy.

##  Prerequisites:
  * CUDA
  * CUDA capable GPU
  
##  To run:
  Navigate to the directory where xnorconv.cu is located. 
  
  `nvcc -arch=sm_50 xnorconv.cu -std=c++11 && ./a.out`
  
  To profile the application:
  
  `nvprof ./a.out`
  
  
##  Note:
  This is a work in progress. There might/should be some mistakes here. I started learning CUDA a month ago. 
  Do let me know if you find any logical errors in the code.

##  TO DO:
  - [x] Add support for variable input sizes
  - [x] Add support for 3D convolution
  - [x] Parallelize per convolution
  - [ ] Add code/function for general matrix multiplication (Already created, PM for code.)
  - [ ] Maximize shared memory usage - balance channel parallelization
  - [ ] Create a full precision verification kernel
  - [ ] Add full support for custom kernel sizes
  - [ ] Build a parser to take in shape arguments


## Related/Relevant resources:

[Paper on XNOR-Nets](https://arxiv.org/abs/1603.05279)

[Blog post 1](https://software.intel.com/en-us/blogs/2017/09/21/art-em-week-2)

[Blog post 2](https://software.intel.com/en-us/blogs/2017/10/02/art-em-artistic-style-transfer-to-virtual-reality-week-4-update)

[Blog post 3](https://software.intel.com/en-us/blogs/2017/10/23/art-em-artistic-style-transfer-to-virtual-reality-week-7-update)

[BinaryNet](https://github.com/MatthieuCourbariaux/BinaryNet)

[XNOR-Net - AllenAI](https://github.com/allenai/XNOR-Net)
