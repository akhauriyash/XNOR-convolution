# XNOR-convolution
Attempting to implement convolution in CUDA following XNOR-net strategy
Related resources:

[Paper on XNOR-Nets](https://arxiv.org/abs/1603.05279)

[Blog post 1](https://software.intel.com/en-us/blogs/2017/09/21/art-em-week-2)
[Blog post 2](https://software.intel.com/en-us/blogs/2017/10/02/art-em-artistic-style-transfer-to-virtual-reality-week-4-update)
[Blog post 3](https://software.intel.com/en-us/blogs/2017/10/23/art-em-artistic-style-transfer-to-virtual-reality-week-7-update)

##  Note:
  This is a work in progress. There might/should be some mistakes here. I started learning CUDA a month ago. 
  Do let me know if you find any logical errors in the code.
##  TO DO:
  - [x] Parallelize per convolution
  - [x] Parallelize over channels / Maximize throughput
  - [ ] Attempt dynamic parallelism
  - [ ] Create a full precision verification kernel
  - [ ] Add full support for custom kernel sizes

