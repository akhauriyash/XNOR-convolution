#include <cuda_runtime.h>
#include <iostream>
#include <chrono>
#include <cublas_v2.h>

using namespace std;


__global__ void bin(float* Img_d, unsigned int *Ker_d, float* Out_d, float* coeffD_img,
					 float* coeffD_ker, int w_img, int h_img, int w_ker,
					int h_ker, int inp_ch, int ker_depth, int dtypelen){

	__shared__ unsigned int x; x = blockIdx.x * blockDim.x + threadIdx.x;
	__shared__ unsigned int y; y = blockIdx.y * blockDim.y + threadIdx.y;
	__shared__ unsigned int pos; pos = blockIdx.x + blockIdx.y*16;
	__shared__ unsigned int idx;
	__shared__ unsigned int accum[256];
	
	idx = x + y*(w_img - w_ker + 1);

	if((x < (w_img - w_ker + 1) && (y < (h_img - h_ker + 1)))){
		for(int kd = 0; kd < ker_depth; kd++){
			for(int ic = 0; ic < inp_ch; ic++){
				accum[pos] = 0;
				for(int dx = 0; dx < w_ker; dx++){
					for(int dy = 0; dy < h_ker; dy++){
						accum[pos] = accum[pos] | ((Img_d[ic*w_img*h_img + x + dx + w_img*(y+dy)] >= 0) << ((w_ker*h_ker - 1) - (dx + w_ker*dy)));
					}
				}
				Out_d[idx + kd*(w_img-w_ker+1)*(h_img-h_ker+1)] += coeffD_ker[ic+kd*inp_ch]*coeffD_img[ic]*(2*(dtypelen - (__popc(accum[pos]^Ker_d[ic+kd*inp_ch]))) - (dtypelen - w_ker*h_ker));
			}
		}
	}
	__syncthreads();
}	

int main()	{
	cudaError_t a, b, c, d, e, f;
	
	//		Input dimensions

	int w_img = 256;	int h_img = 256;	int inp_ch = 256;
	int w_ker = 4;		int h_ker = 4;		int ker_depth = 256;

	printf("\n\nw_img %d h_img %d inp_ch %d w_ker %d h_ker %d ker_depth %d\n\n", w_img, h_img, inp_ch, w_ker, h_ker, ker_depth);

	//		HOST Memory allocation
	
	float* Img_h = (float *) malloc(w_img*h_img*inp_ch*sizeof(float));
	float* Ker_h = (float *) malloc(w_ker*h_ker*inp_ch*ker_depth*sizeof(float));
	float* Out_h = (float *) malloc((w_img - w_ker + 1)*(h_img - h_ker + 1)*ker_depth*sizeof(float));
	float* Out_h2 = (float *) malloc((w_img - w_ker + 1)*(h_img - h_ker + 1)*ker_depth*sizeof(float));
	float* coeffH_ker = (float *) malloc(inp_ch*ker_depth*sizeof(float));
	float* coeffH_img = (float *) malloc(inp_ch*sizeof(float));
 	for (int i = 0; i < w_img*h_img*inp_ch ; i ++) {
		double x = (double)rand() / RAND_MAX;
		Img_h[i] = (x > 0.5) ? 1 : -1;			
	}
	for (int i = 0; i < w_ker*h_ker*inp_ch*ker_depth ; i ++) {
		double x = (double)rand() / RAND_MAX;
		Ker_h[i] = (x < 0.5) ? -1 : 1;
	}
	for(int i = 0; i < inp_ch; i++){
		double x = (double)rand() / RAND_MAX;
		coeffH_img[i] = x;
		for(int j = 0; j < ker_depth; j++){
			double x = (double)rand() / RAND_MAX;
			coeffH_ker[i*j] = x;
		}
	}

	//		DEVICE Memory allocation 

	float *Img_d, *Out_d;	unsigned int *Ker_d;		float *coeffD_ker, *coeffD_img;

	float *Kers;
	cudaMalloc(&Kers, w_ker*h_ker*inp_ch*ker_depth*sizeof(float));
	cudaMemcpy(Kers, Ker_h, w_ker*h_ker*inp_ch*ker_depth*sizeof(float), cudaMemcpyHostToDevice);

	a = cudaMalloc(&Img_d, w_img*h_img*inp_ch*sizeof(float));												//	FP Image memory block DEVICE
	b = cudaMalloc(&Ker_d, inp_ch*ker_depth*sizeof(unsigned int));											//	Ker_d cudaMemcpy from Kconc
	c = cudaMalloc(&Out_d, (w_img - w_ker + 1)*(h_img - h_ker + 1)*ker_depth*sizeof(float));				//	FP Output memory block DEVICE
	d = cudaMemcpy(Img_d, Img_h, w_img*h_img*inp_ch*sizeof(float), cudaMemcpyHostToDevice);					//	FP Image copy from HOST to DEVICE
	e = cudaMemset(Ker_d, 0, inp_ch*ker_depth*sizeof(unsigned int));										//	Memset (temporary)
	f = cudaMemset(Out_d, 0, (w_img - w_ker + 1)*(h_img - h_ker + 1)*ker_depth*sizeof(float));				//	Memset (temporary)
	cout << "Mallocs Memcpy & Memset:\t "<< a << b << c << d << e << f << "\n";
	unsigned int *Img_conc;																					//
	a = cudaMalloc(&Img_conc, (w_img - w_ker + 1)*(h_img - h_ker + 1)*ker_depth*sizeof(unsigned int));		//	Uint Image memory block DEVICE
	b = cudaMemset(Img_conc, 0, (w_img - w_ker + 1)*(h_img - h_ker + 1)*ker_depth*sizeof(unsigned int));	//	Uint Image memory setting
	c = cudaMalloc(&coeffD_img, inp_ch*sizeof(float));
	d = cudaMalloc(&coeffD_ker, inp_ch*ker_depth*sizeof(float));
	e = cudaMemcpy(coeffD_img, coeffH_img, inp_ch*sizeof(float), cudaMemcpyHostToDevice);
	f = cudaMemcpy(coeffD_ker, coeffH_ker, inp_ch*ker_depth*sizeof(float), cudaMemcpyHostToDevice);

	unsigned int* Ker_conc = (unsigned int *) malloc(inp_ch*ker_depth*sizeof(unsigned int));				//	Memset (IMPORTANT)
	if(Ker_conc == NULL){		printf("Ker_conc MALLOC FAILURE\n");	} 
			else 		{		printf("Ker_conc MALLOC SUCCESS\n");	}
	cout << "Img_conc malloc stat:\t" << a << b << "\n" << "Ker&Img malloc & memcpy stat:\t" << c << d 
						<< e << f << "\n";

	int blockx = 16;				int blocky = 16;														//	Block config (Is maximum)
	dim3 block(blockx, blocky);		dim3 grid(w_img/blockx + 1, h_img/blocky + 1);							//	Grid config

	auto conv_xnor = [&](){
		//		Concatenate kernels to unsigned int array Kconc[inp_ch*ker_depth]
		//		cudaMalloc contiguous memory block for array
		//		cudaMemcpy Kconc to Ker_d
		// unsigned int Kconc[ker_depth*inp_ch] = {0};
		unsigned int * Kconc = (unsigned int *) malloc(ker_depth*inp_ch*sizeof(unsigned int));
		if(Kconc == NULL){
			printf("Kconc MALLOC FAILURE\n");
		} else {printf("Kconc MALLOC SUCCESS\n");}

		//		Data arrangement				(w_ker, h_ker, inp_ch, ker_depth)
		for(int kd = 0; kd < ker_depth; kd++){
			for(int ic = 0; ic < inp_ch; ic++){
				for(int shift = 0; shift < w_ker*h_ker; shift++){
					Kconc[ic + inp_ch*kd] = Kconc[ic + inp_ch*kd] | ((Ker_h[shift + ic*(w_ker*h_ker) + kd*(ic*w_ker*h_ker)]>0) << (w_ker*h_ker - 1 - shift));
				}
				for(int shift = w_ker*h_ker; shift < 32; shift++){											//	Handle 32 to sizeof(dtype) in *bits*
					Kconc[ic + inp_ch*kd] = (Kconc[ic + inp_ch*kd] | (1<<shift));
				}
			}
		} 
		// for(int i = 0; i < ker_depth*inp_ch; i++){		cout << Kconc[i] << "\tCheck\n";	}
		a = cudaMemcpy(Ker_d, Kconc, inp_ch*ker_depth*sizeof(unsigned int), cudaMemcpyHostToDevice);
		cout << "Host to device Kconc-Ker_d:\t" << a << "\n";

		//		CudaMalloc and cudaMemcpy for Kernels on DEVICE is DONE Img_d is ALLOCATED
		//		call binConv function
		//		cudaDeviceSynchronize, assign result contiguous memory block on HOST
		//		cudaMemcpy result 

		int dtypelen = 32;																				//	Handle dtypelen to wk*hk bounding 
		cudaDeviceSynchronize();

		bin<<<grid, block>>>(Img_d, Ker_d, Out_d, coeffD_img, coeffD_ker, w_img, h_img, w_ker, h_ker, inp_ch, ker_depth, dtypelen);

		cudaDeviceSynchronize();

		a = cudaMemcpy(Out_h, Out_d, (w_img - w_ker + 1)*(h_img - h_ker + 1)*ker_depth*sizeof(float), cudaMemcpyDeviceToHost);

		cout << "bin Memcpy result:\t" << a << "\n";

		for(int i = 0; i < 25; i++){cout << Out_h[i] << " ";}		printf("\n");
	};
	
	conv_xnor();
}
