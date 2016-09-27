	/*********************************************************************************/
	/* Matrix product program for a multi-core CPU and for a many-core GPU           */
	/* S. Vialle - November 2015                                                     */
	/*********************************************************************************/
	
	#include <stdio.h>
	#include <stdlib.h>
	#include <cuda.h> 
	#include <cuda_runtime.h>
	
	#include "main.h"
	#include "gpu-op.h"
	
	
	/*-------------------------------------------------------------------------------*/
	/* GPU symbols                                                                   */
	/*-------------------------------------------------------------------------------*/
	__device__ double GPU_A[SIZE][SIZE];
	__device__ double GPU_B[SIZE][SIZE];
	__device__ double GPU_C[SIZE][SIZE];
	
	
	/*-------------------------------------------------------------------------------*/
	/* Init and finalize the GPU device.                                             */
	/*-------------------------------------------------------------------------------*/
	void gpuInit(void)
	{
	cuInit(0);
	}
	
	
	void gpuFinalize(void)
	{
	
	}
	
	
	/*-------------------------------------------------------------------------------*/
	/* Transfer of CPU input data into GPU symbols                                   */
	/*-------------------------------------------------------------------------------*/
	void gpuSetDataOnGPU(void)
	{
	//Set GPU_A symbol
	
	CHECK_CUDA_SUCCESS(cudaMemcpyToSymbol(GPU_A, &A[0][0], sizeof(double)*SIZE*SIZE, 0, cudaMemcpyHostToDevice),
	"Transfer A-->GPU_A");
	
	//Set GPU_B symbol
	
	CHECK_CUDA_SUCCESS(cudaMemcpyToSymbol(GPU_B, &B[0][0], sizeof(double)*SIZE*SIZE, 0, cudaMemcpyHostToDevice),
	"Transfer B-->GPU_B");
	}
	
	
	/*-------------------------------------------------------------------------------*/
	/* Transfer of GPU results into CPU array                                        */
	/*-------------------------------------------------------------------------------*/
	void gpuGetResultOnCPU(void)
	{
	// Get GPU_C symbol
	CHECK_CUDA_SUCCESS(cudaMemcpyFromSymbol(&C[0][0], GPU_C, sizeof(double)*SIZE*SIZE, 0, cudaMemcpyDeviceToHost),
	"Transfer GPU_C-->C"); 
	}
	
	
	/*-------------------------------------------------------------------------------*/
	/* Small matrix product on the local GPU.                                        */
	/*-------------------------------------------------------------------------------*/
	__global__ void MatrixProductKernel_v0(void)
	{
	// Index computations
	int col = (blockIdx.y * blockDim.y) + threadIdx.y;
	int lig = (blockIdx.x * blockDim.x) + threadIdx.x;
	
	if(col < SIZE && lig < SIZE){ 
	double res = 0.0;
	
	// Matrix product computation
	int i;
	for(i=0;i<SIZE;i++)
	{
	res += GPU_A[i][lig]*GPU_B[col][i];
	}
	
	GPU_C[col][lig] = res;
	}
	}
	
	/*-------------------------------------------------------------------------------*/
	/* Small matrix product on the local GPU.                                        */
	/*-------------------------------------------------------------------------------*/
__global__ void MatrixProductKernel_v3(void){
	// Index computations
	int i,j;
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;
	int lig = (blockIdx.y * blockDim.y) + threadIdx.y;
	if(col < SIZE && lig < SIZE){
		__shared__ double shared_A[BLOCK_SIZE_X_K3][BLOCK_SIZE_Y_K3];
		__shared__ double shared_B[BLOCK_SIZE_X_K3][BLOCK_SIZE_Y_K3];
		double res = 0.0;
		for(i = 0; i < (SIZE / BLOCK_SIZE_X_K3)+(SIZE%BLOCK_SIZE_X_K3?  1  :  0); i++){
			int offset = i * BLOCK_SIZE_X_K3;
			if((offset+threadIdx.x) < SIZE && (offset+threadIdx.y) < SIZE){
				shared_A[threadIdx.y][threadIdx.x] = GPU_A[lig][offset + threadIdx.x];
				shared_B[threadIdx.y][threadIdx.x] = GPU_B[offset + threadIdx.y][col];
				__syncthreads();	
			
			// Matrix product computation
				for(j=0;j<BLOCK_SIZE_X_K3;j++){
					if(j + offset < SIZE){
					res += shared_A[threadIdx.y][j]*shared_B[j][threadIdx.x];
					}
				}
				__syncthreads();	
				
			}
		}	
		GPU_C[lig][col] = res;
	}
}


__global__ void MatrixProductKernel_v4(void){
	// Index computations
	int i,j;
	int col = (blockIdx.x * blockDim.x) + threadIdx.x;
	int lig = (blockIdx.y * blockDim.y) + threadIdx.y;
	double res;
	__shared__ double shared_A[BLOCK_SIZE_X_K3][BLOCK_SIZE_Y_K3];
	__shared__ double shared_B[BLOCK_SIZE_X_K3][BLOCK_SIZE_Y_K3];
	res = 0.0;
	for(i = 0; i <	 (SIZE / BLOCK_SIZE_X_K3)+(SIZE%BLOCK_SIZE_X_K3?1:0); i++){
		int offset = i * BLOCK_SIZE_X_K3;
		if((offset+threadIdx.x) < SIZE && lig < SIZE){
			shared_A[threadIdx.y][threadIdx.x] = GPU_A[lig][offset + threadIdx.x];
		}
		if((offset+threadIdx.y) < SIZE && col < SIZE){
			shared_B[threadIdx.y][threadIdx.x] = GPU_B[offset + threadIdx.y][col];
		}
			__syncthreads();	
		
		// Matrix product computation
		if(offset < SIZE-1){
			for(j=0;j<BLOCK_SIZE_X_K3;j++){
				res += shared_A[threadIdx.y][j]*shared_B[j][threadIdx.x];
			}
		}else{
			for(j=0;j< (SIZE%BLOCK_SIZE_X_K3);j++){
					res += shared_A[threadIdx.y][j]*shared_B[j][threadIdx.x];			
			}			
		}
	__syncthreads();
	if(col < SIZE && lig < SIZE){
		GPU_C[lig][col] = res;
	}	
  }
}
	
	
	/*-------------------------------------------------------------------------------*/
	/* Small matrix product on the local GPU.                                        */
	/*-------------------------------------------------------------------------------*/
	void gpuProduct(gkid_t kid)
	{
	dim3 Dg, Db;
	
	switch(kid) {
	
	case GK0 : // Kernel v0 - using only global memory (with coalescent data accesses)
	// - init the grid of blocs
	Db.x = BLOCK_SIZE_X_K0;
	Db.y = BLOCK_SIZE_Y_K0;
	Db.z = 1;
	Dg.x = SIZE/BLOCK_SIZE_X_K0;
	Dg.y = SIZE/BLOCK_SIZE_Y_K0;
	Dg.z = 1;
	// - run the Grid of Blocs of threads
	MatrixProductKernel_v0<<<Dg,Db>>>();
	break;
	
	case GK1 :
	Db.x = BLOCK_SIZE_X_K0;
	Db.y = BLOCK_SIZE_Y_K0;
	Db.z = 1;
	Dg.x = SIZE/BLOCK_SIZE_X_K0 + (SIZE%BLOCK_SIZE_X_K0  ?  1  :  0);
	Dg.y = SIZE/BLOCK_SIZE_Y_K0 + (SIZE%BLOCK_SIZE_Y_K0  ?  1  :  0);
	Dg.z = 1;
	// - run the Grid of Blocs of threads
	MatrixProductKernel_v0<<<Dg,Db>>>();
	break;
	
	case GK2 :
	Db.x = BLOCK_SIZE_X_K0;
	Db.y = BLOCK_SIZE_Y_K0;
	Db.z = 1;
	Dg.x = SIZE/BLOCK_SIZE_X_K0 + (SIZE%BLOCK_SIZE_X_K0  ?  1  :  0);
	Dg.y = SIZE/BLOCK_SIZE_Y_K0 + (SIZE%BLOCK_SIZE_Y_K0  ?  1  :  0);
	Dg.z = 1;
	// - run the Grid of Blocs of threads
	MatrixProductKernel_v0<<<Dg,Db>>>();
	break;
	
	case GK3 :
	Db.x = BLOCK_SIZE_X_K3;
	Db.y = BLOCK_SIZE_Y_K3;
	Db.z = 1;
	Dg.x = SIZE/BLOCK_SIZE_X_K3 + (SIZE%BLOCK_SIZE_X_K3  ?  1  :  0);
	Dg.y = SIZE/BLOCK_SIZE_Y_K3 + (SIZE%BLOCK_SIZE_Y_K3  ?  1  :  0);
	Dg.z = 1;
	// - run the Grid of Blocs of threads
	MatrixProductKernel_v3<<<Dg,Db>>>();
	break;
	
	case GK4 :
	Db.x = BLOCK_SIZE_X_K3;
	Db.y = BLOCK_SIZE_Y_K3;
	Db.z = 1;
	Dg.x = SIZE/BLOCK_SIZE_X_K3 + (SIZE%BLOCK_SIZE_X_K3  ?  1  :  0);
	Dg.y = SIZE/BLOCK_SIZE_Y_K3 + (SIZE%BLOCK_SIZE_Y_K3  ?  1  :  0);
	Dg.z = 1;
	// - run the Grid of Blocs of threads
	MatrixProductKernel_v4<<<Dg,Db>>>();
	break;
	
	case GK5 :
	break;
	
	default :
	fprintf(stderr,"Unknown GPU kernel!");
	exit(EXIT_FAILURE);
	}
}
	
	
