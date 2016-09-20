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
 int lig = threadIdx.x;
 int col = threadIdx.y;
 double res = 0.0;

 // Matrix product computation$
 int i;
 for(i=0;i<SIZE;i++)
 {
  res += GPU_A[i][lig]*GPU_B[col][i];
 }

 GPU_C[col][lig] = res;
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
   Db.x = SIZE;
   Db.y = 1;
   Db.z = 1;
   Dg.x = SIZE/BLOCK_SIZE_X_K0;
   Dg.y = 1;
   Dg.z = 1;
   // - run the Grid of Blocs of threads
   MatrixProductKernel_v0<<<Dg,Db>>>();
   break;

 case GK1 :
  break;

 case GK2 :
  break;
  
 case GK3 :
  break;

 case GK4 :
  break;
  
 case GK5 :
  break;

 default :
   fprintf(stderr,"Unknown GPU kernel!");
   exit(EXIT_FAILURE);
 }
}
