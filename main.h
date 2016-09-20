/*********************************************************************************/
/* Matrix product program for a multi-core CPU and for a many-core GPU           */
/* S. Vialle - November 2015                                                     */
/*********************************************************************************/

#ifndef __MATPROD_MAIN__
#define __MATPROD_MAIN__


/*-------------------------------------------------------------------------------*/
/* CONSTANTS.                                                                    */
/*-------------------------------------------------------------------------------*/

// Matrix size (side of the 3 matrixes)
//#define SIZE                1029          // To debug
//#define SIZE              1025        // To debug
#define SIZE              4096        // To benchmark
//#define SIZE              4097        // To debug


// Constants for run configurations 
#define DEFAULT_NB_THREADS  1          // Constant for OpenMP configuration 
#define DEFAULT_ONGPUFLAG   1          // Constant for computation mode configuration
#define DEFAULT_CPUKID      CK0        // Constant for CPU Kernel config
#define DEFAULT_GPUKID      GK0        // Constant for GPU Kernel config

// Block sizes
#define BLOCK_SIZE_X_K0     16
#define BLOCK_SIZE_Y_K0     64


/*-------------------------------------------------------------------------------*/
/* Enumerated type of the different kernels                                      */
/*-------------------------------------------------------------------------------*/
typedef enum _ckid_t {
   CK0 = 0, 
   CK1,
   NB_OF_CPU_KERNELS
} ckid_t;


typedef enum _gkid_t {
   GK0 = 0, 
   GK1,
   GK2,
   GK3,
   GK4,
   GK5,
   NB_OF_GPU_KERNELS
} gkid_t;


/*-------------------------------------------------------------------------------*/
/* Global variable declarations.                                                 */
/*-------------------------------------------------------------------------------*/

/* Matrixes: C = A.B                                                             */
/* We use the Transposed B matrix, in place of B, to improve cache memory usage. */
extern double A[SIZE][SIZE];               /* Matrixes : C = A.B           */
extern double B[SIZE][SIZE];               /* B Matrix.                    */
extern double TB[SIZE][SIZE];
extern double C[SIZE][SIZE];

/* Global variables to control OpenMP computations.                              */
extern int NbThreads;

/* Global vars to control computation on the GPU.                                */
extern int OnGPUFlag;
extern ckid_t CPUKernelId;
extern gkid_t GPUKernelId;


/*-------------------------------------------------------------------------------*/
/* Global functions.                                                             */
/*-------------------------------------------------------------------------------*/
void Computation(void);
void cpuProduct(ckid_t kid);
int main(int argc, char *argv[]);


#endif

// END
