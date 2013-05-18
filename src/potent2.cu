#include "potent2.h"
#include "dev_mem.h"
#include "dev_mem_reuse.h"
#include "device_utils.h"
#include "global_variables.h"
#include "particle_allocator.h"
#include "pic_utils.h"

#include <cufft.h>

#include <stdio.h>
#include <string>

//******************************************************************************
// Name: checkCufftStatus
// Code Type: Kernel
// Purpose: Check a cufft call for errors. If one is found, in all cases print
//          an error message. If in debug mode, throw an exception, otherwise
//          exit.
// Parameters:
// ----------------
// returnCode - The return code set by a cufft routine
//******************************************************************************
void checkCufftStatus(cufftResult returnCode)
{
   std::string errorString;
   FILE *file;

   switch(returnCode)
   {
   case CUFFT_INVALID_PLAN:
      errorString = "ERROR: CUFFT Invalid Plan";
      break;
   case CUFFT_ALLOC_FAILED:
      errorString = "ERROR: CUFFT Alloc Failed";
      break;
   case CUFFT_INVALID_TYPE:
      errorString = "ERROR: CUFFT Invalid Type";
      break;
   case CUFFT_INVALID_VALUE:
      errorString = "ERROR: CUFFT Invalid Value";
      break;
   case CUFFT_INTERNAL_ERROR:
      errorString = "ERROR: CUFFT Internal Error";
      break;
   case CUFFT_EXEC_FAILED:
      errorString = "ERROR: CUFFT Exec Failed";
      break;
   case CUFFT_SETUP_FAILED:
      errorString = "ERROR: CUFFT Setup Failed";
      break;
   case CUFFT_INVALID_SIZE:
      errorString = "ERROR: CUFFT Invalid Size";
      break;
   };

   if(returnCode != CUFFT_SUCCESS)
   {
      fprintf(stderr, "%s\n", errorString.c_str());
      file = fopen("errorLog.txt", "w");
      fprintf(file,"%s\n", errorString.c_str());
      fclose(file);
      assert(returnCode == CUFFT_SUCCESS);
      exit(1);
   }
}

//******************************************************************************
// Name: initPb
// Code Type: Kernel
// Block Structure: One thread per grid column; Blocks and block size should
//                  be one dimensional
// Shared Memory Requirements: None
// Purpose: Initializes the pb array with voltage P0 and the imaginary
//          components to 0
// Parameters:
// ----------------
// pb - The array of size NX1 to initialize
// P0 - TBD
// size - The size of the array to convert
//******************************************************************************
__global__
void initPb(cufftComplex pb[], const float P0, const unsigned int size)
{
   unsigned int threadX = blockDim.x * blockIdx.x + threadIdx.x;
   cufftComplex val;
   val.x = P0;
   val.y = 0.0;

   if(threadX < size)
   {
      pb[threadX] = val;
   }
}

//******************************************************************************
// Name: loadHarmonics
// Code Type: Kernel
// Block Structure: One thread per grid column
// Shared Memory Requirements: None
// Purpose: Calculates the values for cokx
// Parameters:
// ----------------
// cokx - TBD
// size - Should be set to NX, which is 1 more than the number of elements
//******************************************************************************
__global__
void loadHarmonics(float cokx[], const unsigned int size)
{
   unsigned int threadX = blockDim.x * blockIdx.x + threadIdx.x;
   float num;

   if(threadX < size-1)
   {
      num = sin(D_PI * threadX / size);
      cokx[threadX]=4*(num*num);
   }
}

//******************************************************************************
// Name: calcPhif
// Code Type: Kernel
// Block Structure: One thread per grid column
// Shared Memory Requirements: None
// Purpose: Calculates the values for phif
// Parameters:
// ----------------
// phif - The phif array which will be set
// z - The z array is an intermediate working array and is expected to be of
//     size NX1 * NY1
// yyy - The yyy array is an intermediate working array and is expected to be of
//       size NX1 * NY1
// cokx - TBD size NX1
// pb - TBD size NX1
// c  - TBD size NX1 * NY1
// NX1 - The number of columns being calculated
// NY1 - The number of rows being calculated
// DX - The spacing between columns
// DY - The spacing between rows
//******************************************************************************
__global__
void calcPhif(cufftComplex phif[],
              float z [],
              cufftComplex yyy[],
              const float cokx[],
              const cufftComplex pb[],
              const cufftComplex c[],
              const unsigned int NX1,
              const unsigned int NY1,
              const float DX,
              const float DY
              )
{
   const unsigned int threadX = blockDim.x * blockIdx.x + threadIdx.x;
   float2 f2zero;
   f2zero.x = 0;
   f2zero.y = 0;
   float2 tempYyy;
   float2 tempF2;
   float2 tempC;
   float tempZ;
   float tempCokx = cokx[threadX];
   const float dySquared = DY * DY;
   int index;
   int oneBack;
   int oneForward;

   if(threadX >= NX1)
   {
      return;
   }

   z[NX1 * (NY1-1) + threadX] = 0;
   yyy[NX1 * (NY1-1) + threadX] = pb[threadX];
   phif[threadX] = f2zero;

   for(int j = NY1 - 1; j  >= 1; j--)
   {
      index = NX1 * j + threadX;
      oneBack = index - NX1;
      tempZ = (float)1./((float)2. + (tempCokx*dySquared)-z[index]);
      z[oneBack] = tempZ;
      tempYyy = yyy[index];
      tempC = c[index];
      tempF2.x = tempZ * (tempYyy.x + dySquared * tempC.x);
      tempF2.y = tempZ * (tempYyy.y + dySquared * tempC.y);
      yyy[oneBack] = tempF2;
   }

   for(int j = 0; j < NY1; j++)
   {
      index = NX1 * j + threadX;
      oneForward = index + NX1;
      tempF2 = phif[index];
      tempZ = z[index];
      tempF2.x *= tempZ;
      tempF2.y *= tempZ;
      tempYyy = yyy[index];
      tempF2.x += tempYyy.x;
      tempF2.y += tempYyy.y;
      phif[oneForward] = tempF2;
   }
}

//******************************************************************************
// Name: complexToReal
// Code Type: Kernel
// Block Structure: One thread per value
// Shared Memory Requirements: blockDim.x * sizeof(float2)
// Purpose: Copies a cufftComplex array into a real array. All complex values
//          are thrown away
// Parameters:
// ----------------
// complex - The cufftComplex array
// real - The array of reals to be set
// size - The size of the array to convert
//******************************************************************************
__global__
void complexToReal(const cufftComplex complex[], float real[], 
                   const unsigned int size)
{
   int threadX = blockDim.x * blockIdx.x + threadIdx.x;
   extern __shared__ float2 temp[];

   if(threadX < size)
   {
      temp[threadIdx.x] = complex[threadX];
      real[threadX] = temp[threadIdx.x].x;
   }
}

//******************************************************************************
// Name: realToComplex
// Code Type: Kernel
// Block Structure: One thread per value
// Shared Memory Requirements: None
// Purpose: Copies a real array into a cufftComplex array. All complex values
//          are set to 0
// Parameters:
// ----------------
// real - The array of reals
// complex - The cufftComplex array to be set
// size - The size of the array to convert
//******************************************************************************
__global__
void realToComplex(const float real[], cufftComplex complex[], 
                   const unsigned int size)
{
   int threadX = blockDim.x * blockIdx.x + threadIdx.x;
   float2 temp;

   if(threadX < size)
   {
      temp.x = real[threadX];
      temp.y = 0;
      complex[threadX] = temp;
   }
}

//******************************************************************************
// Name: mapToPhi
// Code Type: Kernel
// Block Structure: Block size is 1 dimensional, Num blocks should be cover from
//                  0-NY vertically and 0-NX1 horizontally
// Shared Memory Requirements: None
// Purpose: Takes dens' internal memory layout for phi and maps it back to the
//          layout that field will expect. This process is grossly ineffecient
// Parameters:
// ----------------
// phi - The electric potential that will be passed to field
// packedFormat - The phi generated by dens
// packedWidth - The width of the packedFormat array
// packedHeight - The height of the packedFormat array
// phiWidth - The width of the phi array
//******************************************************************************
__global__
void mapToPhi(float phi[], const float packedFormat[],
              unsigned int packedWidth, unsigned int packedHeight,
              unsigned int phiWidth)
{
   int threadX = blockDim.x * blockIdx.x + threadIdx.x;
   int threadY = blockDim.y * blockIdx.y + threadIdx.y;

   if(threadX >= packedWidth ||
      threadY >= packedHeight)
   {
      return;
   }

   phi[threadX * phiWidth + threadY] = 
      packedFormat[threadY * packedWidth + threadX];
}

//******************************************************************************
// Name: fixPhiSides
// Code Type: Kernel
// Block Structure: A single dimension of NY threads needed
// Shared Memory Requirements: None
// Purpose: Sets the periodic boundary condition for phi along the sides
// Parameters:
// ----------------
// phi - The electric potential
// width - The width of the phi array
// height - The height of the phi array
//******************************************************************************
__global__
void fixPhiSides(float phi[], 
                 unsigned int width,
                 unsigned int height
                 )
{
   int y = blockDim.x * blockIdx.x + threadIdx.x;
   
   phi[width * height + y] = phi[y];
}

//******************************************************************************
// Name: potent2
// Purpose: Calculate the electric potential at all of the grid points and
//          store it in phi.
// Input Parameters:
// -------------------
// dev_rho - The magnetic field at the grid points
//
// Output Parameters:
// -------------------
// dev_phi - The electrical potential at all of the grid points
//******************************************************************************
void potent2(DevMemF &dev_phi, const DevMemF &dev_rho)
{
   static bool first = true;

   unsigned int numThreads;
   dim3 blockSize;
   dim3 numBlocks;
   int sharedSize;
   DevMem<cufftComplex, ParticleAllocator> dev_c(NX1 * NY);
   DevMem<cufftComplex, ParticleAllocator> dev_pb(NX1);
   DevMem<float, DevMemReuse> dev_cokx(NX1);
   DevMem<cufftComplex, ParticleAllocator> dev_phif(NY * NX1);
   DevMem<float, DevMemReuse> dev_z(NY1 * NX1);
   DevMem<cufftComplex, ParticleAllocator> dev_yyy(NY1 * NX1);

   resizeDim3(blockSize, MAX_THREADS_PER_BLOCK / 2);
   resizeDim3(numBlocks, calcNumBlocks(256, NX1 * NY));
   cudaThreadSynchronize();
   checkForCudaError("Beginning of potent2");
   realToComplex<<<numBlocks, blockSize>>>(dev_rho.getPtr(), dev_c.getPtr(),
      dev_rho.size());
   cudaThreadSynchronize();
   checkForCudaError("realToComplex");
   static cufftHandle rhoTransform;
   if(first)
   {
      checkCufftStatus(cufftPlan1d(&rhoTransform, NX1, CUFFT_C2C, NY));
   }
   checkCufftStatus(cufftExecC2C(rhoTransform, dev_c.getPtr(), 
      dev_c.getPtr(), CUFFT_FORWARD));
   //cufftDestroy(rhoTransform);

   //ccccccccccccccccccccccccccccccccccccccccccccccccccccccc
   //           the poisson equation begins
   //ccccccccccccccccccccccccccccccccccccccccccccccccccccccc
   //     boundary conditions
   //ccccccccccccccccccccccccccccccccccccccccccccccccccccccc

   numThreads = MAX_THREADS_PER_BLOCK / 4;
   resizeDim3(blockSize, numThreads);
   resizeDim3(numBlocks, calcNumBlocks(numThreads, NX1));
   initPb<<<numBlocks, blockSize>>>(dev_pb.getPtr(), P0, dev_pb.size());
   checkForCudaError("initPb");

   static cufftHandle pbTransform;
   if(first)
   {
      checkCufftStatus(cufftPlan1d(&pbTransform, NX1, CUFFT_C2C, 1));
   }
   cudaThreadSynchronize();
   checkForCudaError("Before cufft on dev_pb");
   checkCufftStatus(cufftExecC2C(pbTransform, dev_pb.getPtr(), dev_pb.getPtr(),
      CUFFT_FORWARD));
   //cufftDestroy(pbTransform);

   // loading harmonics
   numThreads = MAX_THREADS_PER_BLOCK / 4;
   resizeDim3(blockSize, numThreads);
   resizeDim3(numBlocks, calcNumBlocks(numThreads, NX1));
   loadHarmonics<<<numBlocks, blockSize>>>(
      dev_cokx.getPtr(), dev_cokx.size()+1);
   checkForCudaError("loadHarmonics");

   numThreads = MAX_THREADS_PER_BLOCK / 8;
   resizeDim3(blockSize, numThreads);
   resizeDim3(numBlocks, calcNumBlocks(numThreads, NX1));
   cudaThreadSynchronize();
   checkForCudaError("Before calcPhif");
   calcPhif<<<numBlocks, numThreads>>>(dev_phif.getPtr(),
      dev_z.getPtr(), dev_yyy.getPtr(), dev_cokx.getPtr(), dev_pb.getPtr(),
      dev_c.getPtr(), NX1, NY1, DX, DY);
   checkForCudaError("calcPhif");

   static cufftHandle phifTransform;
   if(first)
   {
      checkCufftStatus(cufftPlan1d(&phifTransform, NX1, CUFFT_C2C, NY));
   }
   cudaThreadSynchronize();
   checkForCudaError("Before inverse cufft on phif");
   cufftExecC2C(phifTransform, dev_phif.getPtr(), 
      dev_phif.getPtr(), CUFFT_INVERSE);
   //cufftDestroy(phifTransform);

   // Make space for transpose
   dev_yyy.freeMem();

   //DevMemF dev_tempPhi(NY * NX1);
   numThreads = MAX_THREADS_PER_BLOCK / 2;
   sharedSize = numThreads * sizeof(float2);
   resizeDim3(blockSize, numThreads);
   resizeDim3(numBlocks, calcNumBlocks(numThreads, NX1 * NY));
   cudaThreadSynchronize();
   checkForCudaError("Before final complex to real call in potent2");
   complexToReal<<<numBlocks, numThreads, sharedSize>>>(
      dev_phif.getPtr(), dev_phi.getPtr(), dev_phif.size());
   checkForCudaError("potent2::complexToReal");

   cudaThreadSynchronize();
   checkForCudaError("Before potent2 divVector");
   // Normalize the inverse transform
   divVector(dev_phi, float(NX1));
   checkForCudaError("potent2::divVector");

   first = true;
}
