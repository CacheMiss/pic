#include "array2d.h"
#include "dev_mem.h"
#include "dev_mem_reuse.h"
#include "device_utils.h"
#include "global_variables.h"
#include "particle_allocator.h"
#include "pic_utils.h"
#include "typedefs.h"

#include <cuda.h>
#include <driver_functions.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#ifndef NO_THRUST
#include <thrust/binary_search.h>
#include <thrust/device_vector.h>
#endif

#ifdef DEVEMU
#include <vector>
#endif


struct ParticleBoundaries
{
   unsigned int beg;
   unsigned int end;

   ParticleBoundaries()
   {}

   ParticleBoundaries(unsigned int b, unsigned int e)
      :beg(b), end(e)
   { }
};


//******************************************************************************
// Function: findQuadIndices
// Code Type: Device
// Block Structure: 1 thread per element in rho to be considered
//                  (Normally NY x NX1)
// Purpose: Calculate the indices in global memory to access.  These are used 
//          to pull the proper values from bucketBeg and bucketEnd
// Input Parameters:
// -------------------
// threadX - The global x index of this thread
// threadY - The global y index of this thread
// bucketWidth - The number of buckets wide rho is
// left - True if the grid point has no elements to its left
// bottom - True if the grid point has no elements beneath it
//
// Output Parameters:
// -------------------
// gQuad1 - The index in the global beg and end array associated with this
//          threads quadrant 1
// gQuad2 - The index in the global beg and end array associated with this
//          threads quadrant 2
// gQuad3 - The index in the global beg and end array associated with this
//          threads quadrant 3
// gQuad4 - The index in the global beg and end array associated with this
//          threads quadrant 4
// quad1 - The index in the local beg and end array associated with this
//          threads quadrant 1
// quad2 - The index in the local beg and end array associated with this
//          threads quadrant 2
// quad3 - The index in the local beg and end array associated with this
//          threads quadrant 3
// quad4 - The index in the local beg and end array associated with this
//          threads quadrant 4
//******************************************************************************
__device__
void findQuadIndices(const unsigned int threadX, 
                     const unsigned int threadY,
                     const unsigned int bucketWidth,
                     const bool left, const bool bottom,
                     unsigned int &gQuad1, unsigned int &gQuad2,
                     unsigned int &gQuad3, unsigned int &gQuad4,
                     unsigned int &quad1, unsigned int &quad2,
                     unsigned int &quad3, unsigned int &quad4)
{
   // Calculate the indices in global memory to access
   // These are used to pull the proper values from bucketBeg and
   // bucketEnd
   gQuad1 = bucketWidth * threadY + threadX;
   if(!left)
   {
      gQuad2 = gQuad1 - 1;
   }
   // If there is nothing to the left, respect the periodic boundary
   // conditions and wrap to the other side of the grid
   else
   {
      gQuad2 = gQuad1 + bucketWidth-1;
   }
   gQuad3 = gQuad2 - bucketWidth;
   gQuad4 = gQuad1 - bucketWidth;

   // The index into the shared memory bucket beg and end variables
   // The + 1 values in this calculation exist because a border of
   // one bucket is needed around the left and bottom
   quad1 = (blockDim.x + 1) * (threadIdx.y + 1) + 
      threadIdx.x + 1;
   quad2 = quad1 - 1;
   quad3 = quad2 - (blockDim.x + 1);
   quad4 = quad1 - (blockDim.x + 1);
}

//******************************************************************************
// Function: densGridPointsLoadShared
// Code Type: Device
// Block Structure: 1 thread per element in rho to be considered
//                  (Normally NY x NX1)
// Purpose: Load the beginning and end particles for each grid square into
//          shared memory
//
// Input Parameters:
// -------------------
// bucketBeg - The global memory array storing the starting index in the array
//             created by loadParticles associated with this grid square
// bucketEnd - The global memory array storing the starting index in the array
//             created by loadParticles associated with this grid square
// gQuad1 - The index in the global beg and end array associated with this
//          threads quadrant 1
// gQuad2 - The index in the global beg and end array associated with this
//          threads quadrant 2
// gQuad3 - The index in the global beg and end array associated with this
//          threads quadrant 3
// gQuad4 - The index in the global beg and end array associated with this
//          threads quadrant 4
// quad1 - The index in the local beg and end array associated with this
//          threads quadrant 1
// quad2 - The index in the local beg and end array associated with this
//          threads quadrant 2
// quad3 - The index in the local beg and end array associated with this
//          threads quadrant 3
// quad4 - The index in the local beg and end array associated with this
//          threads quadrant 4
//
// Output Parameters:
// -------------------
// sharedBeg - The shared memory array storing the starting index in the array
//             created by loadParticles associated with this grid square
// sharedEnd - The shared memory array storing the starting index in the array
//             created by loadParticles associated with this grid square
//******************************************************************************
__device__
void densGridPointsLoadShared(const float* __restrict area1, 
                              const float* __restrict area2,
                              const float* __restrict area3, 
                              const float* __restrict area4,
                              float* __restrict a1, 
                              float* __restrict a2,
                              float* __restrict a3, 
                              float* __restrict a4,
                              const unsigned int quad1,
                              const unsigned int quad2,
                              const unsigned int quad3,
                              const unsigned int quad4,
                              const unsigned int gQuad1,
                              const unsigned int gQuad2,
                              const unsigned int gQuad3,
                              const unsigned int gQuad4,
                              const bool left,
                              const bool bottom
                              )
{
   //////////////////////////////////////////////////////////////////////////
   // NOTE: Commented out assignments of a1-a4 are left for completeness.
   //       They are not however, necessary because of the geometry involved
   //////////////////////////////////////////////////////////////////////////

   // Values associated with quadrants around a grid point
   // Q1 = a4
   // Q2 = a3
   // Q3 = a1
   // Q4 = a2
   
   // Load shared memory
   //a1[quad1] = area1[gQuad1];
   //a2[quad1] = area2[gQuad1];
   a3[quad1] = area3[gQuad1];
   a4[quad1] = area4[gQuad1];
   
   // Pull the extra bottom row into shared memory
   if(!bottom && threadIdx.y == 0)
   {
      a1[quad4] = area1[gQuad4];
      a2[quad4] = area2[gQuad4];
      //a3[quad4] = area3[gQuad4];
      //a4[quad4] = area4[gQuad4];
   }
   // Pull the extra left row into shared memory
   if(threadIdx.x == 0)
   {
      //a1[quad2] = area1[gQuad2];
      //a2[quad2] = area2[gQuad2];
      a3[quad2] = area3[gQuad2];
      //a4[quad2] = area4[gQuad2];
   }
   // Get the bottom left corner
   if(!bottom && threadIdx.x == 0 && threadIdx.y == 0)
   {
      a1[quad3] = area1[gQuad3];
      //a2[quad3] = area2[gQuad3];
      //a3[quad3] = area3[gQuad3];
      //a4[quad3] = area4[gQuad3];
   }
}

//******************************************************************************
// Function: densGridPoints
// Code Type: Kernel
// Block Structure: 1 thread per element in rho to be considered
//                  (Normally NX x NY)
// Shared Memory Requirements: 
//     2 * (blockDim.x + 1) * (blockDim.y + 1) * sizeof(unsigned int) +
//     4 * particlesToBuffer * sizeof(float) * blockDim.x
// Purpose: Find the charge density at grid points the results for each
//          point are then added to the current contents of rho.
// Input Parameters:
// -------------------
// bucketWidth - The number of buckets wide the rho is
// bucketHeight - The number of buckets high rho is
// bucketBeg[] - Lists the starting element in bucktToParticleMap for every
//             grid point
// bucketEnd[] - Lists the last (exclusive) element in bucktToParticleMap for 
//             every grid point
// area[] - The array storing a1, a2, a3, and a4 for all the particles
// cold - False if the particle array is composed of cold particles;
//        True otherwise
// particlesToBuffer - The number of particles per thread to buffer into shared
//        memory.
// NIJ - The avg particles per cell
//
// Output Parameters:
// -------------------
// rho[] - The charge array for the grid points
//******************************************************************************
__global__
void densGridPoints(float* __restrict__ rho,
                    const unsigned int bucketWidth,
                    const unsigned int bucketHeight,
                    const float* __restrict__ area1,
                    const float* __restrict__ area2,
                    const float* __restrict__ area3,
                    const float* __restrict__ area4,
                    const bool cold,
                    const unsigned int particlesToBuffer,
                    const unsigned int NIJ
                    )
{
   extern __shared__ char sharedBase[];
   // Calculate all of the shared memory offsets
   float *a1 = reinterpret_cast<float*>(sharedBase);
   float *a2 = &a1[2 * (blockDim.x + 1)];
   float *a3 = &a2[2 * (blockDim.x + 1)];
   float *a4 = &a3[2 * (blockDim.x + 1)];
   // End of all the shared memory offset calculation

   unsigned int threadX = blockDim.x * blockIdx.x + threadIdx.x;
   unsigned int threadY = blockDim.y * blockIdx.y + threadIdx.y;

   float rhoLocal = 0;

   // Return if there is no work
   bool hasWork = true;
   if(threadY >= bucketHeight ||
      threadX >= bucketWidth)
   {
      hasWork = false;
   }

   // The global bucket id number associated with this thread
   unsigned int gQuad1;
   unsigned int gQuad2;
   unsigned int gQuad3;
   unsigned int gQuad4;
   unsigned int quad1;
   unsigned int quad2;
   unsigned int quad3;
   unsigned int quad4;
   bool bottom = false;
   bool left = false;

   if(blockIdx.y == 0 && threadIdx.y == 0)
   {
      bottom = true;
   }

   if(blockIdx.x == 0 && threadIdx.x == 0)
   {
      left = true;
   }

   // Initialize all the values for the quadrants associated with 
   // this grid point
   findQuadIndices(threadX, threadY, bucketWidth,
      left, bottom,
      gQuad1, gQuad2, gQuad3, gQuad4,
      quad1, quad2, quad3, quad4);

   if(hasWork)
   {
      // Load bucketBeg and bucketEnd into shared memory
      densGridPointsLoadShared(area1, area2, area3, area4,
         a1, a2, a3, a4,
         quad1, quad2, quad3, quad4,
         gQuad1, gQuad2, gQuad3, gQuad4,
         left, bottom);
   }

   // Make sure all threads have finished their load into shared memory
   __syncthreads();

   if(hasWork)
   {
      // Values associated with quadrants around a grid point
      // Q1 = a4
      // Q2 = a3
      // Q3 = a1
      // Q4 = a2

      // Quadrant 1
      //    | *
      // ---|---
      //    |
      rhoLocal += a4[quad1];

      // Quadrant 2
      //  * |
      // ---|---
      //    |
      rhoLocal += a3[quad2];

      if(!bottom)
      {
         // Quadrant 3
         //    |
         // ---|---
         //  * |
         rhoLocal += a1[quad3];

         // Quadrant 4
         //    |
         // ---|---
         //    | *
         rhoLocal += a2[quad4];
      }
   }

   if(hasWork)
   {
      if(cold)
      {
         // Scale the charge of the cold particles
         // This allows us to use our memory to simulate more hot particles
         //rhoLocal = rhoLocal * 10 / NIJ;
         rhoLocal = rhoLocal / NIJ;
      }
      else
      {
         //rhoLocal = rhoLocal / NIJ;
         rhoLocal = rhoLocal / (NIJ * 10);
      }
      rho[bucketWidth * threadY + threadX] += rhoLocal;
   }
}

struct DbgArea
{
   unsigned int threadX;
   unsigned int localBeg;
   unsigned int localEnd;
   uint2 globalBeg;
   uint2 globalEnd;
};

//******************************************************************************
// Function: sumArea
// Code Type: Kernel
// Block Structure: 1 thread per grid point (Normally NX1 x NY grid points)
//                  Blocks cannot contain grid points that are above or below
//                  one another
// Shared Memory Requirements: 
//     4 * sizeof(float) * blockSize * particlesToBuffer + 2 * sizeof(uint2);
// Purpose: Four area values are calculated for each particle in the calcA 
//          kernel. Once these areas are known, sumArea is called to find the
//          the sum of the areas for all particles in each cell.
// Input Parameters:
// -------------------
// bucketBeg[] - An array containing the first particle associated with each
//               grid bucket
// bucketEnd[] - An array containing the last particle associated with each
//               grid bucket
// maxMinArray[] - A book keeping array of global memory which contains 
//                 numGridBins of elements
// a1[] - The a1 values for each particle
// a2[] - The a2 values for each particle
// a3[] - The a3 values for each particle
// a4[] - The a4 values for each particle
// numGridBins - The number of grid bins in the simulation area 
//               This is usually NX1 * NY
// numParticles - The number of particles being considered
// bufferSize - The number of particles to buffer within shared memory at
//              a time. Care should be taken to not select a value that would
//              require more shared memory than is present on the device.
//
// Output Parameters:
// -------------------
// a1Sum[] - The array containing the sums for a1
// a2Sum[] - The array containing the sums for a2
// a3Sum[] - The array containing the sums for a3
// a4Sum[] - The array containing the sums for a4
//******************************************************************************
__global__
void sumArea(float* __restrict__ a1Sum, float* __restrict__ a2Sum, 
             float* __restrict__ a3Sum, float* __restrict__ a4Sum, 
             const unsigned int* __restrict__ bucketBeg, 
				 const unsigned int* __restrict__ bucketEnd,
             uint2* __restrict__ maxMinArray,
             const float* __restrict__ a1, const float* __restrict__ a2, 
             const float* __restrict__ a3, const float* __restrict__ a4,
             unsigned int numGridBins, unsigned int numParticles,
             int bufferSize=4)
{
   // Begin shared memory declarations
   extern __shared__ float begShared[];
   float *partBufA1 = begShared;
   float *partBufA2 = &partBufA1[blockDim.x * bufferSize];
   float *partBufA3 = &partBufA2[blockDim.x * bufferSize];
   float *partBufA4 = &partBufA3[blockDim.x * bufferSize];
   uint2 *blockBeg = 
      reinterpret_cast<uint2*>(&partBufA4[blockDim.x * bufferSize]);
   uint2 *blockEnd = blockBeg + 1;
   // End shared memory declarations

   uint2 *globalMaxMin = &maxMinArray[blockIdx.x];

   unsigned int threadX = blockDim.x * blockIdx.x + threadIdx.x;
   unsigned int beg;
   unsigned int end;

   // For gBeg and gEnd x is the first particle associated with a cell
   // and y is the last particle associated with the the cell (exclusive)

   // The first particle range necessary for this block
   uint2 gBeg;
   // The second particle range necessary for this block
   uint2 gEnd;

   unsigned int tmp1;
   unsigned int tmp2;

   // Local copies for area sums
   float localA1 = 0;
   float localA2 = 0;
   float localA3 = 0;
   float localA4 = 0;

   // Boolean flag marked true if the cell this thread is associated
   // with has particles in it
   bool hasWork = true;

   gBeg.x = 0;
   gBeg.y = 0;
   gEnd = gBeg;

   if(threadX < numGridBins)
   {
      beg = bucketBeg[threadX];
      end = bucketEnd[threadX];
   }
   else
   {
      hasWork = false;
      beg = 0;
      end = 0;
   }

   // Check if this set of areas has work
   if(beg == end)
   {
      hasWork = false;
   }

   // Store the range of particles needed by the first thread in the block
   if(threadIdx.x == 0)
   {
      blockBeg->x = beg;
      blockBeg->y = end;
   }
   // Store the range of particles needed by the last thread in the block
   else if(threadIdx.x == blockDim.x - 1)
   {
      blockEnd->x = beg;
      blockEnd->y = end;
   }

   __syncthreads();

   tmp1 = blockBeg->x;
   tmp2 = blockBeg->y;

   // If the first thread in the block has particles, create a copy for each
   // thread to limit memory contention
   if(tmp1 != tmp2)
   {
      gBeg.x = tmp1;
      gBeg.y = tmp2;
   }

   tmp1 = blockEnd->x;
   tmp2 = blockEnd->y;

   // If the last thread in the block has particles, create a copy for each
   // thread to limit memory contention
   if(tmp1 != tmp2)
   {
      gEnd.x = tmp1;
      gEnd.y = tmp2;
   }

   // If I am unsure if I have an particles in this block, I have to use atomic
   // functions to find out
   if(gBeg.x == gBeg.y || gEnd.x == gEnd.y)
   {
      if(beg != end)
      {
         atomicMin(&globalMaxMin->x, beg);
         atomicMax(&globalMaxMin->y, end);
      }
      __syncthreads();
      // Load the global value into a register to prevent serialized access
      if(threadIdx.x == 0)
      {
         *blockBeg = *globalMaxMin;
      }
      __syncthreads();
      //gBeg.x = blockBeg->x;
      //gEnd.y = blockBeg->y;
      gBeg = *blockBeg;
      gEnd = *blockBeg;
   }

   // Align gBeg and gEnd to allow for coalescing memory operations
   gBeg.x = (gBeg.x / 16) * 16;
   gEnd.y = ((gEnd.y + 15) / 16) * 16;

   // The starting index in the global array to load from for this block
   unsigned int gStartIndex = gBeg.x;
   // The global index this thread will load
   unsigned int gThreadIndex = gStartIndex + threadIdx.x;
   // The shared memory index this thread will load to
   unsigned int localIndex = threadIdx.x;

   unsigned int loopInit;
   unsigned int loopEnd;
   unsigned int lastLoadedIndex;

   // gStartIndex = the beginning of the current section to load
   // gEnd.y = one past the last particle to load
   while(gStartIndex < gEnd.y)
   {
      // Fill the buffer with particles
      localIndex = threadIdx.x;
      for(int i = 0; i < bufferSize; i++)
      {
         // Make sure there really is a particle to load, then load it
         if(gThreadIndex < gEnd.y)
         {
            partBufA1[localIndex] = a1[gThreadIndex];
            partBufA2[localIndex] = a2[gThreadIndex];
            partBufA3[localIndex] = a3[gThreadIndex];
            partBufA4[localIndex] = a4[gThreadIndex];
            localIndex += blockDim.x;
            gThreadIndex += blockDim.x;
         }
      }

      __syncthreads();

      // If buffer isn't full, mark the end as the last valid particle
      if(gStartIndex + blockDim.x >= gEnd.y)
      {
         lastLoadedIndex = gEnd.y;
      }
      // Buffer is full, last particle is end of buffer
      else
      {
         lastLoadedIndex = gStartIndex + blockDim.x * bufferSize;
      }

      if(gStartIndex < end && lastLoadedIndex > beg && hasWork)
      {
         loopEnd = min(end, lastLoadedIndex) - gStartIndex;
         loopInit = (gStartIndex >= beg) ? 0 : beg - gStartIndex;
         for(int i = loopInit; i < loopEnd; i++)
         {
            localA1 += partBufA1[i];
            localA2 += partBufA2[i];
            localA3 += partBufA3[i];
            localA4 += partBufA4[i];
         }
      }

      gStartIndex += blockDim.x * bufferSize;
      __syncthreads();
   }

   if(hasWork)
   {
      a1Sum[threadX] = localA1;
      a2Sum[threadX] = localA2;
      a3Sum[threadX] = localA3;
      a4Sum[threadX] = localA4;
   }
}

//******************************************************************************
// Name: loadParticleLocations
// Code Type: Kernel
// Block Structure: 1 thread per particle
// Purpose: Values and bins are used in calcA, and densGridPoints.
//          This routine loads those arrays with values from the particle
//          array.
// Input Parameters:
// ----------------
// particles - The particles that will be sorted
// size - The number of particles to load
// NX  - The width of the grid + 1
// NX1 - The width of the grid
// NY -  The height of the grid
// coldEle - If the particles are cold electrons, this is set to true
//
// Output Parameters:
// ----------------
// locations - The array storing the x positions of the particles that will be sorted
// associatedBin - The bin each particle belongs in
//******************************************************************************
__global__
void loadParticleLocations(const float2* __restrict__ d_loc, 
                           float2* __restrict__ locCopy, 
                           unsigned int* __restrict__ associatedBin,
                           const unsigned int size,
                           const unsigned int NX,
                           const unsigned int NX1,
                           const unsigned int NY,
                           bool coldEle)
{
   int particleIndex = blockDim.x * blockIdx.x + threadIdx.x;
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   unsigned int row;
   unsigned int column;
   float2 loc;
   if(index < size)
   {
      loc = d_loc[particleIndex];
      locCopy[index] = loc;
      // for 2^126 <= y <= 2^128, __fdividef(x,y) delivers a result of zero,
      column = 
         static_cast<unsigned int>(__fdividef(loc.x, D_DX));
      // for 2^126 <= y <= 2^128, __fdividef(x,y) delivers a result of zero,
      row = 
         static_cast<unsigned int>(__fdividef(loc.y, D_DY));
      if(coldEle && row == NY)
      {
         row = 0;
      }
      if(column == NX1)
      {
         column = 0;
      }
      associatedBin[index] = row * NX1 + column;
   }
}

//******************************************************************************
// Name: calcA
// Code Type: Kernel
// Block Structure: 1 thread per particle
// Shared Memory Requirements: threadDim.x * sizeof(float2)
// Purpose: Calculates a1/tota, a2/tota, a3/tota, and a4/tota for every particle 
//          listed in particleLocations. These values are used in densElectron 
//          and densIon
// Input Parameters:
// ----------------
// particleLocations - xy pairs of all the particles to be calculated
// NY - THe height of the grid
// DX - The horizontal grid spacing
// DY - The veritcal grid spacing
// numParticles - The number of particles
//
// Output Parameters:
// ----------------
// area - A float4 array to store the area values in; a1 = area.x, a2 = area.y
//        a3 = area.z, a4 = area.w
//******************************************************************************
__global__
void calcA(const float2* __restrict__ particleLocations,
           float* __restrict__ area1,
           float* __restrict__ area2,
           float* __restrict__ area3,
           float* __restrict__ area4,
           const unsigned int numParticles,
           const unsigned int NY,
           const float DX,
           const float DY,
           const bool coldElectrons)
{
   int index = blockDim.x * blockIdx.x + threadIdx.x;
   float2 location;

   if(index >= numParticles)
   {
      return;
   }

   location = particleLocations[index];
   int2 gridIndex;
   // for 2^126 <= y <= 2^128, __fdividef(x,y) delivers a result of zero,
   gridIndex.x = __fdividef(location.x, DX);
   // for 2^126 <= y <= 2^128, __fdividef(x,y) delivers a result of zero,
   gridIndex.y = __fdividef(location.y, DX);
   if (coldElectrons && gridIndex.y == NY)
   {
      gridIndex.y = 0; 
   }

   // Find the distance from the associated grid point that the particle is
   float dela = location.x - gridIndex.x * DX;
   float delb = location.y - gridIndex.y * DY;
   float a1;
   float a2;
   float a3;
   float a4;
   float tota;


   // Calculate the areas for the particle
   a1 = dela * delb;
   a2 = DX * delb-a1;
   a3 = DY * dela-a1;
   tota = DX * DY;
   a4 = tota - (a1 + a2 + a3);

   // Write the areas back to global memory
   // for 2^126 <= y <= 2^128, __fdividef(x,y) delivers a result of zero,
   area1[index] = __fdividef(a1, tota);
   area2[index] = __fdividef(a2, tota);
   area3[index] = __fdividef(a3, tota);
   area4[index] = __fdividef(a4, tota);
}

//******************************************************************************
// Name: fixGridSides
// Code Type: Kernel
// Block Structure: 1 thread y position in grid
// Shared Memory Requirements: None
// Purpose: Enforces the periodic boundary condition on Rho, essentially wrapping
//          the rho array so that it is circular
// Parameters:
// ----------------
// rhoe - The rho array for electrons
// rhoi - The rho array for ions
// logicalX - The width of the area that is calculated
// logicalY - The height of the area that is calculated
// physicalX - The number of columns allocated for the array
// physicalY - The number of rows allocated for the array
//******************************************************************************
__global__
void fixRhoGridSides(float* __restrict__ rhoe, float* __restrict__ rhoi,
                     const unsigned int logicalX, 
                     const unsigned int logicalY,
                     const unsigned int physicalX, 
                     const unsigned int physicalY
                     )
{
   float sum;
   int indexBottom = blockDim.x * blockIdx.x + threadIdx.x;
   int indexTop = indexBottom + physicalY * (logicalX - 1);

   if(indexBottom >= logicalY)
   {
      return;
   }

   sum = rhoe[indexBottom] + rhoe[indexTop];
   rhoe[indexBottom] = sum;
   rhoe[indexTop] = sum;
   sum = rhoi[indexBottom] + rhoi[indexTop];
   rhoi[indexBottom] = sum;
   rhoi[indexTop] = sum;
}

//******************************************************************************
// Name: fixRhoGridTopBottom
// Code Type: Kernel
// Block Structure: 1 thread x position in grid
// Shared Memory Requirements: None
// Purpose: Doubles the rho at the top and bottom of the grid
// Parameters:
// ----------------
// rhoe - The rho array for electrons
// rhoi - The rho array for ions
// width - The width of the area that is calculated
// height - The height of the area that is calculated
//******************************************************************************
__global__
void fixRhoGridTopBottom(float* __restrict__ rhoe, float* __restrict__ rhoi,
                         const unsigned int width, 
                         const unsigned int height
                         )
{
   unsigned int threadNum = blockDim.x * blockIdx.x + threadIdx.x;
   unsigned int distanceToTop = width * (height - 1);
   if(threadNum >= width)
   {
      return;
   }

   rhoe[threadNum] *= 2;
   rhoe[threadNum + distanceToTop] *= 2;
   rhoi[threadNum] *= 2;
   rhoi[threadNum + distanceToTop] *= 2;
}

//******************************************************************************
// Name: getFinalRho
// Code Type: Kernel
// Block Structure: 1 thread per grid point to be considered 
//                  Should be (X_GRD * Y_GRD threads)
// Shared Memory Requirements: None
// Purpose: Subtracts rhoe from rhoi
// Parameters:
// ----------------
// rhoe - The rho array for electrons
// rhoi - The rho array for ions
// size - The total number of grid points
//******************************************************************************
__global__ 
void getFinalRho(float* __restrict__ rho,
                 const float* __restrict__ rhoi,
                 const float* __restrict__ rhoe,
                 const unsigned int size)
{
   unsigned int index = blockDim.x * blockIdx.x + threadIdx.x;

   // Return if there is no work
   if(index >= size)
   {
      return;
   }

   rho[index] = rhoi[index] - rhoe[index];
}

__global__
void findUpperLowerBound(const unsigned int* __restrict__ binList,
                         unsigned int* __restrict__ beg, 
								 unsigned int* __restrict__ end,
                         unsigned int numParticles, 
								 unsigned int numBins)
{
   extern __shared__ char bytes[];
   unsigned int *scratch = reinterpret_cast<unsigned int *>(&bytes[0]);
   unsigned int threadX = blockDim.x * blockIdx.x + threadIdx.x;
   int index = threadIdx.x + 1;

   // If there is work to do
   if(threadX < numParticles)
   {
      // Grab the particle to the left of the block if there is one
      if(blockIdx.x != 0 && threadIdx.x == 0)
      {
         scratch[threadIdx.x] = binList[threadX - 1];
      }
      // Grab a particle
      scratch[index] = binList[threadX];
   }

   // Make sure all shared memory gets loaded
   __syncthreads();

   if(threadX < numParticles)
   {
      if((blockIdx.x > 0 || threadIdx.x != 0) &&
         scratch[index] > scratch[threadIdx.x])
      {
         beg[scratch[index]] = threadX;
         end[scratch[threadIdx.x]] = threadX;
      }

      if(threadX + 1 == numParticles)
      {
         end[scratch[index]] = threadX + 1;
      }
   }
}

template<class BinListAllocator, class BegEndAllocator>
void findBounds(const DevMem<unsigned int, BinListAllocator> &dev_binList,
                DevMem<unsigned int, BegEndAllocator> &dev_beg, 
                DevMem<unsigned int, BegEndAllocator> &dev_end,
                unsigned int numParticles, unsigned int numBins, 
                cudaStream_t stream
                )
{
   int  threadsInBlock = MAX_THREADS_PER_BLOCK / 2;
   dim3 blockSize(threadsInBlock);
   dim3 numBlocks(calcNumBlocks(threadsInBlock, numParticles));
   int  sharedMemoryBytes = sizeof(unsigned int) * (threadsInBlock + 1);

   findUpperLowerBound<<<numBlocks, blockSize, sharedMemoryBytes, stream>>>(
      dev_binList.getPtr(), dev_beg.getPtr(), dev_end.getPtr(), 
      numParticles, numBins);
   checkForCudaError("findUpperLowerBound");
}

//******************************************************************************
// Name: calcIntermediateRho
// Purpose: Calculates rho accross the grid for a particle array and adds it
//          to the contents in dev_rho
// Parameters:
// ----------------
// dev_rho - A pointer to device memory used to store the results in
// dev_particles - The [NM_PRTS, 5] particle array used to populate rho
// numParticles - The number of particles stored in dev_particles
// cold - True if the particles are cold
// electron - True of the particles are electrons
//******************************************************************************
void calcIntermediateRho(DevMemF &dev_rho,
                         const DevMem<float2> &d_partLoc, 
                         const unsigned int numParticles,
                         bool cold,
                         bool electron,
                         cudaStream_t stream)
{
   DeviceStats &dev(DeviceStats::getRef());
   DevMem<float2, ParticleAllocator> dev_particleLocations(numParticles);
   // A vector which maps grid buckets back to particle indices
   DevMem<unsigned int, ParticleAllocator> dev_gridBuckets(numParticles);
   DevMem<unsigned int, DevMemReuse> dev_bucketBegin(NY * NX1);
   DevMem<unsigned int, DevMemReuse> dev_bucketEnd(NY * NX1);
   dev_bucketBegin.zeroMem();
   dev_bucketEnd.zeroMem();
   dim3 *numBlocks;
   dim3 *blockSize;
   int threadsInBlock;
   int threadsX;
   int threadsY;
   int sharedMemoryBytes;
   unsigned int particlesToBuffer = 4;

   // Sort particles before calling densGridPoints

   // Run loadParticleLocations to generate copies of all particle
   // locations and calculate the bin associated with each of them.
   // This bin is then used to sort the particles
   threadsInBlock = dev.maxThreadsPerBlock / 2;
   blockSize = new dim3(threadsInBlock);
   numBlocks = new dim3(static_cast<unsigned int>(calcNumBlocks(threadsInBlock, numParticles)));
   cudaStreamSynchronize(stream);
   checkForCudaError("Before loadParticleLocations");
   loadParticleLocations<<<*numBlocks, *blockSize, 0, stream>>>(
      d_partLoc.getPtr(),
      dev_particleLocations.getPtr(),
      dev_gridBuckets.getPtr(),
      numParticles,
      NX, NX1, NY,
      cold && electron);
   delete numBlocks;
   delete blockSize;
   cudaStreamSynchronize(stream);
   checkForCudaError("densLoadSortArrays failed");

   // DEBUG
   //std::vector<int> h_keys(100);
   //std::vector<float2> h_val(100);
   //for(int i = 0; i < 100; i++)
   //{
   //   h_keys[i] = 100-i;
   //   h_val[i].x = i;
   //   h_val[i].y = 100-i;
   //   //h_val[i] = 100-i;
   //}
   //DevMem<int> dev_keys(100);
   //dev_keys = h_keys;
   //DevMem<float2> dev_val(100);
   //dev_val = h_val;
   //picSort(dev_keys, dev_val);
   //checkForCudaErrorSync("Sort test");
   //dev_keys.copyArrayToHost(&h_keys[0]);
   //dev_val.copyArrayToHost(&h_val[0]);
   // END DEBUG

   // Once all particles are loaded and have buckets; sort 
   // the buckets so that I can find all particles within a 
   // certain bucket
   assert(dev_gridBuckets.size() == dev_particleLocations.size());
   picSort(dev_gridBuckets, dev_particleLocations);
   checkForCudaError("Before findBounds");

   /*
   // Begin DEBUG
   cudaStreamSynchronize(stream);
   std::vector<unsigned int> h_gridBuckets(dev_gridBuckets.size());
   std::vector<float2> h_particleLocations(dev_particleLocations.size());
   dev_gridBuckets.copyArrayToHost(&h_gridBuckets[0]);
   dev_particleLocations.copyArrayToHost(&h_particleLocations[0]);
   FILE *dbgSort = fopen("dbgSort.txt", "w");
   for(int i = 0; i < h_gridBuckets.size(); i++)
   {
      fprintf(dbgSort, "%u %f %f\n", h_gridBuckets[i], h_particleLocations[i].x, h_particleLocations[i].y);
   }
   fclose(dbgSort);
   // End DEBUG
   */
   
   // find the beginning and end of each bucket's list of points
   findBounds(dev_gridBuckets, 
      dev_bucketBegin, 
      dev_bucketEnd,
      numParticles, 
      static_cast<unsigned int>(dev_bucketBegin.size()), 
      stream);

   /*
   // Begin DEBUG
   cudaStreamSynchronize(stream);
   std::vector<unsigned int> h_bucketBegin(dev_bucketBegin.size());
   std::vector<unsigned int> h_bucketEnd(dev_bucketEnd.size());
   dev_bucketBegin.copyArrayToHost(&h_bucketBegin[0]);
   dev_bucketEnd.copyArrayToHost(&h_bucketEnd[0]);
   FILE *dbgBounds = fopen("dbgBounds.txt", "w");
   for(int i = 0; i < h_bucketBegin.size(); i++)
   {
      fprintf(dbgBounds, "%u %u\n", h_bucketBegin[i], h_bucketEnd[i]);
   }
   fclose(dbgBounds);
   // End DEBUG
   */

   // Sum area enforces coalesced memory loads and stores, but to do
   // this it can sometimes read more data than it needs. The extra
   // 31 elements in each array exist to ensure that no uninitialized
   // memory is read
   DevMem<float, ParticleAllocator> dev_a1(numParticles + dev.warpSize-1);
   DevMem<float, ParticleAllocator> dev_a2(numParticles + dev.warpSize-1);
   DevMem<float, ParticleAllocator> dev_a3(numParticles + dev.warpSize-1);
   DevMem<float, ParticleAllocator> dev_a4(numParticles + dev.warpSize-1);
   DevMem<float, DevMemReuse> dev_a1Sum(NX1 * NY);
   DevMem<float, DevMemReuse> dev_a2Sum(NX1 * NY);
   DevMem<float, DevMemReuse> dev_a3Sum(NX1 * NY);
   DevMem<float, DevMemReuse> dev_a4Sum(NX1 * NY);

   dev_a1Sum.zeroMem();
   dev_a2Sum.zeroMem();
   dev_a3Sum.zeroMem();
   dev_a4Sum.zeroMem();
   
   // Calculate a1, a2, a3, and a4 for the now sorted particles
   threadsInBlock = MAX_THREADS_PER_BLOCK;
   blockSize = new dim3(threadsInBlock);
   numBlocks = new dim3(static_cast<unsigned int>(calcNumBlocks(threadsInBlock, numParticles)));
   calcA<<<*numBlocks, *blockSize, 0, stream>>>(
      dev_particleLocations.getPtr(),
      dev_a1.getPtr(),
      dev_a2.getPtr(),
      dev_a3.getPtr(),
      dev_a4.getPtr(),
      numParticles, 
      NY, DX, DY,
      cold && electron
      );
   delete numBlocks;
   delete blockSize;
   checkForCudaError("calcA failed");

   threadsInBlock = MAX_THREADS_PER_BLOCK / 8;
   //threadsInBlock = dev.maxThreadsPerBlock / 8;
   blockSize = new dim3(threadsInBlock);
   numBlocks = new dim3(static_cast<unsigned int>(calcNumBlocks(threadsInBlock, NY * NX1)));
   sharedMemoryBytes = 
      4 * sizeof(float) * threadsInBlock * particlesToBuffer + 
      2 * sizeof(uint2);
   uint2 tmpVal;
   tmpVal.x = numParticles;
   tmpVal.y = 0;
#ifndef NO_THRUST
   DevMem<uint2, ParticleAllocator> dev_maxMinArray(numBlocks->x, tmpVal);
#else
   DevMem<uint2, ParticleAllocator> dev_maxMinArray(numBlocks->x);
   setDeviceArray(dev_maxMinArray.getPtr(), dev_maxMinArray.size(), tmpVal);
#endif
   cudaStreamSynchronize(stream);
   checkForCudaError("Finished prep for sumArea");
   sumArea<<<*numBlocks, *blockSize, sharedMemoryBytes, stream>>>(
      dev_a1Sum.getPtr(), dev_a2Sum.getPtr(), 
      dev_a3Sum.getPtr(), dev_a4Sum.getPtr(),
      dev_bucketBegin.getPtr(), dev_bucketEnd.getPtr(),
      dev_maxMinArray.getPtr(),
      dev_a1.getPtr(), dev_a2.getPtr(), 
      dev_a3.getPtr(), dev_a4.getPtr(),
      NY * NX1, numParticles, particlesToBuffer
      );
   delete numBlocks;
   delete blockSize;
   dev_maxMinArray.freeMem();
   dev_a1.freeMem();
   dev_a2.freeMem();
   dev_a3.freeMem();
   dev_a4.freeMem();
   checkForCudaError("sumArea");

   // Calculate the particles effect on rho
   threadsX = MAX_THREADS_PER_BLOCK / 4;
   threadsY = 1;
   blockSize = new dim3(threadsX, threadsY);
   numBlocks = new dim3(static_cast<unsigned int>(calcNumBlocks(threadsX, NX1)),
                        static_cast<unsigned int>(calcNumBlocks(threadsY, NY)));
   sharedMemoryBytes = 8 * (threadsX + 1) * sizeof(float);
   uint2 tmpUint2;
   tmpUint2.x = numParticles;
   tmpUint2.y = 0;
#ifndef NO_THRUST
   DevMem<uint2, ParticleAllocator> topRowBlockBoundaries(numBlocks->x * numBlocks->y, tmpUint2);
   DevMem<uint2, ParticleAllocator> bottomRowBlockBoundaries(numBlocks->x * numBlocks->y, tmpUint2);
#else
   DevMem<uint2, ParticleAllocator> topRowBlockBoundaries(numBlocks->x * numBlocks->y);
   DevMem<uint2, ParticleAllocator> bottomRowBlockBoundaries(numBlocks->x * numBlocks->y);
   setDeviceArray(topRowBlockBoundaries.getPtr(), topRowBlockBoundaries.size(), tmpUint2);
   setDeviceArray(bottomRowBlockBoundaries.getPtr(), bottomRowBlockBoundaries.size(), tmpUint2);
#endif
   cudaStreamSynchronize(stream);
   checkForCudaError("Finished prep for densGridPoints");
   densGridPoints<<<*numBlocks, *blockSize, sharedMemoryBytes, stream>>>(
      dev_rho.getPtr(),
      NX1,
      NY,
      dev_a1Sum.getPtr(),
      dev_a2Sum.getPtr(),
      dev_a3Sum.getPtr(),
      dev_a4Sum.getPtr(),
      cold,
      numParticles,
      NIJ
      );
   delete numBlocks;
   delete blockSize;
   topRowBlockBoundaries.freeMem();
   bottomRowBlockBoundaries.freeMem();
   checkForCudaError("densGridPoints failed");

}

/******************************************************************************
  Function: dens
  Purpose: Find the charge density at grid points
  Parameters:
  -------------------
  dev_rho - The charge density at each grid point from 0 <= x < NX and 
            0 <= y < NY. rho is defined as rhoi - rhoe
  dev_rhoe - The charge density of the electrons at each grid point from 
             0 <= x < NX and 0 <= y < NY
  dev_rhoi - The charge density of the ions at each grid point from 
             0 <= x < NX and 0 <= y < NY
  dev_eleHot[][5] - The hot electron array, dimension 2 is defined as follows:
                [x][0] = Position x
                [x][1] = Position y
                [x][2] = Velocity x
                [x][3] = Velocity y
                [x][4] = Velocity z
  dev_eleCold[][5] - The cold electron array, dimension 2 is defined as follows:
                [x][0] = Position x
                [x][1] = Position y
                [x][2] = Velocity x
                [x][3] = Velocity y
                [x][4] = Velocity z
  dev_ionHot[][5] - The hot ion array, dimension 2 is defined as follows:
                [x][0] = Position x
                [x][1] = Position y
                [x][2] = Velocity x
                [x][3] = Velocity y
                [x][4] = Velocity z
  dev_ionCold[][5] - The cold ion array, dimension 2 is defined as follows:
                [x][0] = Position x
                [x][1] = Position y
                [x][2] = Velocity x
                [x][3] = Velocity y
                [x][4] = Velocity z
  numHotElectrons - The number of particles in dev_eleHot
  numColdElectrons - The number of particles in dev_eleCold
  numHotIons - The number of particles in dev_ionHot
  numColdIons - The number of particles in dev_ionCold
******************************************************************************/
void dens(DevMemF &dev_rho,
          DevMemF &dev_rhoe,
          DevMemF &dev_rhoi,
          const DevMem<float2> &d_eleHotLoc, const DevMem<float2> &d_eleColdLoc, 
          const DevMem<float2> &d_ionHotLoc, const DevMem<float2> &d_ionColdLoc,
          unsigned int numHotElectrons, unsigned int numColdElectrons,
          unsigned int numHotIons, unsigned int numColdIons)
{
   static bool first = true;
   dim3 *numBlocks;
   dim3 *blockSize;
   int threadsInBlock;

   // Clear Arrays
   dev_rho.zeroMem();
   dev_rhoe.zeroMem();
   dev_rhoi.zeroMem();

   static cudaStream_t stream;
   if(first)
   {
      cudaStreamCreate(&stream);
   }
   // Calculate the rho from the hot electrons
   calcIntermediateRho(dev_rhoe, d_eleHotLoc, 
      numHotElectrons, false, true, stream);
   // Calculate the rho from the cold electrons
   calcIntermediateRho(dev_rhoe, d_eleColdLoc,
      numColdElectrons, true, true, stream);
   // Calculate the rho from the hot ions
   calcIntermediateRho(dev_rhoi, d_ionHotLoc,
      numHotIons, false, false, stream);
   // Calculate the rho from the cold ions
   calcIntermediateRho(dev_rhoi, d_ionColdLoc,
      numColdIons, true, false, stream);

   // Double the rho at the top and bottom of the grid
   threadsInBlock = MAX_THREADS_PER_BLOCK;
   blockSize = new dim3(threadsInBlock);
   numBlocks = new dim3(static_cast<unsigned int>(calcNumBlocks(threadsInBlock, NX)));
   cudaThreadSynchronize();
   checkForCudaError("Before fixRhoGridTopBottom");
   fixRhoGridTopBottom<<<*numBlocks, *blockSize>>>(
      dev_rhoe.getPtr(),
      dev_rhoi.getPtr(),
      NX1, NY);
   delete blockSize;
   delete numBlocks;
   checkForCudaError("fixRhoGridTopBottom");

   cudaThreadSynchronize();
   checkForCudaError("Before rhoi - rhoe");
   //////////////////////////////////////////////////////
   //
   // On Linux systems this thrust call took 14 ms while
   // the kernel call took less than one. I've left the
   // thrust code here in case its ever any faster.
   //
   // Windows did not display these runtiem problems
   //
   /////////////////////////////////////////////////////
   // Set rho = rhoi - rhoe
   //#ifndef NO_THRUST
   //thrust::transform(dev_rhoi.getThrustPtr(),
   //   dev_rhoi.getThrustPtr() + dev_rhoi.size(),
   //   dev_rhoe.getThrustPtr(), 
   //   dev_rho.getThrustPtr(),
   //   thrust::minus<float>());
   //#//else
   subVector(dev_rhoi.getPtr(), dev_rhoe.getPtr(), 
             dev_rho.getPtr(), static_cast<unsigned int>(dev_rhoi.size()));
   //#endif
   first = false;
}
