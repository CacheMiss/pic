#pragma once

#include <sstream>

////////////////////////////////////////////////////////////////////////////////
//
// Note: Please #define _DEBUG for additional debug safety checking
//
////////////////////////////////////////////////////////////////////////////////

#include "error_check.h"

template<class T>
struct PitchedPtr
{
   T* ptr;
   std::size_t pitch;
   std::size_t width;
   std::size_t widthBytes;
   std::size_t height;
};

template<class T>
void allocatePitchedPtr(PitchedPtr<T> &p, std::size_t width, std::size_t height)
{
   checkCuda(cudaMallocPitch(reinterpret_cast<void**>(&p.ptr), 
                             &p.pitch, 
                             width * sizeof(T), 
                             height));
   p.width = width;
   p.widthBytes = p.width * sizeof(T);
   p.height = height;

#ifdef _DEBUG
   checkCuda(cudaMemset2D(p.ptr, p.pitch, 0, p.widthBytes, p.height));
#endif
}

template<class T>
void freePitchedPtr(PitchedPtr<T> &p)
{
   if(p.ptr != NULL)
   {
      cudaFree(p.ptr);
      p.ptr = NULL;
   }
}

template<class T>
#ifdef __CUDACC__
__host__
__device__
#endif
T& resolvePitchedPtr(PitchedPtr<T> &p, std::size_t x, std::size_t y)
{
#ifdef _DEBUG
   if(p.width <= x || p.height <= y)
   {
      printf("ERROR: resolvePitchedPtr attempted to resolve x=%u y=%u when size_x=%u and size_y=%u\n",
         x, y, p.width, p.height);
   }
#endif
   return reinterpret_cast<T*>(&reinterpret_cast<char*>(p.ptr)[y * p.pitch])[x];
}

template<class T>
#ifdef __CUDACC__
__host__
__device__
#endif
T const & resolvePitchedPtr(const PitchedPtr<T> &p, std::size_t x, std::size_t y)
{
#ifdef _DEBUG
   if(p.width <= x || p.height <= y)
   {
      printf("ERROR: resolvePitchedPtr(const) attempted to resolve x=%u y=%u when size_x=%u and size_y=%u\n",
         x, y, p.width, p.height);
   }
#endif
   return reinterpret_cast<T*>(&reinterpret_cast<char*>(p.ptr)[y * p.pitch])[x];
}

template<class T>
void fillPitchedPtr(PitchedPtr<T> &p, int val)
{
   cudaMemset2D(p.ptr, p.pitch, val, p.widthBytes, p.height);
}

// Allocate lhs and copy rhs into it
template<class T>
void duplicatePitchedPtr(PitchedPtr<T> &lhs, const PitchedPtr<T> &rhs)
{
   allocatePitchedPtr(lhs, rhs.width, rhs.height);
   checkCuda(cudaMemcpy2D(lhs.ptr, lhs.pitch, rhs.ptr, rhs.pitch, rhs.widthBytes, rhs.height, cudaMemcpyDeviceToDevice));
}

template<class T>
void copyPitchedPtr(PitchedPtr<T> &dst, PitchedPtr<T> &src)
{
   if(src.pitch != dst.pitch ||
      src.width != dst.width ||
      src.height != dst.height)
   {
      std::stringstream s;
      s << "ERROR: Attempted to call copyPitchedPtr when the source and destination were not the same size! ";
      s << std::endl;
      throw CudaRuntimeError(s.str());
   }
   checkCuda(cudaMemcpy2D(dst.ptr, dst.pitch, src.ptr, src.pitch, src.widthBytes, src.height, cudaMemcpyDeviceToDevice));
}