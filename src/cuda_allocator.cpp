#include "cuda_allocator.h"

CudaAllocator* CudaAllocator::m_ref = NULL;

CudaAllocator& CudaAllocator::getRef()
{
   if(m_ref == NULL)
   {
      m_ref = new CudaAllocator;
   }
   return *m_ref;
}
