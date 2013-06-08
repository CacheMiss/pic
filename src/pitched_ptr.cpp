#include "pitched_ptr.h"

CudaPitchedAllocator* CudaPitchedAllocator::m_ref = NULL;

CudaPitchedAllocator& CudaPitchedAllocator::getRef()
{
   if(m_ref == NULL)
   {
      m_ref = new CudaPitchedAllocator;
   }
   return *m_ref;
}