////////////////////////////////////////////////////////////////////////////////
// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
// 
//   http://www.apache.org/licenses/LICENSE-2.0
// 
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <cuda_runtime_api.h>
#include <cstddef>

#include "error_check.h"

class CudaAllocator
{
public:
   static CudaAllocator& getRef();

   template<class T>
   void allocate(T* &ptr, std::size_t nElements);
   template<class T>
   void free(T* m);

private:
   static CudaAllocator *m_ref;
};

template<class T>
void CudaAllocator::allocate(T* &ptr, std::size_t nElements)
{
   checkCuda(cudaMalloc(reinterpret_cast<void**>(&ptr), sizeof(T) * nElements));
}

template<class T>
void CudaAllocator::free(T* m)
{
   cudaFree(m);
}
