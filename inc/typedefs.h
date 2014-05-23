#ifndef TYPEDEFS_H
#define TYPEDEFS_H

#include <cuda_runtime_api.h>

#include "array2d.h"
#include "dev_mem.h"

typedef std::vector<float> HostVecF;
typedef std::vector<unsigned int> HostVecUi;
typedef std::vector<float> DevVecF;
typedef std::vector<float2> DevVecF2;
typedef std::vector<unsigned int> DevVecUi;
typedef Array2d<float, std::vector<float> > Array2dF;

typedef DevMem<float> DevMemF;
typedef DevMem<float2> DevMemF2;
typedef DevMem<float3> DevMemF3;

#endif

