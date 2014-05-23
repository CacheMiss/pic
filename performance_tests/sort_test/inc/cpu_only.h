#ifndef CPU_ONLY
#define CPU_ONLY

#include "sort_thread.h"

void cpuSort(HostMem<SortThread::Particle> &cpuParticles);

#endif