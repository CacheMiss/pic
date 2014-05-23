#include <tbb/parallel_sort.h>
#include "sort_thread.h"

void cpuSort(HostMem<SortThread::Particle> &cpuParticles)
{
   tbb::parallel_sort(cpuParticles.begin(), cpuParticles.end());
}