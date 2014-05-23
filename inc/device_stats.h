#ifndef DEVICE_STATS
#define DEVICE_STATS

#include <cstddef>
#include <driver_types.h>

struct DeviceStats : public cudaDeviceProp
{
public:
   static DeviceStats & getRef(int dev=-1);
   std::size_t getTotalMemBytes() const;
   std::size_t getFreeMemBytes() const;
   std::size_t getTotalMemMb() const;
   std::size_t getFreeMemMb() const;
   double getPercentFreeMem() const;
private:
   DeviceStats(){}
   static DeviceStats *m_ref;
};

#endif

