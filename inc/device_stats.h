#ifndef DEVICE_STATS
#define DEVICE_STATS

#include <driver_types.h>

struct DeviceStats : public cudaDeviceProp
{
public:
   static DeviceStats & getRef();
private:
   DeviceStats(){}
   static DeviceStats *m_ref;
};

#endif

