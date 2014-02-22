#ifndef FIELD_H
#define FIELD_H

#include "typedefs.h"
#include "dev_stream.h"
#include "pitched_ptr.h"

void field(PitchedPtr<float> &ex,
           PitchedPtr<float> &ey,
           const DevMemF &phi,
           DevStream &stream1,
           DevStream &stream2);

#endif

