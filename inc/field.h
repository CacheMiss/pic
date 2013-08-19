#ifndef FIELD_H
#define FIELD_H

#include "typedefs.h"
#include "pitched_ptr.h"

void field(PitchedPtr<float> &ex,
           PitchedPtr<float> &ey,
           const DevMemF &phi);

#endif

