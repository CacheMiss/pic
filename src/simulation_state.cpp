#include "simulation_state.h"

SimulationState *SimulationState::m_ref = 0;

SimulationState & SimulationState::getRef()
{
   if(m_ref == 0)
   {
      m_ref = new SimulationState();
   }
   return *m_ref;
}

