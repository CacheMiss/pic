#ifndef SIMULATION_STATE_H
#define SIMULATION_STATE_H

class SimulationState
{
   public:
   ~SimulationState(){};
   static SimulationState & getRef();

   double simTime;
   unsigned int iterationNum;
   unsigned int numEleHot;
   unsigned int numEleCold;
   unsigned int numIonHot;
   unsigned int numIonCold;

   private:
   static SimulationState *m_ref;
   SimulationState()
     : simTime(0)
     , iterationNum(0) 
     , numEleHot(0)
     , numEleCold(0)
     , numIonHot(0)
     , numIonCold(0)
   {}

};

#endif
