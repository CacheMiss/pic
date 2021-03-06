////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014, Stephen C. Sewell
// All rights reserved.
// 
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
// 
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
// 
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR
// ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
// (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
// LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
// ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
////////////////////////////////////////////////////////////////////////////////
#ifndef COMMANDLINE_OPTIONS
#define COMMANDLINE_OPTIONS

#include <string>

namespace boost
{
   namespace program_options
   {
      class options_description;
      class variables_map;
   };
};

namespace po = boost::program_options;

class CommandlineOptions
{
   public:
   CommandlineOptions();
   ~CommandlineOptions();
   bool parseArguments(int argv, const char * argc[]);
   inline float getMaxSimTime() const;
   inline bool getLogInAscii() const;
   inline int getLogInterval() const;
   inline int getNx1() const;
   inline int getNy1() const;
   inline const std::string& getRestartDir() const;
   inline unsigned int getRestartIdx() const;
   inline float getSigmaHe() const;
   inline float getSigmaCe() const;
   inline float getSigmaHi() const;
   inline float getSigmaCi() const;
   inline float getSigmaHePerp() const;
   inline float getSigmaHiPerp() const;
   inline float getSigmaCeSecondary() const;
   inline double getPercentageSecondary() const;
   inline bool  getParticleBoundCheck() const;
   inline float getB0() const;
   inline double getP0() const;
   inline bool getUniformP0() const;
   inline unsigned int getInjectWidth() const;
   inline std::string getOutputPath() const;
   inline bool getDisableRandRestore() const;
   inline bool getProfile() const;

   private:
   po::options_description *m_description;
   po::variables_map *m_vm;

   float m_maxSimTime;
   bool  m_logInAscii;
   int m_logInterval;
   int m_nx1;
   int m_ny1;
   float m_sigmaHe;
   float m_sigmaHi;
   float m_sigmaCe;
   float m_sigmaCi;
   float m_sigmaHePerp;
   float m_sigmaHiPerp;
   float m_sigmaCeSecondary;
   double m_percentageSecondary;
   float m_b0;
   double m_p0;
   bool m_uniformP0;
   unsigned int m_injectWidth;
   std::string m_restartDir;
   unsigned int m_restartIdx;
   bool m_particleBoundCheck;
   std::string m_outputPath;
   bool m_disableRandRestore;
   bool m_profile;
};

inline float CommandlineOptions::getMaxSimTime() const
{
   return m_maxSimTime;
}

inline bool CommandlineOptions::getLogInAscii() const
{
   return m_logInAscii;
}

inline int CommandlineOptions::getLogInterval() const
{
   return m_logInterval;
}

inline int CommandlineOptions::getNx1() const
{
   return m_nx1;
}

inline int CommandlineOptions::getNy1() const
{
   return m_ny1;
}

inline const std::string& CommandlineOptions::getRestartDir() const
{
   return m_restartDir;
}

inline unsigned int CommandlineOptions::getRestartIdx() const
{
   return m_restartIdx;
}

inline float CommandlineOptions::getSigmaHe() const
{
   return m_sigmaHe;
}

inline float CommandlineOptions::getSigmaCe() const
{
   return m_sigmaCe;
}

inline float CommandlineOptions::getSigmaHi() const
{
   return m_sigmaHi;
}

inline float CommandlineOptions::getSigmaCi() const
{
   return m_sigmaCi;
}

inline float CommandlineOptions::getSigmaCeSecondary() const
{
   return m_sigmaCeSecondary;
}

inline double CommandlineOptions::getPercentageSecondary() const
{
   return m_percentageSecondary;
}

inline bool CommandlineOptions::getParticleBoundCheck() const
{
   return m_particleBoundCheck;
}

inline float CommandlineOptions::getB0() const
{
   return m_b0;
}

inline double CommandlineOptions::getP0() const
{
   return m_p0;
}

inline bool CommandlineOptions::getUniformP0() const
{
   return m_uniformP0;
}

inline unsigned int CommandlineOptions::getInjectWidth() const
{
   return m_injectWidth;
}

inline std::string CommandlineOptions::getOutputPath() const
{
   return m_outputPath;
}

inline float CommandlineOptions::getSigmaHePerp() const
{
   return m_sigmaHePerp;
}

inline float CommandlineOptions::getSigmaHiPerp() const
{
   return m_sigmaHiPerp;
}

inline bool CommandlineOptions::getDisableRandRestore() const
{
   return m_disableRandRestore;
}

inline bool CommandlineOptions::getProfile() const
{
   return m_profile;
}

#endif
