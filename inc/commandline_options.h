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
   ~CommandlineOptions();
   static CommandlineOptions & getRef();
   bool parseArguments(int argv, char * argc[]);
   inline float getMaxSimTime() const;
   inline bool getLogInAscii() const;
   inline int getLogInterval() const;
   inline int getNx1() const;
   inline int getNy1() const;
   inline const std::string& getRestartDir() const;
   inline float getSigmaHe() const;
   inline float getSigmaCe() const;
   inline float getSigmaHi() const;
   inline float getSigmaCi() const;
   inline bool  getParticleBoundCheck() const;
   inline float getB0() const;
   inline unsigned int getInjectWidth() const;
   inline std::string getOutputPath() const;
   void saveOptions(const char fileName[]) const;

   private:
   CommandlineOptions();
   static CommandlineOptions *m_ref;
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
   float m_b0;
   unsigned int m_injectWidth;
   std::string m_restartDir;
   bool m_particleBoundCheck;
   std::string m_outputPath;
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

inline bool CommandlineOptions::getParticleBoundCheck() const
{
   return m_particleBoundCheck;
}

inline float CommandlineOptions::getB0() const
{
   return m_b0;
}

inline unsigned int CommandlineOptions::getInjectWidth() const
{
   return m_injectWidth;
}

inline std::string CommandlineOptions::getOutputPath() const
{
   return m_outputPath;
}

#endif
