#include "commandline_options.h"

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include "pic_utils.h"

CommandlineOptions* CommandlineOptions::m_ref(NULL);

CommandlineOptions::CommandlineOptions()
{
   m_description = new po::options_description("Options");
   m_vm = new po::variables_map();
}

CommandlineOptions::~CommandlineOptions()
{
}

CommandlineOptions & CommandlineOptions::getRef()
{
   if(m_ref == NULL)
   {
      m_ref = new CommandlineOptions();
   }

   return *m_ref;
}

bool CommandlineOptions::parseArguments(int argc, char * argv[])
{
   bool returnVal = true;

   // Declare the supported options.
   m_description->add_options()
       ("help,h", "Print help")
       ("max-time,t", po::value<float>(&m_maxSimTime)->default_value(100), 
        "Set the maximum time time")
       ("ascii", po::value<bool>(&m_logInAscii)->default_value(false),
        "Log in ASCII")
       ("log-interval,l", po::value<int>(&m_logInterval)->default_value(5),
        "The number of info intervals between output files")
       ("x,x", po::value<int>(&m_nx1)->default_value(512), "Width of grid")
       ("y,y", po::value<int>(&m_ny1)->default_value(4096), "Height of grid")
       ("sigma-he", po::value<float>(&m_sigma3)->default_value(1.0), "Sigma Hot Electrons")
       ("sigma-ce", po::value<float>(&m_sigma)->default_value(10.0), "Sigma Cold Electrons")
       ("sigma-hi", po::value<float>(&m_sigma2)->default_value((float)0.3), "Sigma Hot Ions")
       ("sigma-ci", po::value<float>(&m_sigma1)->default_value(10.0), "Sigma Cold Ions")
       ("restart-index", po::value<unsigned int>(&m_restartPoint)->default_value(0), 
        "File index number for restart point")
       ("restart-dir", po::value<std::string>(&m_restartDir)->default_value("."),
        "The directory the files to restart from are located in. (Ignored if restart-index not used)")
   ;
   try
   {
      po::store(po::parse_command_line(argc, argv, *m_description), *m_vm);
   }
   catch (po::error err)
   {
      errExit(err.what());
   }

   po::notify(*m_vm);

   if (m_vm->count("help")) {
       std::cout << *m_description << std::endl;
       exit(0);
   }

   if((m_nx1 & (m_nx1 - 1)) != 0) // Not a power of two
   {
      errExit("x is not a power of 2!");
   }
   else if((m_ny1 & (m_ny1 - 1)) != 0) // Not a power of two
   {
      errExit("y is not a power of 2!");
   }
   
   return returnVal;
}

void CommandlineOptions::saveOptions(const char fileName[]) const
{
}

