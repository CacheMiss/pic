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
       ("sigma-he", po::value<float>(&m_sigmaHe)->default_value(1.0), "Sigma Hot Electrons")
       ("sigma-ce", po::value<float>(&m_sigmaCe)->default_value(10.0), "Sigma Cold Electrons")
       ("sigma-hi", po::value<float>(&m_sigmaHi)->default_value((float)0.3), "Sigma Hot Ions")
       ("sigma-ci", po::value<float>(&m_sigmaCi)->default_value(10.0), "Sigma Cold Ions")
       ("sigma-ce-secondary", po::value<float>(&m_sigmaCeSecondary)->default_value(10.0), 
        "Sigma for seondary cold electron injection. Use this in conjunction with the --percent-secondary option.")
       ("percentage-secondary", po::value<double>(&m_percentageSecondary)->default_value(0), 
        "The percentage (value 0-1) cold electrons that will use the secondary sigma.")
       ("b0", po::value<float>(&m_b0)->default_value(10), "B0 controls the magnetic field strength")
       ("p0", po::value<double>(&m_p0)->default_value(-15.), "The charge at the top boundary of the grid")
       ("uniform-p0", po::value<bool>(&m_uniformP0)->default_value(false), "Use a uniform value for p0 instead of a gaussian distribution")
       ("inject-width", po::value<unsigned int>(&m_injectWidth)->default_value(0), "The width of the injection area for cold particles")
       ("restart-dir", po::value<std::string>(&m_restartDir)->default_value(""),
        "The directory the files to restart from are located in.")
       ("bound-check", po::bool_switch(&m_particleBoundCheck)->default_value(false),
        "Debug option to ensure all particles remain in the grid.")
        ("output-path,o", po::value<std::string>(&m_outputPath)->default_value("run_output"), "The folder to write results to")
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
   /*
   else if((m_ny1 & (m_ny1 - 1)) != 0) // Not a power of two
   {
      errExit("y is not a power of 2!");
   }
   */

   if(m_percentageSecondary > 1 || m_percentageSecondary < 0)
   {
      errExit("--percentage-secondary must be 0 <= x <= 1");
   }

   // Inject across the entire bottom of the grid if left unspecified
   if(m_injectWidth == 0)
   {
      m_injectWidth = getNx1();
   }

   NX1 = getNx1();
   NX12 = NX1 / 2;
   NX = NX1 + 1;
   X_GRD = NX + 1;
   NY1 = getNy1();
   NY12 = NY1 / 2;
   NY = NY1 + 1;
   Y_GRD = NY + 1;
   SIGMA_HE = getSigmaHe();
   SIGMA_HI = getSigmaHi();
   SIGMA_CE = getSigmaCe();
   SIGMA_CI = getSigmaCi();
   SIGMA_CE_SECONDARY = getSigmaCeSecondary();
   PERCENT_SECONDARY = getPercentageSecondary();
   B0 = getB0();
   P0 = getP0();
   UNIFORM_P0 = getUniformP0();
   OOB_PARTICLE = static_cast<float>(NY1) + 1000.0f;
   outputPath = m_outputPath;
   errorLogName = (boost::filesystem::path(outputPath) /= "errorLog.txt").string();

   return returnVal;
}

void CommandlineOptions::saveOptions(const char fileName[]) const
{
}
