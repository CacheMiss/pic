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
#include "commandline_options.h"

#include <boost/filesystem.hpp>
#include <boost/program_options.hpp>

#include "pic_utils.h"

CommandlineOptions::CommandlineOptions()
{
   m_description = new po::options_description("Options");
   m_vm = new po::variables_map();
}

CommandlineOptions::~CommandlineOptions()
{
}

bool CommandlineOptions::parseArguments(int argc, const char* argv[])
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
       ("sigma-he-perp", po::value<float>(&m_sigmaHePerp)->default_value(0.0), 
        "Perpendicular sigma of hot electrons. If no value is specified here, perpendicular and horizontal sigma are the same.")
       ("sigma-hi-perp", po::value<float>(&m_sigmaHiPerp)->default_value(0.0), 
        "Perpendicular sigma of hot ions. If no value is specified here, perpendicular and horizontal sigma are the same.")
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
       ("restart-idx", po::value<unsigned int>(&m_restartIdx)->default_value(0),
        "The log index to restart from. This must be used in conjunction with --restart-dir.")
       ("bound-check", po::bool_switch(&m_particleBoundCheck)->default_value(false),
        "Debug option to ensure all particles remain in the grid.")
       ("output-path,o", po::value<std::string>(&m_outputPath)->default_value("run_output"), "The folder to write results to")
       ("disable-rand-restore", po::bool_switch(&m_disableRandRestore)->default_value(false),
        "Do not restore the random number state when loading a run")
       ("profile", po::bool_switch(&m_profile)->default_value(false),
        "Run one iteration then exit. Sets --disable-rand-restore")
   ;
   try
   {
      po::store(po::parse_command_line(argc, argv, *m_description), *m_vm);
   }
   catch (po::error& err)
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

   if(m_profile)
   {
      m_disableRandRestore = true;
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
   SIGMA_HE_PERP = getSigmaHePerp();
   SIGMA_HI_PERP = getSigmaHiPerp();
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
