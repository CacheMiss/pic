#include <iostream>
#include <fstream>
#include <stdlib.h>

#include <boost/filesystem.hpp>

int main(int argc, char** argv)
{
	bool error = false;

	if(argc != 4)
	{
		std::cerr << "USAGE: " << argv[0] << " PARTICLE_FILE NX1 NY1" << std::endl;
		exit(0);
	}

   boost::filesystem::path path(argv[1]);
	if(!boost::filesystem::exists(path))
	{
		std::cerr << "ERROR: " << argv[1] << " does not exist!" << std::endl;
		exit(1);
	}
	std::ifstream f(path.string().c_str(), std::ios::binary);

	int numParticles;
	int numHot;
	int numCold;
	f.read(reinterpret_cast<char*>(&numParticles), sizeof(numParticles));
	f.read(reinterpret_cast<char*>(&numHot), sizeof(numHot));
	f.read(reinterpret_cast<char*>(&numCold), sizeof(numCold));

	if(numParticles != numHot + numCold)
	{
		error = true;
		std::cout << "ERROR: " << argv[0] << std::endl;
		std::cout << "  numParticles != numHot + numCold" << std::endl;
		std::cout << "  " << numParticles << " != " << numHot << " + " << numCold << std::endl;
	}

	if(!error)
	{
		std::cout << argv[0] << " has been successfully verified!" << std::endl;
	}
}
