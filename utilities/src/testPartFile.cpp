#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <utility>
#include <vector>

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
	const float NX1 = static_cast<float>(atoi(argv[2]));
	const float NY1 = static_cast<float>(atoi(argv[3]));
	std::cout << "NX1 = " << NX1 << std::endl;
	std::cout << "NY1 = " << NY1 << std::endl;

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

   const double percentSize = static_cast<double>(numParticles) / 100.0;
	int lastPercent = 0;
   std::size_t numOobPartX = 0;
   std::size_t numOobPartY = 0;
	const std::size_t sampleSize = 5;
	typedef std::vector<std::pair<float, float> > PartVector_t;
	PartVector_t sampleOobX;
	PartVector_t sampleOobY;

	for(std::size_t i = 0; i < numParticles; i++)
	{
		float x;
		float y;
		f.read(reinterpret_cast<char*>(&x), sizeof(x));
		f.read(reinterpret_cast<char*>(&y), sizeof(y));
		// Skip the velocity
		f.seekg(sizeof(float)*3, std::ios_base::cur);
		if(NX1 <= x)
		{
			error = true;
			++numOobPartX;
			if(sampleOobX.size() < sampleSize)
			{
				sampleOobX.push_back(std::make_pair(x, y));
			}
		}
		if(NY1 <= y)
		{
			error = true;
			++numOobPartY;
			if(sampleOobY.size() < sampleSize)
			{
				sampleOobY.push_back(std::make_pair(x, y));
			}
		}

		if(lastPercent < static_cast<int>(i / percentSize))
		{
			++lastPercent;
			std::cout << lastPercent << "% complete" << std::endl;
		}
	}

	if(!sampleOobX.empty())
	{
		std::cout << numOobPartX << " particles of " << numParticles << " had X values >= " << NX1 << ". A sample has been printed below." << std::endl;
		for(PartVector_t::iterator i = sampleOobX.begin(); i != sampleOobX.end(); i++)
		{
			std::cout << "x=" << (*i).first << " y=" << (*i).second << std::endl;
		}
		std::cout << std::endl;
	}
	if(!sampleOobY.empty())
	{
		std::cout << numOobPartY << " particles of " << numParticles << " had Y values >= " << NY1 << ". A sample has been printed below." << std::endl;
		for(PartVector_t::iterator i = sampleOobY.begin(); i != sampleOobY.end(); i++)
		{
			std::cout << "x=" << (*i).first << " y=" << (*i).second << std::endl;
		}
	}

	if(!error)
	{
		std::cout << argv[0] << " has been successfully verified!" << std::endl;
	}

	return 0;
}

