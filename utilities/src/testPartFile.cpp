#include <iostream>
#include <fstream>
#include <stdlib.h>

#include <boost/filesystem.hpp>

int main(int argc, char** argv)
{
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
}
