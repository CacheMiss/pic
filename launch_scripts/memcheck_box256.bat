cd ..
cuda-memcheck .\build\windows\Win32Debug\pic.exe -x 256 -y 256 -t 4000 --sigma-he 1 --sigma-ce 1 --sigma-hi 1 --sigma-ci 1 -l 20
cd launch_scripts
