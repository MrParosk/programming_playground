
rm -rf build
mkdir build
(cd build && cmake ../ && CXX=Clang cmake --build . && ./neural_network)
