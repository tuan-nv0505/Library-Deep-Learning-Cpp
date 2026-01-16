cmake_minimum_required(VERSION 4.1)
project(Library_Deep_Learning_Cpp)

set(CMAKE_CXX_STANDARD 17)

add_executable(Library_Deep_Learning_Cpp main.cpp
        tensor/Tensor.cpp
        tensor/Tensor.h
)
