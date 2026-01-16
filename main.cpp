#include "tensor/Tensor.h"
#include <iostream>

int main(int argc, char* argv[]) {
    DeepCpp::Tensor tensor({5, 5, 3});
    std::cout << tensor;

    return 0;
}