#include "tensor/Tensor.h"
#include <iostream>

int main(int argc, char* argv[]) {
    std::shared_ptr<deep_cpp::Tensor> tensor = std::make_shared<deep_cpp::Tensor>(deep_cpp::rand({3, 3}, true));
    std::cout << *tensor << std::endl;
    tensor->reshape({9, 1});
    std::cout << *tensor << std::endl;
    return 0;
}