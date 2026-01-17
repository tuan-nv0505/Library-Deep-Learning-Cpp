//
// Created by Nguyễn Tuấn on 16/01/2026.
//

#include "Tensor.h"

#include <random>

std::vector<double>& deep_cpp::Tensor::getData()
{
    return this->data;
}

std::vector<double>& deep_cpp::Tensor::getGrad()
{
    return this->grad;
}

const std::vector<int>& deep_cpp::Tensor::getShape() const
{
    return this->shape;
}

const std::vector<int>& deep_cpp::Tensor::getStrides() const
{
    return this->strides;
}

void deep_cpp::Tensor::print_recursive(std::ostream& os, int dim, int offset, int indent) const
{
    if (dim == this->shape.size() - 1)
    {
        os << "[";
        for (int i = 0; i < this->shape[dim]; i++)
        {
            os << std::fixed << std::setprecision(3) << data[offset + i * strides[dim]];
            if (i < shape[dim] - 1)
                os << ", ";
        }
        os << "]";
    }
    else
    {
        os << "[";
        for (int i = 0; i < shape[dim]; i++)
        {
            if (i > 0)
                os << std::string(indent + 1, ' ');

            print_recursive(os, dim + 1, offset + i * strides[dim], indent + 1);

            if (i < shape[dim] - 1)
                os << ",\n";
        }
        os << "]";
    }
}

int deep_cpp::Tensor::getSize() {
    int size = 1;
    for (const int& dim : this->shape)
        size *= dim;
    return size;
}

bool deep_cpp::Tensor::getRequiresGrad() {
    return this->requires_grad;
}

void deep_cpp::Tensor::setRequiresGrad(bool requires_grad) {
    this->requires_grad = requires_grad;
}

deep_cpp::Tensor::Tensor(std::vector<int>&& shape, bool requires_grad)
{
    this->shape = std::move(shape);
    this->requires_grad = requires_grad;

    int size = this->getSize();
    if (this->requires_grad)
        this->grad.assign(size, 0.0);

    this->makeStrides();
}

void deep_cpp::Tensor::makeStrides() {
    this->strides.resize(this->shape.size());
    int current_stride = 1;
    for (int i = this->strides.size() - 1; i >= 0; i--)
    {
        this->strides[i] = current_stride;
        current_stride *= this->shape[i];
    }
}

void deep_cpp::Tensor::reshape(std::vector<int> &&shape) {
    int newSize = 1;
    for (const int& dim : shape)
        newSize *= dim;

    if (newSize != this->getSize())
        throw std::runtime_error("Tensor reshape error");

    this->shape = std::move(shape);
    this->makeStrides();
}

deep_cpp::Tensor deep_cpp::zeros(std::vector<int>&& shape, bool requires_grad)
{
    Tensor tensor(std::move(shape), requires_grad);
    tensor.getData().assign(tensor.getSize(), 0.0);
    return tensor;
}

deep_cpp::Tensor deep_cpp::ones(std::vector<int>&& shape, bool requires_grad)
{
    Tensor tensor(std::move(shape), requires_grad);
    tensor.getData().assign(tensor.getSize(), 1.0);
    return tensor;
}

deep_cpp::Tensor deep_cpp::rand(std::vector<int>&& shape, bool requires_grad)
{
    Tensor tensor(std::move(shape), requires_grad);
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<double> distribution(0.0, 1.0);

    tensor.getData().resize(tensor.getSize());
    for (double& x : tensor.getData())
        x = distribution(gen);

    return tensor;
}


