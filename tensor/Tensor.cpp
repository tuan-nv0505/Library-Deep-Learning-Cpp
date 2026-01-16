//
// Created by Nguyễn Tuấn on 16/01/2026.
//

#include "Tensor.h"

void DeepCpp::Tensor::print_recursive(std::ostream& os, int dim, int offset, int indent) const
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

DeepCpp::Tensor::Tensor(std::vector<int>&& shape)
{
    this->shape = std::move(shape);

    this->size = 1;
    for (const int& dim : this->shape)
        this->size *= dim;
    this->data.resize(this->size);
    this->grad.resize(this->size);
    this->data.assign(this->data.size(), 0.0);
    this->grad.assign(this->grad.size(), 0.0);

    this->strides.resize(this->shape.size());
    int current_stride = 1;
    for (int i = this->strides.size() - 1; i >= 0; i--)
    {
        this->strides[i] = current_stride;
        current_stride *= this->shape[i];
    }
}

std::vector<double> DeepCpp::Tensor::getData()
{
    return this->data;
}

void DeepCpp::Tensor::setData(std::vector<double> data)
{
    this->data = data;
}

std::vector<double> DeepCpp::Tensor::getGrad()
{
    return this->grad;
}

void DeepCpp::Tensor::setGrad(std::vector<double> grad)
{
    this->grad = grad;
}

std::vector<int> DeepCpp::Tensor::getShape()
{
    return this->shape;
}

void DeepCpp::Tensor::setShape(std::vector<int> shape)
{
    this->shape = shape;
}

std::vector<int> DeepCpp::Tensor::getStrides()
{
    return this->strides;
}

void DeepCpp::Tensor::setStrides(std::vector<int> strides)
{
    this->strides = strides;
}
