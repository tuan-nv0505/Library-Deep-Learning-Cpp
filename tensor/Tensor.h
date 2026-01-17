//
// Created by Nguyễn Tuấn on 16/01/2026.
//

#ifndef LIBRARY_DEEP_LEARNING_CPP_TENSOR_H
#define LIBRARY_DEEP_LEARNING_CPP_TENSOR_H

#include <vector>
#include <iomanip>

namespace deep_cpp
{
    class Tensor
    {
    private:
        std::vector<double> data;
        std::vector<double> grad;
        std::vector<int> shape;
        std::vector<int> strides;
        bool requires_grad;
        void makeStrides();
    public:
        Tensor(std::vector<int>&& shape, bool requires_grad = false);


        std::vector<double>& getData();
        std::vector<double>& getGrad();
        const std::vector<int>& getShape() const;
        const std::vector<int>& getStrides() const;
        int getSize();
        bool getRequiresGrad();
        void setRequiresGrad(bool requires_grad);
        void print_recursive(std::ostream& os, int dim, int offset, int indent) const;
        void reshape(std::vector<int>&& shape);
        friend std::ostream& operator<<(std::ostream& os, const Tensor& tensor)
        {
            if (tensor.data.empty())
            {
                os << "Tensor([])";
                return os;
            }
            tensor.print_recursive(os, 0, 0, 0);
            return os;
        }
    };

    Tensor zeros(std::vector<int>&& shape, bool requires_grad = false);
    Tensor ones(std::vector<int>&& shape, bool requires_grad = false);
    Tensor rand(std::vector<int>&& shape, bool requires_grad = false);
}


#endif //LIBRARY_DEEP_LEARNING_CPP_TENSOR_H