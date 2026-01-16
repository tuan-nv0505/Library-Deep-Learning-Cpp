//
// Created by Nguyễn Tuấn on 16/01/2026.
//

#ifndef LIBRARY_DEEP_LEARNING_CPP_TENSOR_H
#define LIBRARY_DEEP_LEARNING_CPP_TENSOR_H

#include <vector>
#include <iomanip>

namespace DeepCpp
{
    class Tensor
    {
    private:
        std::vector<double> data;
        std::vector<double> grad;
        std::vector<int> shape;
        std::vector<int> strides;
        int size;
    public:
        Tensor(std::vector<int>&& shape);

        void print_recursive(std::ostream& os, int dim, int offset, int indent) const;
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

        std::vector<double> getData();
        void setData(std::vector<double> data);
        std::vector<double> getGrad();
        void setGrad(std::vector<double> grad);
        std::vector<int> getShape();
        void setShape(std::vector<int> shape);
        std::vector<int> getStrides();
        void setStrides(std::vector<int> strides);
    };
}


#endif //LIBRARY_DEEP_LEARNING_CPP_TENSOR_H