#include "Layer.hh"
#include "Tensor.hh"
#include <iostream>

class RectifierLinearUnit : public Layer
{

    int Size;

public:
    RectifierLinearUnit(TSize size) : Layer(size)
    {
        Size = size.Deep * size.Height * size.Width;
    };

    void Forward(const std::vector<Tensor> &X)
    {
#pragma omp parallel for collapse(2)
        for (size_t batchIndex = 0; batchIndex < X.size(); batchIndex++)
        {
            for (int i = 0; i < Size; i++)
            {
                if (X[batchIndex][i] > 0)
                {
                    Output[batchIndex][i] = X[batchIndex][i];
                    Gradient[batchIndex][i] = 1;
                }
                else
                {
                    Output[batchIndex][i] = 0;
                    Gradient[batchIndex][i] = 0;
                }
            }
        }
    }
    void Backward(const std::vector<Tensor> &dout, const std::vector<Tensor> &X, bool calc_dX)
    {
        if (!calc_dX)
            return;

#pragma omp parallel for collapse(2)
        for (size_t batchIndex = 0; batchIndex < dout.size(); batchIndex++)
            for (int i = 0; i < Size; i++)
                Gradient[batchIndex][i] *= dout[batchIndex][i];
    }

    void Save(std::ofstream &f) const
    {
        f << "relu " << ISize.Width << " " << ISize.Height << " " << ISize.Deep << std::endl;
    }
};