#ifndef BILINEAR
#define BILINEAR 

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include "Layer.hh"
#include "Tensor.hh"
class BilinearInterpolation : public Layer
{
    int scale;

    public:
    /// @brief Bilinear Interpolation
    /// @param size 
    /// @param _scale
    BilinearInterpolation(TSize size, int _scale) : Layer(size, size.Width * _scale, size.Height * _scale, size.Deep),
                                                    scale(_scale)
    {
    }

    void Forward(const std::vector<Tensor> &X)
    {
#pragma omp parallel for collapse(4)
        for (size_t batchIndex = 0; batchIndex < X.size(); batchIndex++)
        {
            for (int d = 0; d < ISize.Deep; d++)
            {
                for (int i = 0; i < OSize.Height; i++)
                {
                    for (int j = 0; j < OSize.Width; j++)
                    {
                        int y = std::min(i / scale, ISize.Height - 2);
                        int x = std::min(j / scale, ISize.Width - 2);

                        double dy = (i / (double)scale) - y;
                        double dx = (j / (double)scale) - x;

                        double p1 = X[batchIndex](d, y, x) * (1 - dx) * (1 - dy);
                        double p2 = X[batchIndex](d, y, x + 1) * dx * (1 - dy);
                        double p3 = X[batchIndex](d, y + 1, x) * (1 - dx) * dy;
                        double p4 = X[batchIndex](d, y + 1, x + 1) * dx * dy;

                        Output[batchIndex](d, i, j) = p1 + p2 + p3 + p4;
                    }
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
        {
            for (int d = 0; d < ISize.Deep; d++)
            {
                for (int i = 0; i < ISize.Height; i++)
                    for (int j = 0; j < ISize.Width; j++)
                        Gradient[batchIndex](d, i, j) = 0;

                for (int i = 0; i < OSize.Height; i++)
                {
                    for (int j = 0; j < OSize.Width; j++)
                    {
                        int y = std::min(i / scale, ISize.Height - 2);
                        int x = std::min(j / scale, ISize.Width - 2);

                        double dy = (i / (double)scale) - y;
                        double dx = (j / (double)scale) - x;

                        double delta = dout[batchIndex](d, i, j);

                        Gradient[batchIndex](d, y, x) += delta * (1 - dx) * (1 - dy);
                        Gradient[batchIndex](d, y, x + 1) += delta * dx * (1 - dy);
                        Gradient[batchIndex](d, y + 1, x) += delta * (1 - dx) * dy;
                        Gradient[batchIndex](d, y + 1, x + 1) += delta * dx * dy;
                    }
                }
            }
        }
    }

    void Save(std::ofstream &f) const
    {
        f << "upscalebilinear " << ISize.Width << " " << ISize.Height << " " << ISize.Deep << " " << scale << std::endl;
    }
};

#endif