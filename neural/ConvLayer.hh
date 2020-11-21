#ifndef CONVLAYER
#define CONVLAYER

#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <random>

#include "NGeneral.hpp"
#include "Optimization.hpp"

class ConvolutionalLayer : public Layer
{
public:
    std::default_random_engine generator;
    std::normal_distribution<double> distribution;

    std::vector<Tensor> Filters;
    std::vector<Tensor> FiltersGradient;
    std::vector<std::vector<Tensor>> FilterParams;

    std::vector<double> Bias;
    std::vector<double> BiasGradient;
    std::vector<std::vector<double>> BiasParams;

    int Padding;
    int Step;

    int FilterCount;
    int FilterSize;
    int FilterDepth;

public:
    ConvolutionalLayer(std::ifstream &Load, TSize size, int filterCount,
                       int filterSize, int padding,
                       int step) : Layer(size,
                                         (size.Width - filterSize + 2 * padding) / step + 1,
                                         (size.Height - filterSize + 2 * padding) / step + 1,
                                         filterCount),
                                   distribution(0.0, sqrt(2.0 / (filterSize * filterSize * size.Deep))),
                                   Padding(padding), Step(step),
                                   FilterCount(filterCount), FilterSize(filterSize)
    {
        for (int i = 0; i < filterCount; i++)
        {
            Filters.push_back(Tensor(filterSize, filterSize, FilterDepth));
            FiltersGradient.push_back(Tensor(filterSize, filterSize, FilterDepth));
            Bias.push_back(0);
            BiasGradient.push_back(0);
        }
        CreateParameters();
        LoadWeightsFromFile(Load);
    }
    ConvolutionalLayer(TSize size, int filterCount,
                       int filterSize, int padding,
                       int step) : Layer(size,
                                         (size.Width - filterSize + 2 * padding) / step + 1,
                                         (size.Height - filterSize + 2 * padding) / step + 1,
                                         filterCount),
                                   distribution(0.0, sqrt(2.0 / (filterSize * filterSize * size.Deep)))

    {
        Padding = padding;
        Step = step;
        FilterCount = filterCount;
        FilterSize = filterSize;
        FilterDepth = size.Deep;
        std::cout<< "Burada ne döndüğünü anlamak istiyorum fc : " << filterCount ;
        for (int i = 0; i < filterCount; i++)
        {
            std::cout<< std::endl <<"tur fc : " << filterCount ;
            Filters.push_back(Tensor(filterSize, filterSize, FilterDepth));
            FiltersGradient.push_back(Tensor(filterSize, filterSize, FilterDepth));
            Bias.push_back(0);
            BiasGradient.push_back(0);
        }
        CreateParameters();
        CreateWeights();
    }

    void CreateParameters()
    {
        //initparam fortrain
        for (int i = 0; i < 3; i++)
        {
            FilterParams.push_back(std::vector<Tensor>(FilterCount, Tensor(FilterSize, FilterSize, FilterDepth)));
            BiasParams.push_back(std::vector<double>(FilterCount, 0));
        }
    }

    void CreateWeights()
    {
        for (int index = 0; index < FilterCount; index++)
        {
            for (int i = 0; i < FilterSize; i++)
                for (int j = 0; j < FilterSize; j++)
                    for (int k = 0; k < FilterDepth; k++)
                        Filters[index](k, i, j) = distribution(generator);

            Bias[index] = 0.01;
        }
    }
    void LoadWeightsFromFile(std::ifstream &Load)
    {
        for (int index = 0; index < FilterCount; index++)
        {
            for (int d = 0; d < FilterDepth; d++)
                for (int i = 0; i < FilterSize; i++)
                    for (int j = 0; j < FilterSize; j++)
                        Load >> Filters[index](d, i, j);

            Load >> Bias[index];
        }
    }

    void Forward(const std::vector<Tensor> &X)
    {
#pragma omp parallel for collapse(4)
        for (int n = 0; n < X.size(); n++)
        {
            for (int f = 0; f < FilterCount; f++)
            {
                for (int i = 0; i < OSize.Height; i++)
                {
                    for (int j = 0; j < OSize.Width; j++)
                    {
                        double sum = Bias[f];

                        for (int k = 0; k < FilterSize; k++)
                        {
                            int i0 = Step * i + k - Padding;

                            if (i0 < 0 || i0 >= ISize.Height)
                                continue;

                            for (int l = 0; l < FilterSize; l++)
                            {
                                int j0 = Step * j + l - Padding;

                                if (j0 < 0 || j0 >= ISize.Width)
                                    continue;
                                for (int c = 0; c < FilterDepth; c++)
                                {
                                    sum += (X[n](c, i0, j0)) * (Filters[f](c, k, l));
                                }
                            }
                        }

                        Output[n](f, i, j) = sum;
                    }
                }
            }
        }
    }

    void Backward(const std::vector<Tensor> &dout, const std::vector<Tensor> &X, bool calc_dX)
    {
        TSize size;
        size.Height = Step * (OSize.Height - 1) + 1;
        size.Width = Step * (OSize.Width - 1) + 1;
        size.Deep = OSize.Deep;

        std::vector<Tensor> deltas(dout.size(), Tensor(size));

        for (size_t n = 0; n < dout.size(); n++)
        {
            for (int d = 0; d < size.Deep; d++)
                for (int i = 0; i < OSize.Height; i++)
                    for (int j = 0; j < OSize.Width; j++)
                        deltas[n](d, i * Step, j * Step) = dout[n](d, i, j);
        }

#pragma omp parallel for
        for (int f = 0; f < FilterCount; f++)
        {
            for (size_t n = 0; n < dout.size(); n++)
            {
                for (int k = 0; k < size.Height; k++)
                {
                    for (int l = 0; l < size.Width; l++)
                    {
                        double delta = deltas[n](f, k, l);

                        for (int i = 0; i < FilterSize; i++)
                        {
                            int i0 = i + k - Padding;

                            if (i0 < 0 || i0 >= ISize.Height)
                                continue;

                            for (int j = 0; j < FilterSize; j++)
                            {
                                int j0 = j + l - Padding;

                                if (j0 < 0 || j0 >= ISize.Width)
                                    continue;

                                for (int c = 0; c < FilterDepth; c++)
                                    FiltersGradient[f](c, i, j) += delta * X[n](c, i0, j0);
                            }
                        }

                        BiasGradient[f] += delta;
                    }
                }
            }
        }

        if (calc_dX)
        {
            int pad = FilterSize - Padding - 1;

#pragma omp parallel for collapse(3)
            for (size_t n = 0; n < dout.size(); n++)
            {
                for (int i = 0; i < ISize.Height; i++)
                {
                    for (int j = 0; j < ISize.Width; j++)
                    {
                        for (int c = 0; c < FilterDepth; c++)
                        {
                            double sum = 0;

                            for (int k = 0; k < FilterSize; k++)
                            {
                                int i0 = i + k - pad;

                                if (i0 < 0 || i0 >= size.Height)
                                    continue;

                                for (int l = 0; l < FilterSize; l++)
                                {
                                    int j0 = j + l - pad;

                                    if (j0 < 0 || j0 >= size.Width)
                                        continue;

                                    for (int f = 0; f < FilterCount; f++)
                                        sum += Filters[f](c, FilterSize - 1 - k, FilterSize - 1 - l) * deltas[n](f, i0, j0);
                                }
                            }

                            Gradient[n](c, i, j) = sum;
                        }
                    }
                }
            }
        }
    }

    void Update(bool trainable)
    {
        int batchSize = Output.size();
        int total = FilterDepth * FilterSize * FilterSize;

#pragma omp parallel for
        for (int index = 0; index < FilterCount; index++)
        {
            for (int i = 0; i < total; i++)
            {
                if (trainable)
                    UpdateWeights(FiltersGradient[index][i] / batchSize,
                                  FilterParams[0][index][i], FilterParams[1][index][i],
                                  FilterParams[2][index][i], Filters[index][i]);

                FiltersGradient[index][i] = 0;
            }

            if (trainable)
                UpdateWeights(BiasGradient[index] / batchSize,
                              BiasParams[0][index], BiasParams[1][index], BiasParams[2][index], Bias[index]);

            BiasGradient[index] = 0;
        }
    }

    void Reset()
    {
        int total = FilterDepth * FilterSize * FilterSize;

        for (int index = 0; index < 3; index++)
        {
            for (int i = 0; i < FilterCount; i++)
            {
                for (int j = 0; j < total; j++)
                    FilterParams[index][i][j] = 0;

                BiasParams[index][i] = 0;
            }
        }
    }

    void SaveToFile(std::ofstream &File) const
    {
        File << "conv " << ISize.Width << ISize.Height << ISize.Deep << " ";
        File << FilterSize << " " << FilterCount << " " << Padding << " " << Step << std::endl;

        for (int index = 0; index < FilterCount; index++)
        {
            for (int d = 0; d < FilterDepth; d++)
                for (int i = 0; i < FilterSize; i++)
                    for (int j = 0; j < FilterSize; j++)
                        File << std::setprecision(15) << Filters[index](d, i, j) << " ";

            File << std::setprecision(15) << Bias[index] << std::endl;
        }
    }

    void SetBatchSize(int batchSize)
    {
        Output = std::vector<Tensor>(batchSize, Tensor(OSize));
        Gradient = std::vector<Tensor>(batchSize, Tensor(OSize));
    }

    void SetWeight(int index, int i, int j, int k, double weight)
    {
        Filters[index](i, j, k) = weight;
    }

    void SetBias(int index, double bias)
    {
        Bias[index] = bias;
    }

    void SetParam(int index, double weight)
    {
        int params = FilterSize * FilterSize * FilterDepth + 1;
        int findex = index / params;
        int windex = index % params;

        if (windex == params - 1)
        {
            Bias[findex] = weight;
        }
        else
        {
            Filters[findex][windex] = weight;
        }
    }

    double GetParam(int index) const
    {
        int params = FilterSize * FilterSize * FilterDepth + 1;
        int findex = index / params;
        int windex = index % params;

        if (windex == params - 1)
            return Bias[findex];

        return Filters[findex][windex];
    }

    double GetGradient(int index) const
    {
        int params = FilterSize * FilterSize * FilterDepth + 1;
        int findex = index / params;
        int windex = index % params;

        if (windex == params - 1)
            return BiasGradient[findex];

        return FiltersGradient[findex][windex];
    }

    void ZeroGradient(int index)
    {
        int params = FilterSize * FilterSize * FilterDepth + 1;
        int findex = index / params;
        int windex = index % params;

        if (windex == params - 1)
            BiasGradient[findex] = 0;
        else
            FiltersGradient[findex][windex] = 0;
    }
};

#endif
