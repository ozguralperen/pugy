#include <iostream>
#include <fstream>
#include <iomanip>
#include <vector>
#include <random>

#include "Layer.hh"
#include "Optimization.hh"

class BatchNormalization2D : public Layer
{
    int wh;
    double momentum;

    Tensor gamma;
    Tensor dgamma;
    std::vector<Tensor> paramsgamma;

    Tensor beta;
    Tensor dbeta;
    std::vector<Tensor> paramsbeta;

    std::vector<Tensor> X_norm;
    std::vector<Tensor> dX_norm;

    Tensor mu, var;
    Tensor running_mu, running_var;

public:
    BatchNormalization2D(std::ifstream &File, TSize size, double _momentum) : Layer(size),
                                                                              gamma(1, 1, size.Deep), dgamma(1, 1, size.Deep),
                                                                              beta(1, 1, size.Deep), dbeta(1, 1, size.Deep),
                                                                              mu(1, 1, size.Deep), var(1, 1, size.Deep),
                                                                              running_mu(1, 1, size.Deep), running_var(1, 1, size.Deep),
                                                                              momentum(_momentum)
    {
        LoadWeights(File);
        InitWeights();
    }
    BatchNormalization2D(TSize size, double _momentum) : Layer(size),
                                                         gamma(1, 1, size.Deep), dgamma(1, 1, size.Deep),
                                                         beta(1, 1, size.Deep), dbeta(1, 1, size.Deep),
                                                         mu(1, 1, size.Deep), var(1, 1, size.Deep),
                                                         running_mu(1, 1, size.Deep), running_var(1, 1, size.Deep),
                                                         momentum(_momentum)
    {

        InitWeights();
    }

    void Init()
    {
        for (int i = 0; i < 3; i++)
        {
            paramsgamma.push_back(Tensor(1, 1, OSize.Deep));
            paramsbeta.push_back(Tensor(1, 1, OSize.Deep));
        }
    }

    void InitWeights()
    {
        for (int i = 0; i < OSize.Deep; i++)
        {
            gamma[i] = 1;
            beta[i] = 0;

            running_mu[i] = 0;
            running_var[i] = 0;
        }
    }
    void LoadWeights(std::ifstream &f)
    {
        for (int i = 0; i < OSize.Deep; i++)
            f >> gamma[i];

        for (int i = 0; i < OSize.Deep; i++)
            f >> beta[i];

        for (int i = 0; i < OSize.Deep; i++)
            f >> running_mu[i];

        for (int i = 0; i < OSize.Deep; i++)
            f >> running_var[i];
    }
    void ForwardOutput(const std::vector<Tensor> &X)
    {
#pragma omp parallel for collapse(4)
        for (size_t batchIndex = 0; batchIndex < X.size(); batchIndex++)
            for (int d = 0; d < OSize.Deep; d++)
                for (int i = 0; i < OSize.Height; i++)
                    for (int j = 0; j < OSize.Width; j++)
                        Output[batchIndex](d, i, j) = gamma[d] * (X[batchIndex](d, i, j) - running_mu[d]) / sqrt(running_var[d] + 1e-8) + beta[d];
    }

    void Forward(const std::vector<Tensor> &X)
    {
#pragma omp parallel for
        for (int d = 0; d < OSize.Deep; d++)
        {
            double sum = 0;

            for (size_t batchIndex = 0; batchIndex < X.size(); batchIndex++)
                for (int i = 0; i < OSize.Height; i++)
                    for (int j = 0; j < OSize.Width; j++)
                        sum += X[batchIndex](d, i, j);

            mu[d] = sum / (X.size() * wh);
            sum = 0;

            for (size_t batchIndex = 0; batchIndex < X.size(); batchIndex++)
                for (int i = 0; i < OSize.Height; i++)
                    for (int j = 0; j < OSize.Width; j++)
                        sum += pow(X[batchIndex](d, i, j) - mu[d], 2);

            var[d] = sum / (X.size() * wh);

            for (size_t batchIndex = 0; batchIndex < X.size(); batchIndex++)
            {
                for (int i = 0; i < OSize.Height; i++)
                {
                    for (int j = 0; j < OSize.Width; j++)
                    {
                        X_norm[batchIndex](d, i, j) = (X[batchIndex](d, i, j) - mu[d]) / sqrt(var[d] + 1e-8);
                        Output[batchIndex](d, i, j) = gamma[d] * X_norm[batchIndex](d, i, j) + beta[d];
                    }
                }
            }

            running_mu[d] = momentum * running_mu[d] + (1 - momentum) * mu[d];
            running_var[d] = momentum * running_var[d] + (1 - momentum) * var[d];
        }
    }

    void Backward(const std::vector<Tensor> &dout, const std::vector<Tensor> &X, bool calc_dX)
    {
        for (size_t batchIndex = 0; batchIndex < dout.size(); batchIndex++)
        {
            for (int d = 0; d < OSize.Deep; d++)
            {
                for (int i = 0; i < OSize.Height; i++)
                {
                    for (int j = 0; j < OSize.Width; j++)
                    {
                        double delta = dout[batchIndex](d, i, j);

                        dgamma[d] += delta * X_norm[batchIndex](d, i, j);
                        dbeta[d] += delta;
                    }
                }
            }
        }

        if (calc_dX)
        {
            size_t N = dout.size();

            Tensor d1(1, 1, OSize.Deep);
            Tensor d2(1, 1, OSize.Deep);

            for (size_t batchIndex = 0; batchIndex < N; batchIndex++)
            {
                for (int d = 0; d < OSize.Deep; d++)
                {
                    for (int i = 0; i < OSize.Height; i++)
                    {
                        for (int j = 0; j < OSize.Width; j++)
                        {
                            dX_norm[batchIndex](d, i, j) = dout[batchIndex](d, i, j) * gamma[d];

                            d1[d] += dX_norm[batchIndex](d, i, j);
                            d2[d] += dX_norm[batchIndex](d, i, j) * X_norm[batchIndex](d, i, j);
                        }
                    }
                }
            }

#pragma omp parallel for collapse(4)
            for (int d = 0; d < OSize.Deep; d++)
                for (int i = 0; i < OSize.Height; i++)
                    for (int j = 0; j < OSize.Width; j++)
                        for (size_t batchIndex = 0; batchIndex < N; batchIndex++)
                            Gradient[batchIndex](d, i, j) = (dX_norm[batchIndex](d, i, j) - (d1[d] + X_norm[batchIndex](d, i, j) * d2[d]) / (N * wh)) / (sqrt(var[d] + 1e-8));
        }
    }
    void ResetCache()
    {
        for (int i = 0; i < OSize.Deep; i++)
        {
            running_mu[i] = 0;
            running_var[i] = 0;

            for (int j = 0; j < 3; j++)
            {
                paramsgamma[j][i] = 0;
                paramsbeta[j][i] = 0;
            }
        }
    }
    void Save(std::ofstream &f) const
    {
        f << "batchnormalization2D " << ISize.Width << ISize.Height << ISize.Deep << " " << momentum << std::endl;

        for (int i = 0; i < OSize.Deep; i++)
            f << std::setprecision(15) << gamma[i] << " ";

        f << std::endl;

        for (int i = 0; i < OSize.Deep; i++)
            f << std::setprecision(15) << beta[i] << " ";

        f << std::endl;

        for (int i = 0; i < OSize.Deep; i++)
            f << std::setprecision(15) << running_mu[i] << " ";

        f << std::endl;

        for (int i = 0; i < OSize.Deep; i++)
            f << std::setprecision(15) << running_var[i] << " ";

        f << std::endl;
    }
    void SetBatchSize(int batchSize)
    {
        Output = std::vector<Tensor>(batchSize, Tensor(OSize));
        Gradient = std::vector<Tensor>(batchSize, Tensor(OSize));

        X_norm = std::vector<Tensor>(batchSize, Tensor(ISize));
        dX_norm = std::vector<Tensor>(batchSize, Tensor(ISize));
    }

    void SetParam(int index, double weight)
    {
        if (index / OSize.Deep == 0)
        {
            gamma[index] = weight;
        }
        else
        {
            beta[index % OSize.Deep] = weight;
        }
    }

    float GetParam(int index)
    {
        if (index / OSize.Deep == 0)
        {
            return gamma[index];
        }
        else
        {
            return beta[index % OSize.Deep];
        }
    }

    float GetGradient(int index) const
    {
        if (index / OSize.Deep == 0)
        {
            return dgamma[index];
        }
        else
        {
            return dbeta[index % OSize.Deep];
        }
    }
    float ZeroGradient(int index)
    {
        if (index / OSize.Deep == 0)
        {
            dgamma[index] = 0;
        }
        else
        {
            dbeta[index % OSize.Deep] = 0;
        }
    }
};