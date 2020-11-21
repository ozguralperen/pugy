#include "Tensor.h"

float LossFunction_Deriative(const Tensor &y, const Tensor &t, Tensor der)
{
    double loss = 0;
    for (int i = 0; i < (der.size.Width * der.size.Height * der.size.Deep); i++)
    {
        double e = y[i] - t[i];
        der[i] = 2 * e;
        loss += e * e;
    }
    return loss;
}
float LossFunction_Stat(const Tensor &y, const Tensor &t)
{
    double loss = 0;
    for (int i = 0; i < (y.size.Width * y.size.Height * y.size.Deep); i++)
    {
        double e = y[i] - t[i];
        loss += e * e;
    }
    return loss;
}

float LossFunction_Deriative_TB(const std::vector<Tensor> &y, const std::vector<Tensor> &t, std::vector<Tensor> der)
{
    double loss = 0;
    for (int i = 0; i < der.size() ; i++)
    {
        loss += LossFunction_Deriative(y[i] , t[i] , der[i]);
    }
    return loss;
}

float LossFunction_Stat_TB(const std::vector<Tensor> &y, const std::vector<Tensor> &t)
{
    double loss = 0;
    for (int i = 0; i < y.size(); i++)
    {
        LossFunction_Stat(y[i] , t[i])
    }
    return loss;
}