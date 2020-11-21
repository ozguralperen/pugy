#ifndef LAYER
#define LAYER

#include <iostream>
#include <iomanip>
#include <vector>
#include "Tensor.hh"

class Layer
{
    public:
    TSize ISize;
    TSize OSize;
    std::vector<Tensor> Output;
    std::vector<Tensor> Gradient;
    Layer(TSize inputSize, int outputWidth, int outputHeight, int outputDeep) 
    {
        ISize = TSize(inputSize);
        OSize = TSize(outputWidth , outputHeight , outputDeep);
    }
    Layer(Tensor inputSize, Tensor outputSize) 
    {
        ISize = TSize(inputSize.size);
        OSize = TSize(outputSize.size);
    }
    Layer(TSize Size){
        ISize = TSize(Size);
        OSize = TSize(Size);
    }

    virtual void Forward(const std::vector<Tensor> &X);
	virtual void Backward(const std::vector<Tensor> &dout, const std::vector<Tensor> &X, bool calc_dX);
    
};

#endif