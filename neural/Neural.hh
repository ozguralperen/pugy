#include "Tensor.hh"
#include "Layer.hh"
#include "BatchNorm2D.hh"
#include "ConvLayer.hh"
#include "Optimization.hh"
#include "LossFunction.hh"
#include "ReLU.hpp"
#include "BInterpolation.hh"


enum Layers
{
	conv,
	relu,
	bn,
    convTrans,
    bilinear
};

Layer* New_ConvolutinonalLayer(TSize size , int FilterSize , int FilterCount ,std::string Padding ,int Stride){
    int padding;
    if(Padding == std::string("same"))
        padding = (FilterSize - 1) /2 ;
    else if(Padding == std::string("valid"))
        padding = 0;
    else if(Padding == std::string("full"))
        padding = FilterSize - 1 ;
    else
    {
        padding = std::atoi(Padding.data());
    }

    return new ConvolutionalLayer(size,  FilterCount , FilterSize , padding , Stride);
}

Layer* Load_ConvolutinonalLayer(TSize size , std::ifstream &File){
    int FilterSize , FilterCount ,Padding, Stride;
    File >> FilterSize >> FilterCount >> Padding >> Stride;

    return new ConvolutionalLayer(size,  FilterCount , FilterSize , Padding , Stride);
}

Layer* New_BatchNormalization2D(TSize size , double momentum = 0.9){
        return new  BatchNormalization2D(size , momentum);
}
Layer* Load_BatchNormalization2D(TSize size , std::ifstream &File){
        double momentum;
        File >> momentum;
        return new  BatchNormalization2D(size , momentum);
}


Layer* New_RectifierLinearUnit (TSize size){
    return new RectifierLinearUnit(size);
}


Layer* New_BilinearInterpolation(TSize size , int scale = 2){
    return new  BilinearInterpolation(size , scale);
}