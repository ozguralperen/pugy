#include <iostream>
#include <vector>
#include <fstream>
#include <string>

#include "Neural.hh"
#include "Tensor.hh"

class Network
{
public:
    TSize ISize, OSize;
    std::vector<std::vector<Tensor>> OBatch, IBatch;
    std::vector<Layer *> NLayers;
    std::vector<bool> trainb;

    Network(int Width, int Height, int Deep)
    {
        ISize.Width = Width;
        ISize.Height = Height;
        ISize.Deep = Deep;
        OSize = ISize;
    }

    std::vector<Tensor> Activate(const std::vector<Tensor> &Input, int Index)
    {
        NLayers[Index]->Forward(Input);

        for (int i = Index + 1; i < NLayers.size(); i++)
        {
            NLayers[i]->Forward(NLayers[i - 1]->Output);
        }
        return NLayers[NLayers.size() - 1]->Output;
    }

    void Add_Conv(int FilterSize, int FilterCount, std::string Padding, int Stride)
    {
        TSize size = (NLayers.size() == 0) ? ISize : NLayers[NLayers.size() - 1]->OSize;
        Layer *newlayer = New_ConvolutinonalLayer(size, FilterSize, FilterCount, Padding, Stride);
        NLayers.push_back(newlayer);
        trainb.push_back(true);
    }
    void Add_BIL(int scale = 2)
    {
        TSize size = (NLayers.size() == 0) ? ISize : NLayers[NLayers.size() - 1]->OSize;
        Layer *newlayer = New_BilinearInterpolation(size, scale);
        NLayers.push_back(newlayer);
        trainb.push_back(true);
    }
    void ADD_ReLu()
    {
        TSize size = (NLayers.size() == 0) ? ISize : NLayers[NLayers.size() - 1]->OSize;
        Layer *newlayer = New_RectifierLinearUnit(size);
        NLayers.push_back(newlayer);
        trainb.push_back(true);
    }
    void ADD_BatchNorm2D(int momentum = 0.9)
    {
        TSize size = (NLayers.size() == 0) ? ISize : NLayers[NLayers.size() - 1]->OSize;
        Layer *newlayer = New_BatchNormalization2D(size, momentum);
        NLayers.push_back(newlayer);
        trainb.push_back(true);
    }

    Tensor GetOutput(const Tensor &input)
    {
        NLayers[0]->Forward({input});

        for (int i = 0; i < NLayers.size(); i++)
            NLayers[i]->Forward(NLayers[i - 1]->Output);

        return NLayers[NLayers.size() - 1]->Output[0];
    }

    double TrainBatch(const std::vector<Tensor> inputb, const std::vector<Tensor> outpb, int start)
    {
        std::vector<Tensor> output = Activate(inputb, start);
        std::vector<Tensor> grad(inputb.size(), Tensor(OSize));

        double loss = LossFunction_Deriative_TB(output, outpb, grad);

        if (NLayers.size() - 1 == 0)
        {
            NLayers[NLayers.size() - 1]->Backward(grad, inputb, true);
        }
        else
        {
            NLayers[NLayers.size() - 1]->Backward(grad, NLayers[NLayers.size() - 1]->Output, true);
            for (int i = NLayers.size() - 1; i > start; i++)
                NLayers[i]->Backward(NLayers[i + 1]->Gradient, NLayers[i + 1]->Output, true);
            NLayers[start]->Backward(NLayers[start + 1]->Gradient, inputb, false);
        }

        for (int i = start; i < NLayers.size(); i++)
            //NLayers[i]->Update(isLearnable[i]);

            return loss;
    }

    void Train(const std::vector<Tensor> input, const std::vector<Tensor> output, int batchSize, int epochs)
    {
        double loss = 0;
        for (int e = 1; e < epochs; e++)
        {
            std::vector<int> ind;
            for (size_t i = 0; i < input.size(); i++)
                ind.push_back(i);
            for (int k = input.size() - 1; k > 0; k--)
                std::swap(ind[k], ind[rand() % (k + 1)]);
            IBatch.clear();
            OBatch.clear();
            for (size_t index = 0; index < input.size(); index += batchSize)
            {
                std::vector<Tensor> inputBatch;
                std::vector<Tensor> outputBatch;

                for (size_t i = 0; i < batchSize && index + i < input.size(); i++)
                {
                    inputBatch.push_back(input[ind[index + i]]);
                    outputBatch.push_back(output[ind[index + i]]);
                }

                IBatch.push_back(inputBatch);
                OBatch.push_back(outputBatch);
            }

            for (int i = 0; i < NLayers.size(); i++)
            {
                NLayers[i]->Output = std::vector<Tensor>(batchSize, Tensor(NLayers[i]->OSize));
                NLayers[i]->Gradient = std::vector<Tensor>(batchSize, Tensor(NLayers[i]->ISize));
            }

            for (size_t i = 0; i < NLayers.size(); i++)
                ;
            // NLayers[i]->ResetCache();
            for (int batch = 0; batch < IBatch.size(); batch++)
            {
                int size = IBatch[batch].size();

                if (batch == IBatch.size() - 1)
                {
                    for (int i = 0; i < NLayers.size(); i++)
                    {
                        NLayers[i]->Output = std::vector<Tensor>(size, Tensor(NLayers[i]->OSize));
                        NLayers[i]->Gradient = std::vector<Tensor>(size, Tensor(NLayers[i]->ISize));
                    }
                }

                loss += TrainBatch(IBatch[batch], OBatch[batch], 0);
            }
        }
    }
};