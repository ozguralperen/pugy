#ifndef TENSOR
#define TENSOR

#include <iostream>
#include <vector>
#include <string>
#include <iomanip>

struct TSize
{

    int Deep;
    int Height;
    int Width;

public:
    bool operator==(TSize size)
    {
        return Deep == size.Deep && Height == size.Height && Width == size.Width;
    }
    bool operator!=(TSize size)
    {
        return Deep != size.Deep || Height != size.Height || Width != size.Width;
    }
    TSize(){};
    TSize(int _d, int _h, int _w) : Deep(_d), Height(_h), Width(_w){};
};

class Tensor
{
public:
    std::vector<double> Data;
    TSize size;
    Tensor(int Width, int Height, int Deep)
    {
        size = TSize(Deep, Height, Width);
        Data = std::vector<double>(Deep * Height * Width, 0);
    }

    Tensor(TSize Size)
    {
        size = TSize(Size.Deep , Size.Height , Size.Width);
        Data =  std::vector<double>(Size.Width *Size.Height * Size.Deep , 0);
    }

    double operator()(int x, int y, int z) const
    {
        return Data[z * size.Deep * size.Width + y * size.Deep + x];
    }
    double &operator()(int x, int y, int z)
    {
        return Data[z * size.Deep * size.Width + y * size.Deep + x];
    }
    double operator[](int i) const
    {
        return Data[i];
    }
    double &operator[](int i)
    {
        return Data[i];
    }

    double Min()
    {
        double minimum = Data[0];
        for (int i = 1; i < Data.size(); i++)
            minimum = (Data[i] < minimum) ? Data[i] : minimum;
        return minimum;
    }
    double Max()
    {
        double maximum = Data[0];
        for (int i = 1; i < Data.size(); i++)
            maximum = (Data[i] < maximum) ? Data[i] : maximum;
        return maximum;
    }
    double Mean()
    {
        double sum;
        for (int i = 0; i < Data.size(); i++)
            sum += Data[i];

        return sum / Data.size();
    }
    double StdDev()
    {
        double avg = Mean();
        double stddev = 0;

        for (size_t i = 0; i < Data.size(); i++)
            stddev += (Data[i] - avg) * (Data[i] - avg);

        return stddev / Data.size();
    }

    void ReDefine(int width, int height, int deep)
    {
        if (width * height * deep != size.Width * size.Height * size.Deep)
        {
            printf("Error < Redefining Tensor > : Size");
            exit(EXIT_FAILURE);
        }
        TSize NewSize = TSize(deep, height, width);
        size = NewSize;
    }
};

#endif