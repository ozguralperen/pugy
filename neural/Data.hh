#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

#include "Tensor.hh"
#include "Layer.hh"

class Data
{
    std::vector<Tensor> TROutput;
    std::vector<Tensor> TRInput;
    std::vector<std::string> Labels;
    TSize TRISize;
    double scale;

    std::vector<std::string> SimpleParser(const std::string MainString)
    {
        std::vector<std::string> Parsed;
        std::stringstream ss(MainString);
        while (ss.good())
        {
            std::string substr;
            getline(ss, substr, ',');
            Parsed.push_back(substr);
        }
        return Parsed;
    }

    Tensor Load_Tensors(const std::vector<std::string> &data, int start)
    {
        Tensor Input = Tensor(TRISize);

        if (data.size() != TRISize.Width * TRISize.Height * TRISize.Width + start)
        {
            std::cout << "Error < File , Generating Tensors > : Data size does not match. ";
            exit(EXIT_FAILURE);
        }

        for (int i = 0; i < TRISize.Height; i++)
            for (int j = 0; j < TRISize.Width; j++)
                for (int k = 0; k < TRISize.Deep; k++)
                    Input(i, j, k) = std::stod(data[start++]) / scale;
    }
    void Load_Data(const char *FilePath)
    {
        std::ifstream file(FilePath);
        if (!file)
        {
            std::cout << "Error < File , Reading > : An error occured in file io.";
            exit(EXIT_FAILURE);
        }

        Labels.clear();
        std::string line;
        while (std::getline(file, line))
            Labels.push_back(line);
        file.close();

        std::cout << "[OK] Datas load from file." << std::endl;
    }
    void Load_Train(const char *FilePath, int size)
    {
        std::ifstream file(FilePath);
        if (!file)
        {
            std::cout << "Error < File , Reading > : An error occured in file io.";
            exit(EXIT_FAILURE);
        }

        TROutput.clear();
        TRInput.clear();

        std::string line;
        while (std::getline(file, line))
        {
            std::vector<std::string> data = SimpleParser(line);
            int labelNu;

            for (int k = 0; k < Labels.size(); k++)
                labelNu = (Labels[k] == data[0]) ? k : -1;
            if (labelNu == -1)
            {
                std::cout << "Error < File , Reading > : L not recognized. ";
                exit(EXIT_FAILURE);
            }
            TRInput.push_back(Load_Tensors(data, 1));

            Tensor output(1, 1, Labels.size());

            for (int i = 0; i < Labels.size(); i++)
                output(i, 0, 0) = (labelNu == i) ? 1 : 0;

            TROutput.push_back(output);

            std::cout << "[OK] Loaded Train Samples" << std::endl;
        }
    }
};