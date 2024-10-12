#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

#include "layer.h"
#include <vector>

class NeuralNetwork {
public:
    NeuralNetwork();
    void addLayer(FullyConnectedLayer* layer);
    std::vector<double> forward(const std::vector<double>& input);
    void train(const std::vector<std::vector<double>>& images, const std::vector<int>& labels, int epochs, double learningRate);

private:
    std::vector<FullyConnectedLayer*> layers; // 存储网络的各层
};

#endif
