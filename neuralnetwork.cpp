#include "neuralnetwork.h"
#include <iostream>
#include <cmath>

// 构造函数
NeuralNetwork::NeuralNetwork() {}

// 添加层
void NeuralNetwork::addLayer(FullyConnectedLayer* layer) {
    layers.push_back(layer);
}

// 前向传播
std::vector<double> NeuralNetwork::forward(const std::vector<double>& input) {
    std::vector<double> output = input;
    for (auto layer : layers) {
        output = layer->forward(output);
    }
    return output;
}
double crossEntropyLoss(const std::vector<double>& predicted, int actualLabel) {
    double loss = 0.0;
    for (size_t i = 0; i < predicted.size(); ++i) {
        double target = (i == actualLabel) ? 1.0 : 0.0;
        loss += -target * std::log(predicted[i] + 1e-9);  // 防止 log(0) 的情况
    }
    return loss;
}

// 训练网络
void NeuralNetwork::train(const std::vector<std::vector<double>>& images, const std::vector<int>& labels, int epochs, double learningRate) {
    double totalLoss = 0.0;
    for (size_t i = 0; i < images.size(); ++i) {
        // 前向传播
        std::vector<double> output = forward(images[i]);

        // 计算交叉熵损失
        double loss = crossEntropyLoss(output, labels[i]);
        totalLoss += loss;

        // 计算损失的梯度
        std::vector<double> gradOutput(output.size());
        for (size_t j = 0; j < output.size(); ++j) {
            double target = (j == labels[i]) ? 1.0 : 0.0;
            gradOutput[j] = output[j] - target;  // 使用 softmax 激活函数后的导数
        }

        // 反向传播
        for (auto it = layers.rbegin(); it != layers.rend(); ++it) {
            gradOutput = (*it)->backward(gradOutput, learningRate);
        }
}

}