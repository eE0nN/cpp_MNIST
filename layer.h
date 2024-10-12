#ifndef LAYER_H
#define LAYER_H

#include <vector>

class FullyConnectedLayer
{
public:
    FullyConnectedLayer(int inputSize, int outputSize);
    std::vector<double> forward(const std::vector<double> &input);
    std::vector<double> backward(const std::vector<double> &gradOutput, double learningRate);
    void updateWeights(double learningRate);

private:
    std::vector<std::vector<double>> weights; // 权重矩阵
    std::vector<double> biases;               // 偏置
    std::vector<double> input;                // 存储前向传播的输入
    std::vector<double> output;               // 存储前向传播的输出

    void initializeWeights(); // 初始化权重和偏置
};

#endif