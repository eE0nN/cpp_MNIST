#include "layer.h"
#include <random>
#include <cmath>

// 构造函数
FullyConnectedLayer::FullyConnectedLayer(int inputSize, int outputSize)
    : weights(outputSize, std::vector<double>(inputSize)),
      biases(outputSize, 0.0),
      input(inputSize, 0.0),
      output(outputSize, 0.0)
{
    initializeWeights();
}

//初始化权重和偏置（随机小值）
void FullyConnectedLayer::initializeWeights()
{
    std::random_device rd;
    std::mt19937 gen(rd());
    // Xavier 初始化: 标准差为 sqrt(1 / inputSize)
    std::normal_distribution<> dis(0.0, std::sqrt(1.0 / input.size()));
    for (auto& row : weights) {
        for (auto& w : row) {
            w = dis(gen);
        }
    }
}


//前向传播
std::vector<double> FullyConnectedLayer::forward(const std::vector<double> &input)
{
    this->input = input; // 将输入向量 input 存储在类的成员变量 input 中，以便在反向传播时使用
    output.assign(output.size(),0.0); //重置输出为0

    //计算每个输出节点的加权和
    for(size_t i = 0;i < weights.size();++i){
        for(size_t j = 0;j < input.size();++j){
            output[i] += weights[i][j] * input[j];
        }
        output[i] += biases[i];
    }

    //激活函数，这里是relu，可以改成其他的
    for(auto& o : output){
        o = std::max(0.0,o);
    }
    
    return output;
}

std::vector<double> FullyConnectedLayer::backward(const std::vector<double> &gradOutput, double learningRate)
{   
    //存储输入层的梯度
    std::vector<double> gradInput(input.size(),0.0);

    //计算输入层的梯度
    for(size_t i = 0; i < input.size();++i){
        for(size_t j = 0; j < weights.size();++j){
            gradInput[i] += gradOutput[j] * weights[j][i];
        }
    }

    // 计算权重的梯度并更新
    for (size_t i = 0; i < weights.size(); ++i) {
        for (size_t j = 0; j < input.size(); ++j) {
            weights[i][j] -= learningRate * gradOutput[i] * input[j];
        }
        biases[i] -= learningRate * gradOutput[i];
    }

    return gradInput;
}
