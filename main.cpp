#include "neuralnetwork.h"
#include "dataloader.h"
#include <iostream>
#include <algorithm>
#include <vector>

int main() {
    // 创建 DataLoader 对象，加载 MNIST 数据集
    DataLoader dataLoader("./mnist/train-images-idx3-ubyte", "./mnist/train-labels-idx1-ubyte");
    dataLoader.load();

    // 获取图像和标签
    auto images = dataLoader.getImages();
    auto labels = dataLoader.getLabels();

    // 检查数据加载是否成功
    if (images.empty() || labels.empty()) {
        std::cerr << "Failed to load MNIST data!" << std::endl;
        return -1;
    }

    // 创建神经网络
    NeuralNetwork nn;

 // 添加层：输入层大小为 784（28x28），隐藏层大小为 128，输出层大小为 10
    // 添加多层神经网络
    nn.addLayer(new FullyConnectedLayer(784, 256));  // 输入层到第一个隐藏层（256 个节点）
    nn.addLayer(new FullyConnectedLayer(256, 128));  // 第一个隐藏层到第二个隐藏层（128 个节点）
    nn.addLayer(new FullyConnectedLayer(128, 64));   // 第二个隐藏层到第三个隐藏层（64 个节点）
    nn.addLayer(new FullyConnectedLayer(64, 10));    // 第三层到输出层（10 个节点）

    // 训练参数
    int epochs = 10;                // 训练轮数
    double learningRate = 0.0005;    // 学习率

    // 开始训练
    nn.train(images, labels, epochs, learningRate);

    // 测试部分
    std::vector<double> testInput = images[0];  // 测试第一张图片
    auto output = nn.forward(testInput);        // 前向传播得到输出

    // 打印输出
    std::cout << "Test output: ";
    for (const auto& val : output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    // 获取预测结果（输出中最大值的索引即为预测的类别）
    int predictedLabel = std::distance(output.begin(), std::max_element(output.begin(), output.end()));
    std::cout << "Predicted label: " << predictedLabel << ", Actual label: " << labels[0] << std::endl;

    return 0;
}
