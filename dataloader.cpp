#include "dataloader.h"
#include <fstream>
#include <iostream>
#include <vector>

// 构造函数，初始化图像和标签文件路径
DataLoader::DataLoader(const std::string &imagesPath, const std::string &labelsPath) : imagesPath(imagesPath), labelsPath(labelsPath) {}

// 加载图像和标签数据
void DataLoader::load()
{
    loadImages();
    loadLabels();
}

void DataLoader::loadImages()
{
    std::ifstream file(imagesPath, std::ios::binary); // 以二进制模式打开图像文件,搜一下具体含义 不懂
    if (!file.is_open())
    {
        std::cerr << "Unable to open image file: " << imagesPath << std::endl; // 如果文件无法打开，打印错误信息
        return;
    }
    // 读取文件头信息，包括魔数、图像数量、行和列信息
    int32_t magicNumber = 0;
    int32_t numImages = 0;
    int32_t numRows = 0;
    int32_t numCols = 0;

    // 读取 4 字节的魔数，并转换为大端序
    file.read(reinterpret_cast<char *>(&magicNumber), sizeof(magicNumber));
    magicNumber = __builtin_bswap32(magicNumber); // MNIST 文件存储为大端格式，需要转换

    // 读取 4 字节的图像数量，并转换为大端序
    file.read(reinterpret_cast<char *>(&numImages), sizeof(numImages));
    numImages = __builtin_bswap32(numImages);

    // 读取 4 字节的行数，并转换为大端序
    file.read(reinterpret_cast<char *>(&numRows), sizeof(numRows));
    numRows = __builtin_bswap32(numRows);

    // 读取 4 字节的列数，并转换为大端序
    file.read(reinterpret_cast<char *>(&numCols), sizeof(numCols));
    numCols = __builtin_bswap32(numCols);

    // 验证文件头信息是否正确
    if (magicNumber != 2051)
    {
        std::cerr << "Invalid MNIST image file!" << std::endl; // 如果魔数不对，说明不是有效的 MNIST 图像文件
        return;
    }

    // 分配空间存储所有图像数据，大小为 numImages x (numRows * numCols)
    images.resize(numImages, std::vector<double>(numRows * numCols));

    // 读取每一张图像的数据
    for (int i = 0; i < numImages; ++i)
    {
        for (int j = 0; j < numRows * numCols; ++j)
        {
            unsigned char pixel = 0; // MNIST 数据以字节形式存储
            file.read(reinterpret_cast<char *>(&pixel), sizeof(pixel));
            images[i][j] = static_cast<double>(pixel) / 255.0; // 将像素值归一化到 [0, 1] 之间
        }
    }
    file.close(); // 关闭文件
}
// 加载标签数据
void DataLoader::loadLabels()
{
    std::ifstream file(labelsPath, std::ios::binary); // 以二进制模式打开标签文件
    if (!file.is_open())
    {
        std::cerr << "Unable to open label file: " << labelsPath << std::endl; // 如果文件无法打开，打印错误信息
        return;
    }

    // 读取文件头信息，包括魔数和标签数量
    int32_t magicNumber = 0;
    int32_t numLabels = 0;

    // 读取 4 字节的魔数，并转换为大端序
    file.read(reinterpret_cast<char *>(&magicNumber), sizeof(magicNumber));
    magicNumber = __builtin_bswap32(magicNumber); // MNIST 文件存储为大端格式，需要转换

    // 读取 4 字节的标签数量，并转换为大端序
    file.read(reinterpret_cast<char *>(&numLabels), sizeof(numLabels));
    numLabels = __builtin_bswap32(numLabels);

    // 验证文件头信息是否正确
    if (magicNumber != 2049)
    {
        std::cerr << "Invalid MNIST label file!" << std::endl; // 如果魔数不对，说明不是有效的 MNIST 标签文件
        return;
    }

    // 分配空间存储所有标签
    labels.resize(numLabels);

    // 读取每一个标签
    for (int i = 0; i < numLabels; ++i)
    {
        unsigned char label = 0; // MNIST 标签以字节形式存储
        file.read(reinterpret_cast<char *>(&label), sizeof(label));
        labels[i] = static_cast<int>(label); // 将标签值转换为 int 类型
    }

    file.close(); // 关闭文件
}

// 返回已加载的图像数据
std::vector<std::vector<double>> DataLoader::getImages() const
{
    return images;
}

// 返回已加载的标签数据
std::vector<int> DataLoader::getLabels() const
{
    return labels;
}