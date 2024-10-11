#include "DataLoader.h"
#include <iostream>

int main() {
    DataLoader dataLoader("./mnist/train-images-idx3-ubyte", "./mnist/train-labels-idx1-ubyte");
    dataLoader.load();

    // 获取加载的图片和标签
    auto images = dataLoader.getImages();
    auto labels = dataLoader.getLabels();

    // 打印第一张图片的第一个像素和标签
    if (!images.empty() && !labels.empty()) {
        std::cout << "First image, first pixel: " << images[0][0] << std::endl;
        std::cout << "First label: " << labels[0] << std::endl;
    }

    return 0;
}