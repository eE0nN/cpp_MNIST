#ifndef DATALOADER_H
#define DATALOADER_H

#include <vector>
#include <string>

class DataLoader
{
public:
    DataLoader(const std::string &imagesPath, const std::string &labelsPath);
    void load();
    std::vector<std::vector<double>> getImages() const;
    std::vector<int> getLabels() const;

private:
    std::string imagesPath;
    std::string labelsPath;
    std::vector<std::vector<double>> images;
    std::vector<int> labels;

    void loadImages();
    void loadLabels();
};

#endif