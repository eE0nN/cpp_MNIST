#ifndef ACTIVATION_H
#define ACTIVATION_H

#include <vector>
#include <cmath>

namespace Activation {

    inline double sigmoid(double x) {
        return 1.0 / (1.0 + exp(-x));
    }

    inline double relu(double x) {
        return std::max(0.0, x);
    }

    inline std::vector<double> relu(const std::vector<double>& input) {
        std::vector<double> output = input;
        for (auto& val : output) {
            val = relu(val);
        }
        return output;
    }
    
    inline std::vector<double> sigmoid(const std::vector<double>& input) {
        std::vector<double> output = input;
        for (auto& val : output) {
            val = sigmoid(val);
        }
        return output;
    }

}


#endif