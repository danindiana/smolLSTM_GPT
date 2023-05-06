#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

class NeuralNetwork {
public:
    NeuralNetwork(int inputSize, int hiddenSize, int outputSize) {
        srand(time(0));
        initializeWeights(inputSize, hiddenSize, outputSize);
    }

    vector<double> forward(const vector<double>& input) {
        vector<double> hiddenLayer(hiddenSize);

        for (int i = 0; i < hiddenSize; ++i) {
            double activation = 0.0;
            for (int j = 0; j < inputSize; ++j) {
                activation += input[j] * inputHiddenWeights[j][i];
            }
            hiddenLayer[i] = sigmoid(activation);
        }

        vector<double> outputLayer(outputSize);

        for (int i = 0; i < outputSize; ++i) {
            double activation = 0.0;
            for (int j = 0; j < hiddenSize; ++j) {
                activation += hiddenLayer[j] * hiddenOutputWeights[j][i];
            }
            outputLayer[i] = sigmoid(activation);
        }

        return outputLayer;
    }

private:
    int inputSize, hiddenSize, outputSize;
    vector<vector<double>> inputHiddenWeights;
    vector<vector<double>> hiddenOutputWeights;

    void initializeWeights(int inputSize, int hiddenSize, int outputSize) {
        this->inputSize = inputSize;
        this->hiddenSize = hiddenSize;
        this->outputSize = outputSize;

        inputHiddenWeights.resize(inputSize, vector<double>(hiddenSize));
        hiddenOutputWeights.resize(hiddenSize, vector<double>(outputSize));

        for (int i = 0; i < inputSize; ++i) {
            for (int j = 0; j < hiddenSize; ++j) {
                inputHiddenWeights[i][j] = randomWeight();
            }
        }

        for (int i = 0; i < hiddenSize; ++i) {
            for (int j = 0; j < outputSize; ++j) {
                hiddenOutputWeights[i][j] = randomWeight();
            }
        }
    }

    double randomWeight() {
        return ((double)rand() / (RAND_MAX)) * 2 - 1;
    }

    double sigmoid(double x) {
        return 1 / (1 + exp(-x));
    }
};

int main() {
    int inputSize = 2;
    int hiddenSize = 3;
    int outputSize = 1;

    NeuralNetwork nn(inputSize, hiddenSize, outputSize);
    
    vector<double> input = {0.5, 0.8};
    vector<double> output = nn.forward(input);

    cout << "Output: " << output[0] << endl;

    return 0;
}
