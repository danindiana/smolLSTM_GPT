#include <iostream>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>

using namespace std;

class LSTM {
public:
    LSTM(int inputSize, int hiddenSize) {
        srand(time(0));
        this->inputSize = inputSize;
        this->hiddenSize = hiddenSize;

        initializeWeights();
    }

    vector<double> forward(const vector<double>& input, const vector<double>& prevState, const vector<double>& prevCell) {
        vector<double> gates(inputSize + hiddenSize, 0.0);
        for (size_t i = 0; i < input.size(); ++i) {
            gates[i] = input[i];
        }
        for (size_t i = 0; i < prevState.size(); ++i) {
            gates[inputSize + i] = prevState[i];
        }

        vector<double> f_gate(hiddenSize, 0.0), i_gate(hiddenSize, 0.0), o_gate(hiddenSize, 0.0), g_gate(hiddenSize, 0.0);
        for (int i = 0; i < hiddenSize; ++i) {
            for (int j = 0; j < inputSize + hiddenSize; ++j) {
                f_gate[i] += gates[j] * fWeights[j][i];
                i_gate[i] += gates[j] * iWeights[j][i];
                o_gate[i] += gates[j] * oWeights[j][i];
                g_gate[i] += gates[j] * gWeights[j][i];
            }
            f_gate[i] = sigmoid(f_gate[i]);
            i_gate[i] = sigmoid(i_gate[i]);
            o_gate[i] = sigmoid(o_gate[i]);
            g_gate[i] = tanh(g_gate[i]);
        }

        vector<double> cell(hiddenSize, 0.0), state(hiddenSize, 0.0);
        for (int i = 0; i < hiddenSize; ++i) {
            cell[i] = f_gate[i] * prevCell[i] + i_gate[i] * g_gate[i];
            state[i] = o_gate[i] * tanh(cell[i]);
        }

        return state;
    }

private:
    int inputSize, hiddenSize;
    vector<vector<double>> fWeights, iWeights, oWeights, gWeights;

    void initializeWeights() {
        fWeights.resize(inputSize + hiddenSize, vector<double>(hiddenSize));
        iWeights.resize(inputSize + hiddenSize, vector<double>(hiddenSize));
        oWeights.resize(inputSize + hiddenSize, vector<double>(hiddenSize));
        gWeights.resize(inputSize + hiddenSize, vector<double>(hiddenSize));

        for (int i = 0; i < inputSize + hiddenSize; ++i) {
            for (int j = 0; j < hiddenSize; ++j) {
                fWeights[i][j] = randomWeight();
                iWeights[i][j] = randomWeight();
                oWeights[i][j] = randomWeight();
                gWeights[i][j] = randomWeight();
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
    int hiddenSize = 4;

    LSTM lstm(inputSize, hiddenSize);

    vector<double> input = {0.5, 0.8};
    vector<double> prevState(hiddenSize, 
