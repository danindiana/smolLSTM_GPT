use Random;
use Math;

class LSTM {
  var inputSize, hiddenSize: int;
  var fWeights, iWeights, oWeights, gWeights: [1..inputSize + hiddenSize, 1..hiddenSize] real;

  proc init(inputSize: int, hiddenSize: int) {
    this.inputSize = inputSize;
    this.hiddenSize = hiddenSize;
    initializeWeights();
  }

  proc initializeWeights() {
    var rng = new RandomStream(real);
    for i in 1..inputSize + hiddenSize {
      for j in 1..hiddenSize {
        fWeights[i, j] = rng.getNext() * 2 - 1;
        iWeights[i, j] = rng.getNext() * 2 - 1;
        oWeights[i, j] = rng.getNext() * 2 - 1;
        gWeights[i, j] = rng.getNext() * 2 - 1;
      }
    }
  }

  proc sigmoid(x: real): real {
    return 1 / (1 + exp(-x));
  }

  proc forward(input: [] real, prevState: [] real, prevCell: [] real): [1..hiddenSize] real {
    var gates: [1..inputSize + hiddenSize] real;
    gates[1..input.size] = input;
    gates[inputSize + 1..inputSize + hiddenSize] = prevState;

    var f_gate, i_gate, o_gate, g_gate: [1..hiddenSize] real;
    for i in 1..hiddenSize {
      for j in 1..inputSize + hiddenSize {
        f_gate[i] += gates[j] * fWeights[j, i];
        i_gate[i] += gates[j] * iWeights[j, i];
        o_gate[i] += gates[j] * oWeights[j, i];
        g_gate[i] += gates[j] * gWeights[j, i];
      }
      f_gate[i] = sigmoid(f_gate[i]);
      i_gate[i] = sigmoid(i_gate[i]);
      o_gate[i] = sigmoid(o_gate[i]);
      g_gate[i] = tan(g_gate[i]);
    }

    var cell, state: [1..hiddenSize] real;
    for i in 1..hiddenSize {
      cell[i] = f_gate[i] * prevCell[i] + i_gate[i] * g_gate[i];
      state[i] = o_gate[i] * tanh(cell[i]);
    }

    return state;
  }
}

proc main() {
  const inputSize = 2;
  const hiddenSize = 4;

  var lstm = new LSTM(inputSize, hiddenSize);

  var input: [1..inputSize] real = [0.5, 0.8];
  var prevState: [1..hiddenSize] real = [0.0, 0.0, 0.0, 0.0];
  var prevCell: [1..hiddenSize] real = [0.0, 0.0, 0.0, 0.0];

  var state = lstm.forward(input, prevState, prevCell);

  writeln("State: ", state);
}
