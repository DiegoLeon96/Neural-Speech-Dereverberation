import torch.nn as nn

#################################
# MLP neural network
# adaptive implementation
#################################
class DNN(nn.Module):
    """
    Adaptive implementation of MLP for speech dereverberation
    """

    def __init__(self, neural_layers, act_fun):
        """
        neural_layers: list with number of units in each layer
        act_fun: list with activation functions

        Example: neural network with 128 input units, 1000 hidden units and 1 output unit,
                 ReLU activation function and Identity output activation function

                 net = DNN([128, 1000, 1], [nn.ReLU(), nn.Identity()])
        """
        super(DNN, self).__init__()
        self.layers = nn.Sequential()

        if len(neural_layers) < 2:
            print('len(neural_layes) must be higher than 2')
            return
        for i in range(len(neural_layers) - 1):
            self.layers.add_module('layer_{}'.format(i + 1),
                                   nn.Linear(neural_layers[i],
                                             neural_layers[i + 1])
                                   )
            self.layers.add_module('act_fun_{}'.format(i + 1),
                                   act_fun[i])

    def forward(self, x):
        x = self.layers(x)
        return x

#################################
# LSTM RNN for speech dereverberation
#################################
class LSTMDNN(nn.Module):
    """
    LSTM based dereverberation
    """

    def __init__(self):
        super(LSTMDNN, self).__init__()
        self.lstm1 = nn.LSTM(128 * 11, 512)
        self.lstm2 = nn.LSTM(512, 512)
        self.linear = nn.Linear(512, 128)

    def forward(self, x):
        x, _ = self.lstm1(x.view(x.shape[0], 1, x.shape[1]))
        x, _ = self.lstm2(x)
        x = self.linear(x[:, 0, :])
        return x

class SupLSTM(nn.Module):
    """
    Late supression dereverberation using LSTM
    """

    def __init__(self):
        super(SupLSTM, self).__init__()
        self.lstm1 = nn.LSTM(128, 512)
        self.lstm2 = nn.LSTM(512, 512)
        self.linear = nn.Linear(512, 128)

    def forward(self, input):
        x, _ = self.lstm1(input.view(input.shape[0], 1, input.shape[1]))
        x, _ = self.lstm2(x)
        x = self.linear(x[:, 0, :])
        output = input - x
        return output