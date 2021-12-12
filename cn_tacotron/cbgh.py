import torch
from torch import nn



class BatchNormConv1d(nn.Module):
    r"""A wrapper for Conv1d with BatchNorm. It sets the activation
    function between Conv and BatchNorm layers. BatchNorm layer
    is initialized with the TF default values for momentum and eps.
    Args:
        in_channels: size of each input sample
        out_channels: size of each output samples
        kernel_size: kernel size of conv filters
        stride: stride of conv filters
        padding: padding of conv filters
        activation: activation function set b/w Conv1d and BatchNorm
    Shapes:
        - input: batch x dims
        - output: batch x dims
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 activation=None):

        super(BatchNormConv1d, self).__init__()
        self.padding = padding
        self.padder = nn.ConstantPad1d(padding, 0)
        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=False)
        # Following tensorflow's default parameters
        self.bn = nn.BatchNorm1d(out_channels, momentum=0.99, eps=1e-3)
        self.activation = activation
        # self.init_layers()

    def init_layers(self):
        if type(self.activation) == torch.nn.ReLU:
            w_gain = 'relu'
        elif type(self.activation) == torch.nn.Tanh:
            w_gain = 'tanh'
        elif self.activation is None:
            w_gain = 'linear'
        else:
            raise RuntimeError('Unknown activation function')
        torch.nn.init.xavier_uniform_(
            self.conv1d.weight, gain=torch.nn.init.calculate_gain(w_gain))

    def forward(self, x):
        x = self.padder(x)
        x = self.conv1d(x)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Highway(nn.Module):
    # TODO: Try GLU layer
    def __init__(self, in_size, out_size):
        super(Highway, self).__init__()
        self.H = nn.Linear(in_size, out_size)
        self.H.bias.data.zero_()
        self.T = nn.Linear(in_size, out_size)
        self.T.bias.data.fill_(-1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # self.init_layers()

    def init_layers(self):
        torch.nn.init.xavier_uniform_(
            self.H.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(
            self.T.weight, gain=torch.nn.init.calculate_gain('sigmoid'))

    def forward(self, inputs):
        H = self.relu(self.H(inputs))
        T = self.sigmoid(self.T(inputs))
        return H * T + inputs * (1.0 - T)


class CBHG(nn.Module):
    """CBHG module: a recurrent neural network composed of:
        - 1-d convolution banks
        - Highway networks + residual connections
        - Bidirectional gated recurrent units
        Args:
            in_features (int): sample size
            K (int): max filter size in conv bank
            projections (list): conv channel sizes for conv projections
            num_highways (int): number of highways layers
        Shapes:
            - input: B x D x T_in
            - output: B x T_in x D*2
    """

    def __init__(self,
                 in_features,
                 K=16,
                 conv_bank_features=128,
                 conv_projections=[128, 128],
                 highway_features=128,
                 gru_features=128,
                 num_highways=4):
        super(CBHG, self).__init__()
        self.in_features = in_features
        self.conv_bank_features = conv_bank_features
        self.highway_features = highway_features
        self.gru_features = gru_features
        self.conv_projections = conv_projections
        self.relu = nn.ReLU()
        # list of conv1d bank with filter size k=1...K
        # TODO: try dilational layers instead
        self.conv1d_banks = nn.ModuleList([
            BatchNormConv1d(in_features,
                            conv_bank_features,
                            kernel_size=k,
                            stride=1,
                            padding=[(k - 1) // 2, k // 2],
                            activation=self.relu) for k in range(1, K + 1)
        ])
        # max pooling of conv bank, with padding
        # TODO: try average pooling OR larger kernel size
        out_features = [K * conv_bank_features] + conv_projections[:-1]
        activations = [self.relu] * (len(conv_projections) - 1)
        activations += [None]
        # setup conv1d projection layers
        layer_set = []
        for (in_size, out_size, ac) in zip(out_features, conv_projections,
                                           activations):
            layer = BatchNormConv1d(in_size,
                                    out_size,
                                    kernel_size=3,
                                    stride=1,
                                    padding=[1, 1],
                                    activation=ac)
            layer_set.append(layer)
        self.conv1d_projections = nn.ModuleList(layer_set)
        # setup Highway layers
        if self.highway_features != conv_projections[-1]:
            self.pre_highway = nn.Linear(conv_projections[-1],
                                         highway_features,
                                         bias=False)
        self.highways = nn.ModuleList([
            Highway(highway_features, highway_features)
            for _ in range(num_highways)
        ])
        # bi-directional GPU layer
        self.gru = nn.GRU(gru_features,
                          gru_features,
                          1,
                          batch_first=True,
                          bidirectional=True)

    def forward(self, inputs):
        # (B, in_features, T_in)
        x = inputs
        # (B, hid_features*K, T_in)
        # Concat conv1d bank outputs
        outs = []
        for conv1d in self.conv1d_banks:
            out = conv1d(x)
            outs.append(out)
        x = torch.cat(outs, dim=1)
        assert x.size(1) == self.conv_bank_features * len(self.conv1d_banks)
        for conv1d in self.conv1d_projections:
            x = conv1d(x)
        x += inputs
        x = x.transpose(1, 2)
        if self.highway_features != self.conv_projections[-1]:
            x = self.pre_highway(x)
        # Residual connection
        # TODO: try residual scaling as in Deep Voice 3
        # TODO: try plain residual layers
        for highway in self.highways:
            x = highway(x)
        # (B, T_in, hid_features*2)
        # TODO: replace GRU with convolution as in Deep Voice 3
        self.gru.flatten_parameters()
        outputs, _ = self.gru(x)
        return outputs


if __name__ == '__main__':
    from cn_tacotron.text import label_to_sequence
    import numpy as np
    data = np.asanyarray(label_to_sequence("datasets/labels/000001.lab"))
    input_data = torch.LongTensor(data)
    model = CBHG(113)
    model(input_data)
