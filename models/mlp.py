import torch.nn as nn
import torch


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=(128, 128), activation='tanh', is_bn=False, is_dropout=False):
        super().__init__()
        if activation == 'tanh':
            self.activation = torch.tanh
        elif activation == 'relu':
            self.activation = torch.relu
        elif activation == 'sigmoid':
            self.activation = torch.sigmoid
        self.is_dropout= is_dropout
        if is_dropout:
            self.do = torch.nn.Dropout(0.1)

        self.out_dim = hidden_dims[-1]
        self.affine_layers = nn.ModuleList()
        last_dim = input_dim
        for nh in hidden_dims:
            if is_bn:
                self.affine_layers.append(nn.Sequential(nn.Linear(last_dim, nh),nn.BatchNorm1d(nh)))
            else:
                self.affine_layers.append(nn.Linear(last_dim, nh))
            last_dim = nh

    def forward(self, x):
        for affine in self.affine_layers:
            x = self.activation(affine(x))
            if self.is_dropout:
                x = self.do(x)
        return x
