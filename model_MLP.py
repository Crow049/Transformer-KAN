import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(MLP, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 创建隐藏层列表（每层后接 LayerNorm 和 ReLU）
        layers = []
        in_dim = input_dim
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))  # Layer normalization
            layers.append(nn.Sigmoid())
            # layers.append(nn.ReLU())
            in_dim = hidden_dim

        self.hidden_layers = nn.Sequential(*layers)
        self.batch_norm = nn.BatchNorm1d(hidden_dim)  # Batch normalization
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = x.mean(dim=1)  # -> (batch_size, input_dim = 15)
        out = self.hidden_layers(x)
        out = self.batch_norm(out)
        out = self.output_layer(out)
        return out.reshape(-1, )
