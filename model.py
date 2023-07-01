import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers):
        super(MLPModel, self).__init__()

        self.embedding = nn.Linear(input_size, hidden_size)
        self.linears = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for i in range(n_layers)])
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        for layer in self.linears:
            x = layer(x)
            x = F.relu(x)

        x = self.output(x)
        x = F.sigmoid(x)

        return x


class EncoderModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers, n_head):
        super(EncoderModel, self).__init__()

        self.embedding = nn.Linear(1, hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=n_head, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output = nn.Sequential(nn.Linear(input_size * hidden_size, hidden_size),
                                    nn.ReLU(),
                                    nn.Linear(hidden_size, output_size))

    def forward(self, x):
        x = x.unsqueeze(dim=-1)
        x = self.embedding(x)
        x = self.encoder(x)

        x = x.view(x.shape[0], -1)

        x = self.output(x)
        x = F.sigmoid(x)
        return x


