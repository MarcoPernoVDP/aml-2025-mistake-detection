import torch
from torch import nn

from core.models.blocks import RNN, MLP, fetch_input_dim

class RNNMLP(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config

        input_dimension = fetch_input_dim(config)
        
        rnn_hidden_size = 512
        rnn_num_layers = 2

        mlp_hidden_size = 512
        mlp_output_size = 1

        self.rnn = RNN(
            input_size=input_dimension,
            hidden_size=rnn_hidden_size,
            num_layers=rnn_num_layers,
            dropout=0.5,
        )

        self.decoder = MLP(
            input_size=rnn_hidden_size,
            hidden_size=mlp_hidden_size,
            output_size=mlp_output_size
        )

    def forward(self, input_data):
        # input_data shape attuale: [T, D]
        input_data = torch.nan_to_num(input_data, nan=0.0, posinf=1.0, neginf=-1.0)

        # => rnn_features: [1, 512]
        rnn_features = self.rnn(input_data)

        # => final_output: [1, 1]
        final_output = self.decoder(rnn_features.unsqueeze(0))  # shape [1,1]


        return final_output
