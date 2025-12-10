import torch
from torch import nn

from core.models.blocks import RNN, MLP, fetch_input_dim


class RNNMLP(nn.Module):
    def __init__(self, config, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config = config
        
        # --- CONFIGURAZIONE ---
        # Assumiamo che input_dimension sia letto dal config o dai dati
        input_dimension = fetch_input_dim(config) 
        
        rnn_hidden_size = 512
        rnn_num_layers = 2
        
        # Parametri interni dell'MLP
        mlp_hidden_size = 512
        mlp_output_size = 1

        # --- INIZIALIZZAZIONE LAYER ---
        
        # 1. Encoder RNN
        self.rnn = RNN(
            input_size=input_dimension, 
            hidden_size=rnn_hidden_size, 
            num_layers=rnn_num_layers, 
            dropout=0.5, 
            bidirectional=False
        )
        
        # 2. Decoder MLP
        # NOTA CRUCIALE: L'input dell'MLP deve essere uguale all'output della RNN
        mlp_input_dim = rnn_hidden_size 
        
        self.decoder = MLP(
            input_size=mlp_input_dim, 
            hidden_size=mlp_hidden_size, 
            output_size=mlp_output_size
        )

    def forward(self, input_data):
        # 1. Pulizia dati (NaN check)
        input_data = torch.nan_to_num(input_data, nan=0.0, posinf=1.0, neginf=-1.0)

        # 2. Feature Extraction (RNN)
        # L'RNN restituisce un tensore: [Batch Size, 512]
        rnn_features = self.rnn(input_data)
        
        # (Opzionale) Debug dimensioni se necessario:
        # print(f"Shape uscita RNN: {rnn_features.shape}") 

        # 3. Classificazione/Regressione (MLP)
        # Passiamo le feature direttamente al decoder
        final_output = self.decoder(rnn_features)

        return final_output