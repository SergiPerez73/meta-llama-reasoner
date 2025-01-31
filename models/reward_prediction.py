import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer

model_embeddings = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
def get_embedding(sentence):
    embedding = model_embeddings.encode(sentence)
    return embedding


class ModelRewardPrediction(nn.Module):
    def __init__(self, layer_config_embedding, layer_config_general, n_embeddings=2):
        super(ModelRewardPrediction, self).__init__()
        
        # Create embedding layers
        self.embedding_layers = nn.ModuleList()
        for _ in range(n_embeddings):
            layers = []
            input_dim = 768
            layer_dims = [int(dim) for dim in layer_config_embedding.split()]
            
            for dim in layer_dims:
                layers.append(nn.Linear(input_dim, dim))
                layers.append(nn.ReLU())
                input_dim = dim
            
            layers.pop()  # Remove the last ReLU
            self.embedding_layers.append(nn.Sequential(*layers))
        
        # Create general layers
        layers = []
        input_dim = n_embeddings * layer_dims[-1]
        layer_dims = [int(dim) for dim in layer_config_general.split()]
        
        for dim in layer_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            input_dim = dim
        
        layers.pop()  # Remove the last ReLU
        layers.append(nn.Sigmoid())  # Add Sigmoid for the final layer
        
        self.general_layers = nn.Sequential(*layers)

    def forward(self, x_list):
        # Apply embedding layers to each input
        x_output_list = []
        for i, x in enumerate(x_list):
            x_output_list.append(self.embedding_layers[i](x))
        
        # Concatenate the outputs
        x = torch.cat(x_output_list, dim=1)
        
        # Apply general layers
        return self.general_layers(x)