
import torch
import torch.nn as nn
from torch.nn.modules.transformer import TransformerEncoderLayer, TransformerEncoder

class SimpleClassifier(nn.Module):
    def __init__(self,num_classes, embed_size=1024, hidden_dim1=512,  dropout_rate=0.4):
        """
        Simple neural network for Nominal classification. 

        Args:
            embed_size: Dimension of protein embeddings
            hidden_dim1: Dimension of hidden layer
            dropout: Dropout probability
            num_classes: output_dim
        """
        super().__init__()
        seq = [ 
            nn.Linear(embed_size, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.BatchNorm1d(hidden_dim1),
            nn.Linear(hidden_dim1, num_classes),]
        
        self.network = nn.Sequential(
           *seq
        )

    def forward(self, x):
        
        
        return self.network(x)
class DeepSimpleClassifier(nn.Module):
    def __init__(self,num_classes, embed_size=1024, hidden_dim1=512,  dropout_rate=0.4):
        """
        Deep Simple neural network for Nominal classification. 

        Args:
            embed_size: Dimension of protein embeddings
            hidden_dim1: Dimension of hidden layer
            dropout: Dropout probability
            num_classes: output_dim
        """
        super().__init__()
        seq = [ 
            nn.Linear(embed_size, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, hidden_dim1//2),
            nn.ReLU(),
            nn.Linear(hidden_dim1//2, hidden_dim1//4),
            nn.ReLU(),
            nn.Linear(hidden_dim1//4, hidden_dim1//8),
            nn.ReLU(),
            nn.Linear(hidden_dim1//8, num_classes),]
        
        self.network = nn.Sequential(
           *seq
        )

    def forward(self, x):
        
        
        return self.network(x)
class TransformerClassifier(nn.Module):
    def __init__(self,num_classes, embed_size=1024, hidden_dim1=512,  dropout_rate=0.4,nhead = 4,dim_feedforward=2048,num_layers_transformer = 2):
        """
        
        Transformer neural network for Nominal classification/regression. 

        Args:
            embed_size: Dimension of protein embeddings
            hidden_dim1: Dimension of hidden layer
            dropout: Dropout probability
            num_classes: output_dim
        """
        super().__init__()
        transformer_layer = TransformerEncoderLayer(
                d_model=embed_size,nhead=nhead,dim_feedforward=dim_feedforward,dropout=dropout_rate,batch_first=True)
        self.transformer_encoder = TransformerEncoder(transformer_layer, num_layers=num_layers_transformer)
        seq = [ 
            nn.Linear(embed_size, hidden_dim1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim1, hidden_dim1//2),
            nn.ReLU(),
            nn.Linear(hidden_dim1//2, num_classes),]
        
        self.network = nn.Sequential(
           *seq
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.transformer_encoder(x)
        x = x.squeeze(1)
        return self.network(x)