import torch
import torch.nn as nn
from ..model import PositionalEncoding

class ImagineNet_CA(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        # Position
        self.positionEncoder = PositionalEncoding(d_model=2048, dropout=0.1)
        
        self.encoder_layer1 = nn.CrossTransformerEncoderLayer(
            d_model=2048, 
            nhead=4, 
            dropout=0.1, 
            batch_first=True
            )
        # self.encoder_layer2 = nn.TransformerEncoderLayer(
        #     d_model=2048, 
        #     nhead=4, 
        #     dropout=0.1, 
        #     batch_first=True
        #     )
        # self.encoder_layer3 = nn.TransformerEncoderLayer(
        #     d_model=2048, 
        #     nhead=4, 
        #     dropout=0.1, 
        #     batch_first=True
        #     )

        self.fc1 = nn.Linear(in_channels, int(in_channels / 4))
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(int(in_channels / 4), num_classes)

        self.loss_evaluator = nn.BCEWithLogitsLoss()
    
    def forward(self, x1, x2, target=None):
        
        x1 = self.positionEncoder(x1)
        x2 = self.positionEncoder(x2)

        x = self.encoder_layer1(x1, x2) # (8, 8, 2048)
        # x = self.encoder_layer2(x)
        # x = self.encoder_layer3(x)

        x = torch.mean(x, dim=1)
        
        x = self.fc1(x)
        x = self.dropout(x)
        cls_score = self.fc2(x)

        if self.training:
            return cls_score, self.loss_evaluator(cls_score, target)
        else:
            return cls_score
