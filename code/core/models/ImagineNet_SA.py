import torch
import torch.nn as nn
from ..model import PositionalEncoding

class ImagineNet_SA(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        # Position
        self.positionEncoder = PositionalEncoding(d_model=2048, dropout=0.1)

        self.encoder_layer1 = nn.TransformerEncoderLayer(d_model=2048, nhead=4, batch_first=True) # FLOPs: 67239936
        # self.encoder_layer2 = nn.TransformerEncoderLayer(d_model=2048, nhead=4, batch_first=True) # FLOPs: 67239936
        # self.encoder_layer3 = nn.TransformerEncoderLayer(d_model=2048, nhead=4, batch_first=True) # FLOPs: 67239936

        self.fc1 = nn.Linear(in_channels, int(in_channels / 4))
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(int(in_channels / 4), num_classes)

        self.loss_evaluator = nn.BCEWithLogitsLoss()
    
    def forward(self, x, target=None):
        x = self.positionEncoder(x)
        x = self.encoder_layer1(x) # [8, 8, 2048] / test: [1, 8, 2048]
        # x = self.encoder_layer2(x)
        # x = self.encoder_layer3(x)

        x = torch.mean(x, dim=1) # [8, 2048]
        
        x = self.fc1(x)             # FLOPs: 1048576
        x = self.dropout(x)
        cls_score = self.fc2(x)     # FLOPs: 7168

        if self.training:
            return cls_score, self.loss_evaluator(cls_score, target)
        else:
            return cls_score

