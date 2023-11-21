import torch
import torch.nn as nn


class ImagineNet_FC(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, int(in_channels / 4))
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(int(in_channels / 4), num_classes)

        self.act = nn.ReLU()

        self.loss_evaluator = nn.BCEWithLogitsLoss()
    
    def forward(self, x, target=None):
        x = self.fc1(x)
        x = self.act(self.dropout(x))
        cls_score = self.fc2(x)

        if self.training:
            return cls_score, self.loss_evaluator(cls_score, target)
        else:
            return cls_score
