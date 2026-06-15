import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models



class CRNN_Model(nn.Module): 
    def __init__(self, vocab_size, hidden_size=384, num_layers=2, dropout=0.3):
        super(CRNN_Model, self).__init__()  
        resnet_model = models.resnet18(pretrained=True)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # ========== ENCODER ==========
        self.conv1 = resnet_model.conv1
        self.bn1 = resnet_model.bn1
        self.relu = resnet_model.relu
        
        self.layer1 = resnet_model.layer1
        self.layer2 = resnet_model.layer2

        #modified layer3
        self.layer3 = resnet_model.layer3
        self.layer3[0].conv1.stride = (1, 1)
        self.layer3[0].downsample[0].stride = (1, 1)
        
        # Pool height AFTER all convolutions
        self.height_pool = nn.AdaptiveAvgPool2d((6, None))  # Pool to height=6
        
        self.encoder_channels = 256
        self.encoder_height = 6
        self.feature_dims = self.encoder_channels * self.encoder_height  # 768
        
        # ========== DROPOUT ==========
        self.dropout = nn.Dropout(dropout)
        
        # ========== LSTM ==========
        self.lstm = nn.LSTM(
            input_size=self.feature_dims,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # ========== OUTPUT ==========
        self.fc = nn.Linear(hidden_size * 2, vocab_size )
        
    def forward(self, x):
        # Encoder forward (NO maxpool)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        # Pool height AFTER convolutions (critical!)
        x = self.height_pool(x)  # (batch, 256, 6, width)
        
        # Reshape for LSTM
        x = x.permute(0, 3, 2, 1)  # (batch, width, 6, 256)
        x = x.flatten(2)           # (batch, width, 6*256)
        
        x = self.dropout(x)
        
        # LSTM
        x, (hidden, cell) = self.lstm(x)  # (batch, width, 512)
        
        x = self.dropout(x)
        x = self.fc(x)  # (batch, width, vocab_size)
        
        # CTC expects (time_steps, batch, vocab_size)
        x = x.permute(1, 0, 2)  # (width, batch, vocab_size)
        
        return x
