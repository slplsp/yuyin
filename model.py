import torch.nn as nn
import torch
import torch.nn as nn
import torch
import torch.nn.functional as F

class VoiceEncoder(nn.Module):
    def __init__(self, embedding_size=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU()
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, embedding_size)

    def forward(self, x):  # x: [B, mel, time]
        x = x.unsqueeze(1)  # [B, 1, mel, time]
        x = self.conv(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return nn.functional.normalize(x, p=2, dim=1)



class VoiceEncoder1(nn.Module):
    def __init__(self, embedding_size=256, lstm_hidden_size=512, num_lstm_layers=2):
        super().__init__()
        
        # CNN front-end
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, None))  # Keep temporal dimension
        )
        
        # BiLSTM layers
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=lstm_hidden_size,
            num_layers=num_lstm_layers,
            bidirectional=True,
            batch_first=True
        )
        
        # Projection layers
        self.projection = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, lstm_hidden_size),
            nn.BatchNorm1d(lstm_hidden_size),
            nn.ReLU(),
            nn.Linear(lstm_hidden_size, embedding_size)
        )
        
        # Attention pooling
        self.attention = nn.Sequential(
            nn.Linear(lstm_hidden_size * 2, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        
        self.embedding_size = embedding_size
        
    def forward(self, x):
        # x shape: [B, mel, time]
        x = x.unsqueeze(1)  # [B, 1, mel, time]
        
        # CNN processing
        x = self.conv(x)  # [B, 512, 1, T]
        x = x.squeeze(2)  # [B, 512, T]
        x = x.transpose(1, 2)  # [B, T, 512]
        
        # BiLSTM processing
        x, _ = self.lstm(x)  # [B, T, lstm_hidden_size*2]
        
        # Attention pooling
        attn_weights = self.attention(x)  # [B, T, 1]
        x = torch.sum(x * attn_weights, dim=1)  # [B, lstm_hidden_size*2]
        
        # Projection to embedding space
        x = self.projection(x)
        
        return F.normalize(x, p=2, dim=1)