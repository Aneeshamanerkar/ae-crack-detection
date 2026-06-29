#models/lstm.py

import torch.nn as nn
from .base import SequenceLabeler

class LSTMSequenceLabeler(SequenceLabeler):
    def __init__(self, n_freq, n_classes, hidden_size=128, num_layers=2, bidirectional=True, dropout=0.2):
        super().__init__(n_freq, n_classes)
        
        self.lstm = nn.LSTM(
            input_size=n_freq,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        num_directions = 2 if bidirectional else 1
        self.classifier = nn.Linear(hidden_size * num_directions, n_classes)

    def forward(self, x):
        # x: (Batch, Time, Freq)
        lstm_out, _ = self.lstm(x)
        # lstm_out: (Batch, Time, Hidden * Directions)
        logits = self.classifier(lstm_out)
        # logits: (Batch, Time, Classes)
        return logits