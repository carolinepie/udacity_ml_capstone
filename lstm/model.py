import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, hidden_dim):
        """
        Initialize the model by settingg up the various layers.
        """
        super(LSTMClassifier, self).__init__()

        self.lstm = nn.LSTM(input_size=16, hidden_dim=hidden_dim)
        self.dense = nn.Linear(in_features=hidden_dim, out_features=1)
        self.hardswish = nn.Hardswish()
        
    def forward(self, x):
        """
        Perform a forward pass of our model on some input.
        """
        lstm_out, _ = self.lstm(x)
        out = self.dense(lstm_out)
        print(out)
        return self.hardswish(out)