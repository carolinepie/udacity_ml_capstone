import torch.nn as nn

class LSTMClassifier(nn.Module):
    def __init__(self, sequence_size, input_size, hidden_dim):
        """
        Initialize the model by settingg up the various layers.
        """
        super(LSTMClassifier, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size = hidden_dim, batch_first=True, num_layers=5)
        self.dense = nn.Linear(in_features=hidden_dim, out_features=1)
        self.hardswish = nn.Hardswish()
        
    def forward(self, x):
        """
        Perform a forward pass of our model on some input.
        """
#         print(x.shape)
#         print(x)
        lstm_out, (h_n, c_n) = self.lstm(x)
#         print(h_n[-1])
        out = self.dense(h_n[-1])
        final=out
#         final = self.hardswish(out)
#         print('final')
#         print(final)
        return final