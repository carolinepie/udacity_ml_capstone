import torch.nn as nn
from torch.nn.init import kaiming_normal_

class LSTMClassifier(nn.Module):
    def __init__(self, sequence_size, input_size, hidden_dim):
        """
        Initialize the model by settingg up the various layers.
        """
        super(LSTMClassifier, self).__init__()

        self.lstm = nn.LSTM(input_size=input_size, hidden_size = hidden_dim, batch_first=True, num_layers=1)
        kaiming_normal_(self.lstm.weight_ih_l0)
        kaiming_normal_(self.lstm.weight_hh_l0)
        self.lstm.bias_ih_l0.data.zero_()
        self.lstm.bias_hh_l0.data.zero_()
        n = self.lstm.bias_hh_l0.size(0)
        start, end = n//4, n//2
        self.lstm.bias_hh_l0.data[start:end].fill_(1.)
        
        interim = 1
        self.dense = nn.Linear(in_features=hidden_dim, out_features=interim)
        kaiming_normal_(self.dense.weight.data)
        self.activation = nn.ReLU()
        
        self.dense2 = nn.Linear(in_features=interim, out_features=1)
        kaiming_normal_(self.dense2.weight.data)
#         self.activation2 = nn.ReLU()
    def forward(self, x):
        """
        Perform a forward pass of our model on some input.
        """
        lstm_out, (h_n, c_n) = self.lstm(x)
        out = self.dense(lstm_out[:, -1])
#         linear1 = self.activation(out)
        
#         out2 = self.dense2(linear1)
#         final = self.activation2(out2)        
        
#         print('final')
#         print(final)
        return out