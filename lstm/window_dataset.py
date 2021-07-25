import torch
import torch.utils.data as data

# full x, full y: 

class WindowDataset(data.Dataset):
    def __init__(self, x, y, seq_length):
        self.x = x
        self.y = y
        
        self.seq_length = seq_length
    def __len__(self):
#         print(self.x.__len__())
#         print(self.seq_length)
        return self.x.__len__() - self.seq_length + 1 
    def __getitem__(self, idx):
#         if idx + self.seq_length >= self.__len__():
#             x_final = torch.zeros(self.seq_length, self.x[0].__len__())
#             y_final = torch.zeros(self.seq_length)
#             x_final[:self.x.__len__()-idx] = self.x[idx:]
#             return x_final, self.y[-1]
#         else:
#             return self.x[idx:idx+self.seq_length], self.y[idx+self.seq_length-1]
        return self.x[idx:idx+self.seq_length], self.y[idx+self.seq_length -1]
    