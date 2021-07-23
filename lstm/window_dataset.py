import torch
import torch.utils.data as data

# def listToTensor(list):
#     tensor = torch.empty(list.__len__(), list[0].__len__())
#     for i in range(list.__len__()):
#         tensor[i, :] = torch.FloatTensor(list[i])
#     return tensor

class WindowDataset(data.Dataset):
    def __init__(self, x, y, seq_length):
        self.x = x
        print('self.x[0]')
        print(self.x[0])
        self.y = y
        print('self.y[0]')
        print(self.y[0])
        self.seq_length = seq_length
    def __len__(self):
        return self.x.__len__()    
    def __getitem__(self, idx):
        if idx + self.seq_length >= self.__len__():
            x_final = torch.zeros(self.seq_length, self.x[0].__len__())
            y_final = torch.zeros(self.seq_length)
            x_final[:self.x.__len__()-idx] = self.x[idx:]
            return x_final, self.y[-1]
        else:
            return self.x[idx:idx+self.seq_length], self.y[idx+self.seq_length]
    