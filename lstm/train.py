import argparse
import json
import os
import pandas as pd
import numpy as np
import torch
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms

# imports the model in model.py by name
from model import LSTMClassifier
from window_dataset import WindowDataset
def model_fn(model_dir):
    """Load model"""
    print("Load model.")

    # First, load the parameters used to create the model.
    model_info = {}
    model_info_path = os.path.join(model_dir, 'model_info.pth')
    with open(model_info_path, 'rb') as f:
        model_info = torch.load(f)

    print("model_info: {}".format(model_info))

    # Determine the device and construct the model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMClassifier(model_info['sequence_size'], model_info['input_size'], model_info['hidden_dim'])

    # Load the stored model parameters.
    model_path = os.path.join(model_dir, 'model.pth')
    with open(model_path, 'rb') as f:
        model.load_state_dict(torch.load(f))

    # set to eval mode
    model.to(device).train()

    print("Done loading model.")
    return model

def _get_train_data_loader(input_size, seq_length, batch_size, training_dir, train_file):
    print("Get train data loader.")
    print('batch_size: ' + str(batch_size))

    train_data = pd.read_csv(os.path.join(training_dir, train_file), header=None, names=None)
    
    train_y = torch.from_numpy(train_data[[0]].values).float().squeeze()
    train_x = torch.from_numpy(np.reshape(train_data.drop([0], axis=1).iloc[:,:input_size].values, (-1, input_size))).float()
    
    train_ds = WindowDataset(train_x, train_y, seq_length=seq_length)
    return torch.utils.data.DataLoader(train_ds, batch_size=batch_size)


# Provided training function
def train(model, train_loader, epochs, criterion, optimizer, device):
    """
    This is the training method that is called by the PyTorch training script. The parameters
    passed are as follows:
    model        - The PyTorch model that we wish to train.
    train_loader - The PyTorch DataLoader that should be used during training.
    epochs       - The total number of epochs to train for.
    criterion    - The loss function used for training. 
    optimizer    - The optimizer to use during training.
    device       - Where the model and data should be loaded (gpu or cpu).
    """
    print('epochs: ' + str(epochs))
    # training loop is provided
    for epoch in range(1, epochs + 1):
        model.train() # Make sure that the model is in training mode.

        total_loss = 0

        for batch in train_loader:
            # get data
            batch_x, batch_y = batch
#             print('batch_x')
#             print(batch_x)
#             print('batch_y')
#             print(batch_y)
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()

            # get predictions from model
            y_pred = model(batch_x).squeeze()
#             print(y_pred)
            
            # perform backprop
            loss = criterion(y_pred, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.data.item()

        print("Epoch: {}, Loss: {}".format(epoch, total_loss / len(train_loader)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    # training params
    parser.add_argument('--batch-size', type=int, default=10, metavar='N',
                        help='input batch size for training (default: 16)')
    parser.add_argument('--epochs', type=int, default=500, metavar='N',
                        help='number of epochs to train (default: 500)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--input-size', type=int, default=8, metavar='N',
                        help='hidden dimension (default: 8)')
    parser.add_argument('--sequence-size', type=int, default=8, metavar='N',
                        help='hidden dimension (default: 8)')
    parser.add_argument('--hidden-dim', type=int, default=8, metavar='N',
                        help='hidden dimension (default: 8)')
    parser.add_argument('--learning-rate', type=float, default=0.1, metavar='N',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--train-file', type=str, default='', metavar='N',
                        help='training file')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device {}.".format(device))

    torch.manual_seed(args.seed)

    # Load training data.
    train_loader = _get_train_data_loader(args.input_size, args.sequence_size, args.batch_size, args.data_dir, args.train_file)

    model = LSTMClassifier(args.sequence_size, args.input_size, args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = torch.nn.MSELoss() # we are using MSE as we are using RMSE to evaluate the model

    # Trains the model
    train(model, train_loader, args.epochs, criterion, optimizer, device)

    model_info_path = os.path.join(args.model_dir, 'model_info.pth')
    with open(model_info_path, 'wb') as f:
        model_info = {
            'train_file': args.train_file,
            'input_size': args.input_size,
            'hidden_dim': args.hidden_dim,
            'learning_rate': args.learning_rate,
            'sequence_size': args.sequence_size
        }
        torch.save(model_info, f)    

    # Save the model parameters
    model_path = os.path.join(args.model_dir, 'model.pth')
    with open(model_path, 'wb') as f:
        torch.save(model.cpu().state_dict(), f)