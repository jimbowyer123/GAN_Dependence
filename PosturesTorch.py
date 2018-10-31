import torch
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
import warnings
warnings.filterwarnings("ignore")

# print(torch.__version__)

# print(torch.cuda.is_available())
use_cuda = torch.cuda.is_available()
# print('Available devices', torch.cuda.device_count())
# print('Current cuda device', torch.cuda.current_device())
# device = torch.cuda.set_device(0 if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

class Model(nn.Module):
    def __init__(self):
        #network goes here
        super(Model, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(36, 108),
            nn.ReLU(),
            nn.Linear(108, 5)
        )
    
    def forward(self,x, **kwargs):
    #     #define forward pass
    #     # X =
        return self.net(x)

class HandPosture(Dataset):
    """Hand Postures Dataset"""

    def __init__(self, csv_file, transforms=None):
        #transforms
        # self.to_tensor = transforms.ToTensor()
        #Read the csv file
        self.data = pd.read_csv(csv_file, na_values=(['?'])).fillna(0)
        #First column contains labels
        self.label_arr = np.asarray(self.data.iloc[:, 0])
        #second column is user data
        self.user_arr = np.asarray(self.data.iloc[:, 1])
        #return coordinate data
        self.coordinate = (self.data.iloc[:, 2:].values)
        #return number of examples
        self.data_len = len(self.data.index)
        self.transforms = transforms
        
    def __len__(self):
        return self.data_len

    def __getitem__(self, index):
        #return single tensor label from pd df
        classification = self.label_arr[index]
        #reshape coordinates into input
        coordinate = self.coordinate[index]
        coordinate = coordinate.astype(np.float32)
        X = coordinate
        y = classification
        # print(X)
        # print(y)
        # print(X.dtype)
        # print(y.dtype)
        #Return classification label and coordinate frame
        return X, y

#Parameter
# batch_size denoted number of samples contained in each generated batch
# shuffle=True shuffles order of examples fed into the classifier
# num_workers denoted the number of processes that generate batches in parallel
params = {'batch_size':64, 'shuffle':True, 'num_workers':8}
validation_split = .2
random_seed = 42

Poses = ('Fist', 'Stop', 'Point1'. 'Point2', 'Grab')

dataset = HandPosture("Postures.csv", transforms=transforms.Compose([transforms.ToTensor()]))
# print(len(dataset))

indices = list(range(len(dataset)))
split = int(np.floor(validation_split*len(dataset)))
train_indices, val_indices = indices[split:], indices[split:]

train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)

train_loader = DataLoader(dataset, batch_size=64, sampler=train_sampler)
val_loader = DataLoader(dataset, batch_size=64, sampler=val_sampler)

model = Model()
device = torch.device('cuda:0')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optim = torch.optim.SGD(params, lr=1e-3, momentum=0.9)

max_epochs = 100
for epoch in range(max_epochs):
    
    running_loss = 0.0

    #train
    for batch_index, (X,y) in enumerate(train_loader):

        X = X.to(device)
        y = y.to(device)

        pred = model(X)
        print(pred.size())
        print(y.size())


    #test
    for batch_index, (X, y) in enumerate(val_loader):

        X = X.to(device)
        y = y.to(device)

        pred = model(X)

print("Finished training")