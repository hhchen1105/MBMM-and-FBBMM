import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torchvision
from sklearn.preprocessing import MinMaxScaler
from sklearn import datasets
import matplotlib.pyplot as plt 

class MyDataset(Dataset):
    def __init__(self, X, Y):
        super(MyDataset, self).__init__()
        MMScaler=MinMaxScaler()
        X = MMScaler.fit_transform(X).astype(np.float32)
        self.X = torch.from_numpy(X)
        self.Y = Y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder=nn.Sequential(             
            nn.Linear(13, 2),
            nn.Sigmoid()                                           
        )                                       
        self.decoder=nn.Sequential(                                                             
            nn.Linear(2, 13),  
            nn.Sigmoid()                       
        )  

    def forward(self, inputs):
        codes = self.encoder(inputs)
        decoded = self.decoder(codes)
        return codes, decoded    

def load_data():
    dataset = datasets.load_wine()
    data = torch.from_numpy(dataset.data.astype(np.float32))
    target = torch.from_numpy(dataset.target)
    return data, target

def train(train_loader, model, optimizer, loss_function, epochs):
    for epoch in range(epochs):
        for data, labels in train_loader:
            inputs = data.to(device)

            model.zero_grad()
            # Forward
            codes, decoded = model(inputs)

            # Backward
            optimizer.zero_grad()
            loss = loss_function(decoded, inputs)
            loss.backward()
            optimizer.step()

        # Show progress
        if epoch % 100 == 0:
            print('[{}/{}] Loss:'.format(epoch+1, epochs), loss.item())

    #Save
    #torch.save(model, 'wine_dataset.pth')
    return model

def dimension_reduction(my_dataset, model):
    data_2d = []
    for i, (x, y) in enumerate(my_dataset):
        codes, decoded = model_fit(x.to(device))
        data_2d.append(codes.detach().cpu().numpy())
    data_2d = np.array(data_2d)
    return data_2d

def plot_figure(data_2d, target):
    plt.figure(figsize=(4, 4))
    for i in range(data_2d.shape[0]):
        plt.text(data_2d[i,0], data_2d[i,1], str(target[i]), color=plt.cm.Set1(target[i]), 
                     fontdict={'weight': 'bold', 'size': 8})
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
if __name__ == "__main__":    
    #load data
    data, target = load_data()

    #dataloader
    my_dataset = MyDataset(data, target)
    train_loader = DataLoader(my_dataset, batch_size=16,shuffle=True)

    #AutoEncoder setting
    epochs = 200
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoEncoder().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_function = nn.MSELoss().to(device)

    #training
    model_fit = train(train_loader, model, optimizer, loss_function, epochs)

    #Dimension reduction
    data_2d = dimension_reduction(my_dataset, model)
    
    #plot 
    plot_figure(data_2d, target.numpy())