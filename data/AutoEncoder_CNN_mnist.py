import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

import matplotlib.pyplot as plt

from keras.datasets import mnist, fashion_mnist
    
class MyDataset(Dataset):
    def __init__(self, X, Y):
        self.X = torch.from_numpy(X)     
        self.Y = Y
        
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]    

class CNN_Model(nn.Module):
    def __init__(self):
        super(CNN_Model, self).__init__()
        # Convolution 1 , input_shape=(1,28,28)
        self.cnn1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=0) #output_shape=(16,24,24)
        self.relu1 = nn.ReLU() # activation
        # Max pool 1
        self.maxpool1 = nn.MaxPool2d(kernel_size=2) #output_shape=(16,12,12)
        # Convolution 2
        self.cnn2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=5, stride=1, padding=0) #output_shape=(32,8,8)
        self.relu2 = nn.ReLU() # activation
        # Max pool 2
        self.maxpool2 = nn.MaxPool2d(kernel_size=2) #output_shape=(32,4,4)
        # Fully connected 1 ,#input_shape=(32*4*4)
        self.fc1 = nn.Linear(32 * 4 * 4, 10) 
    
    def forward(self, x, get_representation = False):
        # Convolution 1
        out = self.cnn1(x)
        out = self.relu1(out)
        # Max pool 1
        out = self.maxpool1(out)
        # Convolution 2 
        out = self.cnn2(out)
        out = self.relu2(out)
        # Max pool 2 
        out = self.maxpool2(out)
        out = out.view(out.size(0), -1)
        if get_representation:
            return out
 
        # Linear function (readout)
        out = self.fc1(out)
        return out

class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()

        self.encoder=nn.Sequential(             
            nn.Linear(512, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 2),
            nn.Sigmoid()     
        )                                       
        self.decoder=nn.Sequential(             
            nn.Linear(2, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 512),
            nn.Sigmoid()                                                    
        )  

    def forward(self, inputs):
        codes = self.encoder(inputs)
        decoded = self.decoder(codes)

        return codes, decoded

def load_fashion_mnist_data():
    (X_train, Y_train), (X_test, Y_test) = fashion_mnist.load_data()
    X_train = X_train.astype('float32') / 255 #60000,28,28
    X_test = X_test.astype('float32') / 255 #10000,28,28
    data = np.concatenate((X_train, X_test), axis = 0) #70000,28,28
    target = np.concatenate((Y_train, Y_test), axis = 0)
    
    data = torch.from_numpy(data)
    target = torch.from_numpy(target).type(torch.LongTensor) # data type is long
    return data, target

def load_mnist_data():
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    X_train = X_train.astype('float32') / 255 #60000,28,28
    X_test = X_test.astype('float32') / 255 #10000,28,28

    data = np.concatenate((X_train, X_test), axis = 0) #70000,28,28
    target = np.concatenate((Y_train, Y_test), axis = 0)
    return data, target


def cnn_train(train_loader, model, optimizer, loss_function, epochs, input_shape):
    training_loss = []
    training_accuracy = []

    for epoch in range(epochs):

        correct_train = 0
        total_train = 0
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            train = Variable(images.view(input_shape))
            labels = Variable(labels)
   
            optimizer.zero_grad()

            outputs = model(train)
   
            train_loss = loss_function(outputs, labels)

            train_loss.backward()

            optimizer.step()
 
            predicted = torch.max(outputs.data, 1)[1]
 
            total_train += len(labels)

            correct_train += (predicted == labels).float().sum()
        
        train_accuracy = 100 * correct_train / float(total_train)
        training_accuracy.append(train_accuracy)

        training_loss.append(train_loss.data)
 
        print('Train Epoch: {}/{} Traing_Loss: {} Traing_acc: {:.6f}%'.format(epoch+1, epochs, train_loss.data, train_accuracy))
    
    #torch.save(model, 'fashion_mnist_cnn.pth')    
    return training_loss, training_accuracy

def cnn_pretrained():
    #load fashion_mnist dataset
    data, target = load_fashion_mnist_data()
    
    #dataloader
    my_dataset = TensorDataset(data, target)
    train_loader = DataLoader(my_dataset, batch_size=100, shuffle=True)
    
    #CNN setting
    batch_size = 100
    n_iters = 10000
    epochs = int(n_iters / (len(data) / batch_size))
    
    model = CNN_Model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)  
    loss_function = nn.CrossEntropyLoss().to(device)
    input_shape = (-1,1,28,28)
    
    #training
    training_loss, training_accuracy = cnn_train(train_loader, model, optimizer, loss_function, epochs, input_shape)
    
    return model

def mnist_feature_extraction(model):
    data, target = load_mnist_data()
    representation = []
    #model = torch.load('fashion_mnist_cnn.pth')
   
    model.eval()
    input_shape = (-1,1,28,28)
    for i, images in enumerate(data):
        
        images = torch.tensor(images).to(device)
        train = Variable(images.view(input_shape))
        outputs = model(train, True)
        representation.append(outputs.detach().cpu().numpy())

    representation = np.array(representation)
    return representation, target

def AutoEncoder_train(train_loader, model, optimizer, loss_function, epochs):
    for epoch in range(epochs):
        for data, labels in train_loader:

            inputs = data.view(-1, 512).to(device) 
            model.zero_grad()
            # Forward
            codes, decoded = model(inputs)

            # Backward
            optimizer.zero_grad()
            loss = loss_function(decoded, inputs)

            loss.backward(retain_graph=True)
            optimizer.step()

        # Show progress
        print('[{}/{}] Loss:'.format(epoch+1, epochs), loss.item())


    # Save
    #torch.save(model, 'mnist_autoencoder.pth')
    
def dimension_reduction(representation, model):
    data_2d = []
    for i in range(len(representation)):
        x = torch.tensor(representation[i])
        inputs = x.view(-1, 512).to(device) 
        codes, decoded = model(inputs)
        data_2d.append(codes[0].detach().cpu().numpy())
        
    data_2d = np.array(data_2d)
    return data_2d

def plot_figure(data_2d, target):
    plt.figure(figsize=(10, 10))
    for i in range(data_2d.shape[0]):
        plt.text(data_2d[i,0], data_2d[i,1], str(target[i]), color=plt.cm.Set1(target[i]), 
                     fontdict={'weight': 'bold', 'size': 8})
    plt.xticks([])
    plt.yticks([])
    plt.show()
    
if __name__ == "__main__":   
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #CNN pretrained on fashion mnist
    model = cnn_pretrained()
    
    #MNIST feature extraction
    representation, target = mnist_feature_extraction(model)
    
    #AutoEncoder dataloader
    my_dataset = MyDataset(representation, target)
    train_loader = DataLoader(my_dataset, batch_size=64,shuffle=True)

    #AutoEncoder setting
    epochs = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    AE_model = AutoEncoder().to(device)
    optimizer = torch.optim.Adam(AE_model.parameters(), lr=0.001)
    loss_function = nn.MSELoss().to(device)

    #training
    AutoEncoder_train(train_loader, AE_model, optimizer, loss_function, epochs)

    #Dimension reduction
    data_2d = dimension_reduction(representation, AE_model)
    
    #plot 
    plot_figure(data_2d, target)