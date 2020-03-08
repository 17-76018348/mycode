import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import trange

import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader



def get_dataloader(train_batch_size, val_batch_size):
    train_dataset = datasets.MNIST('./MNIST_Dataset', train = True, download = True, transform = transforms.ToTensor())
    validation_dataset = datasets.MNIST("./MNIST_Dataset", train = False, download = True, transform = transforms.ToTensor())
    
    train_loader = DataLoader(dataset = train_dataset, batch_size = train_batch_size, shuffle = True)
    validation_loader = DataLoader(dataset = validation_dataset, batch_size = val_batch_size, shuffle = True)
    return train_loader, validation_loader

class MNIST_Autoencoder(nn.Module):
    def __init__(self, mode = 'training'):
        super(MNIST_Autoencoder, self).__init__()
        self.mode = mode
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU(True)
            )
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 28*28),
            nn.Sigmoid()
            )
        self.encoder_outputs = []
        
    def forward(self, x):
        encoder_out = self.encoder(x)
        decoder_out = self.decoder(encoder_out)
        if self.mode == 'test':
            self.encoder_outputs.append(encoder_out)
        return decoder_out



def MLP_train():
    epochs = 10
    train_batch_size, val_batch_size = 10, 2000
    lr = 0.001
    
    loss_list = []
    
    train_loader, validation_loader = get_dataloader(train_batch_size, val_batch_size)
    
    model = MNIST_Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = lr)
    
    for epoch in trange(epochs):
        print(epoch)
        for step, (img, label) in enumerate(train_loader):
            img, label = img.view(-1,28*28).to(device), label.to(device)
            
            m_img = model(img)
            
            optimizer.zero_grad()
            loss = criterion(m_img, img)
            loss.backward()
            optimizer.step()

def dict_from_result():
    print(model.state_dict().keys())
    trained_dict = model.state_dict()
    from collections import OrderedDict
    
    enc_state_dict = OrderedDict()
    dec_state_dict = OrderedDict()
    
    for k, v in trained_dict.items():
        if k.startswith('encoder'):
            enc_state_dict[k] = v
    
    for k, v in trained_dict.items():
        if k.startswith('decoder'):
            dec_state_dict[k] = v
    
class MNIST_encoder(nn.Module):
    def __init__(self):
        super(MNIST_encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(True),
            nn.Linear(512, 256),
            nn.ReLU()
            )
    
    def forward(self, x):
        return self.encoder(x)
    
    
class MNIST_generator(nn.Module):
    def __init__(self):
        super(MNIST_generator, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 28*28),
            nn.Sigmoid()
            )
    
    def forward(self, x):
        return self.decoder(x)

def encoder_generator():
    encoder = MNIST_encoder()
    encoder.load_state_dict(enc_state_dict)
    
    generator = MNIST_generator()
    generator.load_state_dict(dec_state_dict)        
            
    for step, (img, label) in enumerate(validation_loader):
        img = img.view(-1, 28*28)
        
        encoded_arr = encoder(img)
        m_img = generator(encoded_arr)
        
        
        t_img = m_img[0].view(28,28).detach().cpu().numpy()
        fig, ax = plt.subplots(figsize = (30,30))
        ax.imshow(t_img, 'gray')
        print(encoded_arr.shape)


class MNIST_MLP(nn.Module):
    def __init__(self, p):
        super(MNIST, self).__init__()
        self.model = nn.Sequential(
                nn.Dropout(p = p),
                nn.Linear(28*28,128),
                nn.ReLU(),
                nn.Dropout(p = p),
                nn.Linear(128,64),
                nn.ReLU(),

                nn.Linear(64,10),
                nn.LogSoftmax(dim = 1)
            )
        
        
    def forward(self, x):
        x = self.model(x)
        return x

def train_MNIST_MLP():
    epochs = 1
    trian_batch_size, val_batch_size = 10, 2000
    lr = 0.001
    cnt = 0
    loss_list = []
    val_acc_list = []
    train_loader, validation_loader = get_dataloader(trian_batch_size,val_batch_size)
    
    model = MNIST(p = 0.3).to(device)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(),lr = lr)
    
    
    
    for epoch in trange(epochs):
        model.train()
        loss_epoch = 0
        for step, (img, label) in enumerate(train_loader):
    
            img, label = img.view(-1,28*28).to(device), label.to(device)
            print(img.shape)
            pred = model(img)
            optimizer.zero_grad()
            loss = criterion(pred,label)
            loss_epoch += loss.item() * pred.shape[0]
            loss.backward()
            optimizer.step()
        loss_epoch /= len(train_loader.dataset)
        loss_list.append(loss_epoch)
        
        model.eval()
        val_acc = 0
        
        for step, (img, label) in enumerate(validation_loader):
            img, label = img.view(-1,28*28).to(device), label.to(device)
            
            pred = model(img)
            topv, topi = pred.topk(1, dim = 1)
            n_correct = (topi.view(-1) == label).type(torch.int)
            val_acc += n_correct.sum().item()
        val_acc /= len(validation_loader.dataset)
        val_acc_list.append(val_acc)
        print(epoch, loss_epoch, val_acc)
    
    
    fig, ax = plt.subplots(2, 1, figsize = (30, 15))
    ax[0].plot(loss_list)
    ax[1].plot(val_acc_list)