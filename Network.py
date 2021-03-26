
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import torch.nn.functional as f 
from torch.autograd import Variable
torch.backends.cudnn.deterministic = True
import pandas as pd
import numpy as np
import torch.utils.data as data_utils
from sklearn.preprocessing import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn import metrics
import random 
from matplotlib import pyplot as plt
from itertools import cycle
from sklearn import decomposition
from sklearn import manifold
from scipy import stats

class Net(nn.Module):
    def __init__(self, input_size, hidden_size,hidden_size2, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        #self.relu = nn.ReLU()
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size2)  
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, num_classes) 
        #self.sigmoid = nn.Sigmoid()
        #self.dropout = nn.Dropout(p=0.1)
        #self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        #self.batchnorm2 = nn.BatchNorm1d(hidden_size)
        
        
    def forward(self, x):
        out1 = self.fc1(x)
        #out = self.batchnorm1(out)
        #out = self.relu(out)
        out = self.relu(out1)
        out2 = self.fc2(out)
        #out = self.batchnorm1(out)
        out = self.relu2(out)
        out3 = self.fc3(out)
        #x = self.dropout(x)
        #out = self.sigmoid(out)
        return out3, out2, out1


def train(net, train_loader, train_loader_prior, val_loader, EPS1, EPS2, learning_rate, 
            input_size, num_epochs_prior=1000, aux_loss_activated=True): 

    loss_prior = torch.tensor(0)
    hist_train = []
    hist_val = []
    aux_loss_behaviour = []

    criterion = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  
    optimizer_prior = torch.optim.Adam(net.parameters(), lr=learning_rate)   
    locked_masks = {n: torch.abs(w) < EPS1 for n, w in net.named_parameters() if n.endswith('weight')}
    locked_masks2 = {n: torch.abs(w) < EPS2 for n, w in net.named_parameters() if n.endswith('weight')}

    for epoch in range(num_epochs_prior):
        epoch_loss = 0.0
        running_loss = 0.0
        epoch_loss_prior = 0.0    
        running_loss_prior = 0.0
        
        for item1, item2 in zip(train_loader, cycle(train_loader_prior)):
            star_prior, labels_prior = item2
            star, labels = item1
            
            star = Variable(star.view(-1, input_size)).cuda()
            labels = Variable(labels).cuda()
            optimizer.zero_grad()  
            outputs, _, _ = net(star)
            loss = criterion(outputs, labels.long())      
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if aux_loss_activated:
                star_prior = Variable(star_prior.view(-1, input_size)).cuda()
                labels_prior = Variable(labels_prior).cuda()
                optimizer_prior.zero_grad()  # zero the gradient buffer
                outputs_prior, _, _ = net(star_prior)
                #print("-------------------before----------------------")
                aux_loss_behaviour.append(loss_prior.item())
                loss_prior = criterion(outputs_prior, labels_prior.long()) #custom_loss_auxiliar(outputs_prior)      
                #print("-------------------later-----------------------")
                aux_loss_behaviour.append(loss_prior.item())
                loss_prior.backward()
                
                for n, w in net.named_parameters():                                                                                                                                                                           
                    if w.grad is not None and n in locked_masks2:                                                                                                                                                                                   
                        w.grad[locked_masks2[n]] = 0 
                    
                    
                optimizer_prior.step()
                epoch_loss_prior += outputs_prior.shape[0] * loss_prior.item()      
                running_loss_prior += loss_prior.item()
            
        hist_train.append(running_loss/len(train_loader))    
        print('training:', 'epoch: ', str(epoch+1),' loss: ', str(running_loss/len(train_loader)))
        
        epoch_loss = 0.0
        running_loss = 0.0
        for i, (star, labels) in enumerate(val_loader):  
            
            star = Variable(star.view(-1, input_size)).cuda()
            labels = Variable(labels).cuda()
            optimizer.zero_grad()  
            outputs, _, _ = net(star)
            loss = criterion(outputs, labels.long())      
            loss.backward()
            
            for n, w in net.named_parameters():                                                                                                                                                                           
                if w.grad is not None and n in locked_masks:                                                                                                                                                                                   
                    w.grad[locked_masks[n]] = 0 
            
            optimizer.step()
            running_loss += loss.item()
        
        print('validating:', 'epoch: ', str(epoch+1),' loss: ', str(running_loss / len(val_loader)))
        hist_val.append(running_loss / len(val_loader))


def get_results(net, data, input_size):
    correct = 0
    total = 0
    for star, labels in data:
        images = Variable(star.view(-1, input_size)).cuda()
        outputs, _, _ = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted.cpu() == labels.long()).sum()
    print('Accuracy of the network on test objects: %d %%' % (100 * correct / total))
    acc = 100*correct / total
    print(np.asarray(acc))
    return np.asarray(acc)