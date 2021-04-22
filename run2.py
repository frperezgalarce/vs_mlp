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
from sklearn import decomposition
from sklearn import manifold
from scipy import stats
from itertools import cycle
import sys
import utilities as ut
from Network import Net
import Network as nn
SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


results = []

n = 5000#train_dataset.shape[0] 
epsilon = 0.1

hidden_size = 500
hidden_size2 = 500
num_classes = 2
num_epochs = 200
batch_size = 128
learning_rate = 0.001
learning_rate2 = 0.001
regularization = False
add_DR_based_data = True

samples = 10000

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True



#for epsilon in [0.1, 0.05, 0.025, 0.15]:
for batch_size in [128, 256, 512]:
    for hidden_size in [4, 8]:
        for aux_loss_activated in [True, False]:
            for EPS1 in [1e-3,1e-2, 1e-4, 1e-5]:
                for n in [100000, 200000,300000]:
                    for opt in [1]:
                        train_dataset, test_dataset = ut.load_files(dataset=1)
                        input_size = train_dataset.shape[1]-1
                        train_dataset = train_dataset.sample(n)
                        train_dataset, test_dataset = ut.delete_outliers(train_dataset, test_dataset)

                        train_dataset = ut.sort_columns(train_dataset)
                        test_dataset = ut.sort_columns(test_dataset)

                        test_dataset_pred = test_dataset.copy()
                        train_dataset_pred = train_dataset.copy()
                        try:
                            data_prior = ut.generate_samples_2D(samples, train_dataset)

                            train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.1, random_state=42)

                            train_dataset_prior, val_dataset_prior = train_test_split(data_prior, test_size=0.1, random_state=42)

                            _, _, train_target_prior, train_loader_prior = ut.get_tensors(train_dataset_prior, batch_size)
                            _, _, val_target_prior, val_loader_prior     = ut.get_tensors(val_dataset_prior, batch_size)
                            _, _, train_target, train_loader             = ut.get_tensors(train_dataset, batch_size)
                            _, _, train_target_pred, train_loader_pred   = ut.get_tensors(train_dataset_pred, batch_size)
                            _, _, val_target, val_loader                 = ut.get_tensors(val_dataset_prior, batch_size)
                            _, _, test_target, test_loader               = ut.get_tensors(test_dataset, batch_size)
                            _, _, test_target_pred, test_loader_pred     = ut.get_tensors(test_dataset_pred, batch_size)

                            net = Net(input_size, hidden_size, hidden_size, num_classes)
                            net.cuda()

                            #aux_loss_activated = True
                            #num_epochs_prior = 2000
                            #EPS1 = 1e-3
                            EPS2 = 1e-6

                            hist_val, hist_train = nn.train(net, train_loader, train_loader_prior, val_loader,
                                                   EPS1, EPS2, learning_rate, input_size, aux_loss_activated=aux_loss_activated)

                            acc_train = nn.get_results(net, train_loader, input_size)
                            acc_test =nn.get_results(net, test_loader, input_size)
                            results.append([acc_train, acc_test, epsilon, batch_size, hidden_size, aux_loss_activated, EPS1, n, opt])
                            pd.DataFrame(results, columns=['acc_train', 'acc_test', 'epsilon', 'batch_size', 'hidden_size',
                                'aux_loss_activated', 'EPS1', 'n', 'opt']).to_csv('20-04-2021-results_2d.csv')
                        except: 
                            print(str(epsilon)+"-"+str(batch_size)+"-"+str(hidden_size)+"-"+str(aux_loss_activated)+"-"+str(EPS1))