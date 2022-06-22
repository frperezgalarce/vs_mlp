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

sc = '_RRd'


results = []
num_classes = 2
learning_rate = 0.005
samples = 3000
epsilon=0
#for epsilon in [0.1, 0.05, 0.025, 0.15]:
for batch_size in [256]:
    for hidden_size in [100]:
        for aux_loss_activated in [True]:
            for EPS1 in [0.025]:
                for n in [100, 200, 400]:
                    for opt in [1]:
                        for t in range(30):
                            train_dataset, test_dataset = ut.load_files(dataset=1, subclass=sc)
                            input_size = train_dataset.shape[1]-1
                            train_dataset, test_dataset = ut.delete_outliers(train_dataset, test_dataset)
                            if n < 50000:
                                train_dataset = ut.down_sampling(train_dataset)
                                train_dataset = train_dataset.sample(n)
                                print(train_dataset)
                            else: 
                                trainig_dataset_a = train_dataset[train_dataset.label=='ClassA']
                                print('shape: ', trainig_dataset_a.shape[0])
                                n2 = n - trainig_dataset_a.shape[0]
                                print('clase no RR Lrae', n2)
                                trainig_dataset_b = train_dataset[~(train_dataset.label=='ClassA')].sample(n2)
                                train_dataset = pd.concat([trainig_dataset_a, trainig_dataset_b])
                            


                            train_dataset = ut.sort_columns(train_dataset)
                            test_dataset = ut.sort_columns(test_dataset)
                            #train_dataset, test_dataset = ut.normalize(train_dataset, test_dataset)
                            test_dataset_pred = test_dataset.copy()
                            train_dataset_pred = train_dataset.copy()

                            try:
                                data_prior = ut.generate_samples_2D(samples, train_dataset, distribution='gaussian', subclass=True)
                                if train_dataset[train_dataset.label=='ClassB'].shape[0] >= samples:
                                    samples_prior = samples 
                                else: 
                                    samples_prior = train_dataset[train_dataset.label=='ClassB'].shape[0] 

                                data_prior = pd.concat([data_prior, train_dataset[train_dataset.label=='ClassB'].sample(samples_prior)])


                                train_dataset, test_dataset, data_prior = ut.normalize(train_dataset, test_dataset, data_prior)

                                train_dataset, val_dataset = train_test_split(train_dataset, test_size=0.2)

                                train_dataset_prior, val_dataset_prior = train_test_split(data_prior, test_size=0.2)
                                print(train_dataset_prior.columns)
                                _, _, train_target_prior, train_loader_prior = ut.get_tensors(train_dataset_prior, batch_size)
                                _, _, val_target_prior, val_loader_prior     = ut.get_tensors(val_dataset_prior, batch_size)
                                _, _, train_target, train_loader             = ut.get_tensors(train_dataset, batch_size)
                                _, _, train_target_pred, train_loader_pred   = ut.get_tensors(train_dataset_pred, batch_size)
                                _, _, val_target, val_loader                 = ut.get_tensors(val_dataset_prior, batch_size)
                                _, _, test_target, test_loader               = ut.get_tensors(test_dataset, batch_size)
                                _, _, test_target_pred, test_loader_pred     = ut.get_tensors(test_dataset_pred, batch_size)

                                net = Net(input_size, hidden_size, hidden_size, num_classes)
                                net.cuda()

                                hist_val, hist_train = nn.train(net, train_loader, train_loader_prior, val_loader, test_loader,
                                EPS1, learning_rate, input_size, aux_loss_activated=aux_loss_activated)

                                acc_train, recall_train, f1_train = nn.get_results(net, train_loader, input_size)
                                acc_test, recall_test, f1_test  = nn.get_results(net, test_loader, input_size)
                                roc_train = nn.get_roc_curve(net, train_loader, input_size)
                                roc_test = nn.get_roc_curve(net, test_loader, input_size)
                                results.append([acc_train, acc_test,recall_train, recall_test, f1_train, f1_test, roc_train, roc_test, epsilon, batch_size, hidden_size, aux_loss_activated, EPS1, n, opt])
                                pd.DataFrame(results, columns=['acc_train', 'acc_test','recall_train', 'recall_test','f1_train', 'f1_test', 
                                                                   'roc_train', 'roc_test', 'epsilon', 'batch_size', 'hidden_size',
                                     'aux_loss_activated', 'EPS1', 'n', 'opt']).to_csv('05-06-2022-sc-2d_gaussian-RRLyrae.csv')
                            except Exception as e:
                                print(e) 
                                print(str(batch_size)+"-"+str(hidden_size)+"-"+str(aux_loss_activated)+"-"+str(EPS1))