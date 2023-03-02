
import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from sklearn.metrics import roc_curve, roc_auc_score
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
import matplotlib
matplotlib.rcParams["figure.dpi"] = 100
from itertools import cycle
from sklearn import decomposition
from sklearn import manifold
from scipy import stats
from matplotlib.lines import Line2D
dropout=False
batcn_norm = True
deterministic = False
if deterministic:
    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
torch.backends.cudnn.benchmark = True

class Net(nn.Module):
    def __init__(self, input_size, hidden_size,hidden_size2, num_classes):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        #self.relu = nn.ReLU()
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, hidden_size2)  
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(hidden_size2, num_classes) 
        if dropout:
            self.dropout = nn.Dropout(p=0.1) #

        if batcn_norm: 
            self.batchnorm1 = nn.BatchNorm1d(hidden_size)
            self.batchnorm2 = nn.BatchNorm1d(hidden_size)
        self.sigmoid = nn.Sigmoid()
        #self.dropout = nn.Dropout(p=0.1)
        #self.batchnorm1 = nn.BatchNorm1d(hidden_size)
        #self.batchnorm2 = nn.BatchNorm1d(hidden_size)
        
        
    def forward(self, x):
        out1 = self.fc1(x)
        if batcn_norm:
            out1 = self.batchnorm1(out1)
        #out = self.relu(out)
        out2 = self.relu(out1)
        out2 = self.fc2(out2)
        if batcn_norm:
            out2 = self.batchnorm1(out2)
        out3 = self.relu2(out2)
        out3 = self.fc3(out3)
        if dropout:
            out3 = self.dropout(out3) #test
        out3 = self.sigmoid(out3)
        return out3, out2, out1

def train(net, train_loader, train_loader_prior, val_loader, test_loader, EPS1, learning_rate, 
            input_size, num_epochs_prior=1500, aux_loss_activated=True, patience = 10, model_number=1, size = 1000): 
    
    for param in net.parameters():
        param.requires_grad = True
    
    l1_activated = False
    l2_activated = False
    lambda_pen = 0.001
    trigger_times = 0
    verbose = True
    dynamic = True
    epochs_to_learn_capacity = 50
    loss_prior = torch.tensor(0)
    hist_train = []
    hist_val = []
    hist_prior = []
    criterion = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  
    optimizer_prior = torch.optim.Adam(net.parameters(), lr=0.5*learning_rate)

    locked_masks2 = {n: (torch.abs(w) > EPS1) | (n.endswith('bias') | ("batch" in str(n)))  for n, w in net.named_parameters()}
    locked_masks = {n: torch.abs(w) <= EPS1 for n, w in net.named_parameters()}
    
    running_loss_val = 0.0
    for epoch in range(num_epochs_prior):
        if dynamic and (num_epochs_prior<epochs_to_learn_capacity): 
                locked_masks = {n: (torch.abs(w) > EPS1) |  (n.endswith('bias'))for n, w in net.named_parameters() }
                locked_masks2 = {n: torch.abs(w) < EPS1 for n, w in net.named_parameters()}     
        the_last_loss = running_loss_val
        running_loss = 0.0
        running_loss_prior = 0.0
        
        if aux_loss_activated:
            for item1, (star, labels) in enumerate(train_loader): 
                star = Variable(star.view(-1, input_size), requires_grad=True).cuda()
                labels = Variable(labels, requires_grad=True).cuda()
                outputs, _, _ = net(star)
                loss = criterion(outputs, labels.long())
                loss.backward(retain_graph=True)
                torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0, norm_type=2)
                for n, w in net.named_parameters():                                                                                                                                                              
                    if w.grad is not None and n in locked_masks:                                                                                                                                                                                   
                        w.grad[locked_masks[n]] = 0.0                                                             
                optimizer.step()
                optimizer.zero_grad()  
                running_loss += loss.item()   

            for item1, (star_prior, labels_prior) in enumerate(train_loader_prior):
                star_prior = Variable(star_prior.view(-1, input_size), requires_grad=True).cuda()
                labels_prior = Variable(labels_prior, requires_grad=True).cuda()
                outputs_prior, _, _ = net(star_prior)
                loss_prior = criterion(outputs_prior, labels_prior.long()) 
                loss_prior.backward(retain_graph=True)

                for n, w in net.named_parameters():                                                                                                                                                                           
                    if w.grad is not None and n in locked_masks2:                                                                                                                                                                                   
                        w.grad[locked_masks2[n]] = 0.0
                optimizer_prior.step()
                optimizer_prior.zero_grad() 
                running_loss_prior += loss_prior.item()
        else:
            running_loss_prior = 0 
            for item1, (star, labels) in enumerate(train_loader): 
                star = Variable(star.view(-1, input_size)).cuda()
                labels = Variable(labels).cuda()
                outputs, _, _ = net(star)
                loss = criterion(outputs, labels.long()) 
                
                if l2_activated:
                    l2 = sum(torch.norm(p, 2) for p in net.parameters())
                    loss = lambda_pen*l2 + loss
                    loss.backward()
                elif l1_activated:
                    l1 = sum(torch.norm(p, 1) for p in net.parameters())
                    loss = lambda_pen*l1 + loss
                    loss.backward()
                else: 
                    loss.backward()
                optimizer.step()
                optimizer.zero_grad()  
                running_loss += loss.item()   

                optimizer.zero_grad()  
                running_loss += loss.item()                    
        hist_train.append(running_loss)
        hist_prior.append(running_loss_prior)

        print('training:', 'epoch: ', str(epoch+1),' loss: ', str(running_loss), '-- aux loss: ', str(running_loss_prior))    
        
        if ((epoch+1)%10==0):
            running_loss_val = 0.0 
            for i, (star, labels) in enumerate(val_loader):      
                star = Variable(star.view(-1, input_size)).cuda()
                labels = Variable(labels).cuda()
                outputs, _, _ = net(star)
                loss = criterion(outputs, labels.long())      
            running_loss_val += loss.item()       
            print('the_last_loss: ', the_last_loss)
            print('running_loss_val: ', running_loss_val)
            print('validating:', 'epoch: ', str(epoch+1),' loss: ', str(running_loss_val))        
            hist_val.append(running_loss / len(val_loader))
        
            # Early stopping
            print('The current loss:', running_loss_val)
            print('the_last_loss:', the_last_loss)
            if running_loss_val >= the_last_loss:
                trigger_times += 1
                print('trigger times:', trigger_times)
                if trigger_times >= patience:
                    print('Early stopping!\nStart to test process.')
                    id_model = str(model_number)+'_'+str(aux_loss_activated)+'_'+ str(size)
                    torch.save({'model'+str(id_model): net.state_dict()}, 'model_'+str(id_model) +'.pt')
                    return hist_val, hist_train, hist_prior
            else:
                print('trigger times: 0')
                trigger_times = 0
            if verbose:
                        print('sum mask2 - L1: ', str(locked_masks2['fc1.weight'].sum()))
                        print('sum mask2 - L2: ', str(locked_masks2['fc2.weight'].sum()))
                        print('sum mask2 - L3: ', str(locked_masks2['fc3.weight'].sum()))
                        print('sum mask1 - L1 (aux): ', str(locked_masks['fc1.weight'].sum()))
                        print('sum mask1 - L2 (aux): ', str(locked_masks['fc2.weight'].sum()))
                        print('sum mask1 - L3 (aux): ', str(locked_masks['fc3.weight'].sum()))
    id_model = str(model_number)+'_'+str(aux_loss_activated)+'_'+ str(size)
    torch.save({'model'+str(id_model): net.state_dict()}, 'model_'+str(id_model) +'.pt')
    return hist_val, hist_train, hist_prior

def get_results(net, data, input_size):
    target_true=0
    predicted_true=0
    correct_true=0
    for star, labels in data:
        light_curve = Variable(star.view(-1, input_size)).cuda()
        outputs, _, _ = net(light_curve)
        predicted_classes = torch.argmax(outputs.cpu(), dim=1) == 1
        target_classes = labels.data
        target_true += torch.sum(target_classes == 1).float().numpy()
        predicted_true += torch.sum(predicted_classes).float().numpy()
        correct_true += torch.sum(target_classes + predicted_classes.float() == 2).float()
    recall = correct_true / target_true
    precision = correct_true / (predicted_true+10e-10)
    f1_score = 2 * precision * recall / (precision + recall)
    print('recall')
    print(recall)
    print('precision')
    print(precision)
    print('f1_score')
    print(f1_score)
    print('Accuracy of the network on test objects: %d %%' % (100. * precision))
    acc = 100.*precision
    print(np.asarray(acc))
    return np.asarray(acc), np.asarray(recall), np.asarray(f1_score)

def get_results_prob(net, data, input_size):
    pred_prob = []
    labels_gt = []
    for star, labels in data:
        light_curve = Variable(star.view(-1, input_size)).cuda()
        outputs, _, _ = net(light_curve)
        pred_prob.append(outputs)
        labels_gt.append(labels)
    return pred_prob,labels_gt

def get_roc_curve(net, test_loader, input_size, name="name", title="Baseline model"):

    pred_soft, labels =get_results_prob(net, test_loader, input_size)
    y_pred = []
    y_true = []
    for j in range(len(pred_soft)):
        for i in range(len(pred_soft[j][:,1])):
            #print(pred_soft[j][i,1].item())
            y_pred.append(pred_soft[j][i,1].item())
            y_true.append(labels[j][i])
    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    '''optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = np.round(thresholds[optimal_idx],4)
    
    plt.figure(1)
    lw = 2
    plt.plot(fpr, tpr, color="darkorange", label=title+" (area = {:.3f})".format(roc_auc))
    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.vlines(optimal_threshold,0,1, color='black')
    print(optimal_threshold)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc="lower right")
    plt.savefig(name+'roc.png')
    plt.show()'''
    return roc_auc

def plot_grad_flow(named_parameters):
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    Usage: Plug this function in Trainer class after loss.backwards() as 
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow'''
    ave_grads = []
    max_grads= []
    layers = []
    for n, p in named_parameters:
        if(p.requires_grad) and ("bias" not in n):
            print(n)
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k" )
    plt.xticks(range(0,len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom = -0.001, top=0.02) # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])

def train_mask_opt(net, train_loader, train_loader_prior, val_loader, test_loader, EPS1, learning_rate, 
            input_size, num_epochs_prior=10000, aux_loss_activated=True, patience = 10): 
    
    trigger_times = 0
    verbose = True
    dynamic = True
    epochs_to_learn_capacity = 100
    loss_prior = torch.tensor(0)
    hist_train = []
    hist_val = []
    aux_loss_behaviour = []
    criterion = nn.CrossEntropyLoss() 
    criterion2 = nn.L1Loss(reduction='mean') 
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)  
    optimizer_prior = torch.optim.Adam(net.parameters(), lr=learning_rate*0.5)   
    locked_masks2 = {n: torch.abs(w) > EPS1 for n, w in net.named_parameters() if n.endswith('weight')}
    locked_masks = {n: torch.abs(w) < EPS1 for n, w in net.named_parameters() if n.endswith('weight')}

    print('Epochs: ', str(num_epochs_prior))
    the_current_loss = 0.0
    for epoch in range(num_epochs_prior):
        if dynamic and (num_epochs_prior<epochs_to_learn_capacity): 
                #print(torch.quantile(net.named_parameters(), 0.1))
                locked_masks2 = {n: torch.abs(w) > EPS1 for n, w in net.named_parameters() if n.endswith('weight')}
                locked_masks = {n: torch.abs(w) < EPS1 for n, w in net.named_parameters() if n.endswith('weight')}
        print('Epoch: ', str(epoch))
        
        the_last_loss = the_current_loss
        the_current_loss = 0.0
        running_loss = 0.0
        running_loss_prior = 0.0
        
        if aux_loss_activated:
            for item1, (star, labels) in enumerate(train_loader): 
                star = Variable(star.view(-1, input_size)).cuda()
                labels = Variable(labels).cuda()
                optimizer.zero_grad()  
                outputs, _, _ = net(star)
                loss = criterion(outputs, labels.long())      
                loss.backward()
                for n, w in net.named_parameters():                                                                                                                                                                           
                    if w.grad is not None and n in locked_masks2:                                                                                                                                                                                   
                        w.grad[locked_masks2[n]] = 0

                            
                optimizer.step()
                running_loss += loss.item()    

                for item1, (star_prior, labels_prior) in enumerate(train_loader_prior):
                    star_prior = Variable(star_prior.view(-1, input_size)).cuda()
                    labels_prior = Variable(labels_prior).cuda()
                    optimizer_prior.zero_grad()  # zero the gradient buffer
                    outputs_prior, _, _ = net(star_prior)
                    aux_loss_behaviour.append(loss_prior.item())
                    loss_prior = criterion(outputs_prior, labels_prior.long()) #criterion2(outputs_prior[:,0], labels_prior) #criterion(outputs_prior, labels_prior.long()) ...cambiar Noise por 1.0      
                    aux_loss_behaviour.append(loss_prior.item())
                    loss_prior.backward()
                    
                    for n, w in net.named_parameters():                                                                                                                                                                           
                        if w.grad is not None and n in locked_masks:                                                                                                                                                                                   
                            w.grad[locked_masks[n]] = 0
                    
                    optimizer_prior.step()
                    running_loss_prior += loss_prior.item()
        else: 
            for item1, (star, labels) in enumerate(train_loader): 
                star = Variable(star.view(-1, input_size)).cuda()
                labels = Variable(labels).cuda()
                optimizer.zero_grad()  
                outputs, _, _ = net(star)
                loss = criterion(outputs, labels.long())      
                loss.backward()   
                optimizer.step()
                running_loss += loss.item()    
                                
        hist_train.append(running_loss)    
        print('training:', 'epoch: ', str(epoch+1),' loss: ', str(running_loss), '-- aux loss: ', str(running_loss_prior))
        
        running_loss_val = 0.0
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
            running_loss_val += loss.item()
        
       
        print('validating:', 'epoch: ', str(epoch+1),' loss: ', str(running_loss_val))
        
        hist_val.append(running_loss / len(val_loader))

        
         # Early stopping
        the_current_loss = running_loss_val

        if ((epoch+1)%10==0):  
            print('The current loss:', the_current_loss)
            print('the_last_loss:', the_last_loss)
            if the_current_loss > the_last_loss:
                trigger_times += 1
                print('trigger times:', trigger_times)

                if trigger_times >= patience:
                    print('Early stopping!\nStart to test process.')
                    return hist_val, hist_train

            else:
                print('trigger times: 0')
                trigger_times = 0
            
            acc_train = get_results(net, train_loader, input_size)
            acc_val =get_results(net, val_loader, input_size)
            acc_test = get_results(net, test_loader, input_size)

            if verbose:
                        print('sum mask2 - L1: ', str(locked_masks2['fc1.weight'].sum()))
                        print('sum mask2 - L2: ', str(locked_masks2['fc2.weight'].sum()))
                        print('sum mask2 - L3: ', str(locked_masks2['fc3.weight'].sum()))
                        print('sum mask1 - L1: ', str(locked_masks['fc1.weight'].sum()))
                        print('sum mask1 - L2: ', str(locked_masks['fc2.weight'].sum()))
                        print('sum mask1 - L3: ', str(locked_masks['fc3.weight'].sum()))

    return hist_val, hist_train