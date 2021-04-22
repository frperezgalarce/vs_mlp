
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
from scipy.stats import multivariate_normal as normal
from sklearn import decomposition
from sklearn import manifold
from scipy import stats

def load_files(dataset=1): 
    number = dataset
    fileTrain = '/home/franciscoperez/Documents/GitHub/data/BIASEDFATS/Train_rrlyr-'+str(number)+'.csv'
    fileTest = '/home/franciscoperez/Documents/GitHub/data/BIASEDFATS/Test_rrlyr-'+str(number)+'.csv'
    train_dataset = pd.read_csv(fileTrain, index_col ='Unnamed: 0')
    test_dataset = pd.read_csv(fileTest)
    try:
        train_dataset =  train_dataset.drop(['Pred', 'Pred2', 'h', 'e', 'u','ID'], axis = 1)
        for col in train_dataset.columns:
            if col not in ['label']:
                if train_dataset[col].var()==0:
                    print(col)
                    del train_dataset[col]
        test_dataset = test_dataset[list(train_dataset.columns)]
    except:
        print(col)
        print('---')
    return train_dataset, test_dataset


def delete_outliers(train_dataset, test_dataset): 
    label = train_dataset['label']
    del train_dataset['label']
    train_dataset_z=(train_dataset-train_dataset.mean())/train_dataset.std()
    z_scores = stats.zscore(train_dataset_z)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    train_dataset['label'] = label
    train_dataset = train_dataset[filtered_entries]
    print(train_dataset.shape)
    label = test_dataset['label']
    del test_dataset['label']
    print(test_dataset.shape)
    test_dataset_z=(test_dataset-test_dataset.mean())/test_dataset.std()
    z_scores = stats.zscore(test_dataset_z)
    abs_z_scores = np.abs(z_scores)
    filtered_entries = (abs_z_scores < 3).all(axis=1)
    test_dataset['label'] = label
    print(test_dataset.shape)
    test_dataset = test_dataset[filtered_entries]
    return train_dataset, test_dataset


def sort_columns(data): 
    data = data[['PeriodLS', 'Amplitude', 'AndersonDarling', 'Autocor_length', 'Beyond1Std',
        'CAR_mean', 'CAR_sigma', 'CAR_tau', 'Con', 'Eta_e',
        'FluxPercentileRatioMid20', 'FluxPercentileRatioMid35',
        'FluxPercentileRatioMid50', 'FluxPercentileRatioMid65',
        'FluxPercentileRatioMid80', 'Freq1_harmonics_amplitude_0',
        'Freq1_harmonics_amplitude_1', 'Freq1_harmonics_amplitude_2',
        'Freq1_harmonics_amplitude_3', 'Freq1_harmonics_rel_phase_1',
        'Freq1_harmonics_rel_phase_2', 'Freq1_harmonics_rel_phase_3',
        'Freq2_harmonics_amplitude_0', 'Freq2_harmonics_amplitude_1',
        'Freq2_harmonics_amplitude_2', 'Freq2_harmonics_amplitude_3',
        'Freq2_harmonics_rel_phase_1', 'Freq2_harmonics_rel_phase_2',
        'Freq2_harmonics_rel_phase_3', 'Freq3_harmonics_amplitude_0',
        'Freq3_harmonics_amplitude_1', 'Freq3_harmonics_amplitude_2',
        'Freq3_harmonics_amplitude_3', 'Freq3_harmonics_rel_phase_1',
        'Freq3_harmonics_rel_phase_2', 'Freq3_harmonics_rel_phase_3', 'Gskew',
        'LinearTrend', 'MaxSlope', 'Mean', 'Meanvariance', 'MedianAbsDev',
        'MedianBRP', 'PairSlopeTrend', 'PercentAmplitude',
        'PercentDifferenceFluxPercentile', 'Period_fit', 'Psi_CS',
        'Psi_eta', 'Q31', 'Rcs', 'Skew', 'SlottedA_length', 'SmallKurtosis',
        'Std', 'StetsonK', 'StetsonK_AC', 'StructureFunction_index_21',
        'StructureFunction_index_31', 'StructureFunction_index_32', 'label']]
    return data

def plot_confusion_matrix(labels, pred_labels, ax):
    #fig = plt.figure(figsize = (10, 10));
    #ax = fig.add_subplot(1, 1, 1);
    cm = metrics.confusion_matrix(labels, pred_labels, normalize='pred');
    cm = metrics.ConfusionMatrixDisplay(cm);
    cm.plot(cmap = 'Blues', ax = ax)
    cm.im_.colorbar.remove()

def fast_cdist(x1, x2):
    res=f.mse_loss(x1, x2, size_average=False)
    #res=f.l1_loss(x1, x2, size_average=False)
    return res

def softmax(x): 
    return x.exp() / (x.exp().sum(-1)).unsqueeze(-1)

def custom_loss_auxiliar(output, lambda_pen=1.0):
    #print(softmax(output)[:,1])
    expected_loss = softmax(output)[:,1].mean()
    #print(expected_loss)
    threshold = torch.tensor(0.5).cuda()
    aux_loss = torch.tensor(expected_loss-threshold, requires_grad=True).cuda()
    #print(aux_loss**2)
    return torch.tensor(lambda_pen*aux_loss**2, requires_grad=True).cuda()

def custom_loss(output, labels, weigths=None, weigths_prior=None):
    regularization_loss = 0
    lambda_1 = 0.00001
    if weigths is not None: 
        regularization_loss += fast_cdist(weigths, weigths_prior)
            #print(regularization_loss)
        loss = criterion(outputs, labels) + lambda_1*regularization_loss #nn.L1Loss()(weigths, weigths)
    else:
        loss = criterion(outputs, labels)
        #print(loss)
    return loss


def get_predictions(model, iterator, device):
    model.eval()

    images = []
    labels = []
    probs = []
    with torch.no_grad():

        for (x, y) in iterator:

            x = x.to(device)

            y_pred, _, _ = model(x)

            y_prob = f.softmax(y_pred, dim = -1)
            top_pred = y_prob.argmax(1, keepdim = True)

            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())

    images = torch.cat(images, dim = 0)
    labels = torch.cat(labels, dim = 0)
    probs = torch.cat(probs, dim = 0)

    return images, labels, probs

def regularization_method(params):
    l1_regularization = 0
    l2_regularization = 0
    lambda1 = 0.001
    lambda2 = 0.001
    for param in params:
        l1_regularization += torch.norm(param, 1)**2
        l2_regularization += torch.norm(param, 2)**2
    loss = loss + lambda1*l1_regularization + lambda2*l2_regularization
    
def plot_weights(weights, n_weights):

    rows = int(np.sqrt(n_weights))
    cols = int(np.sqrt(n_weights))

    fig = plt.figure(figsize = (20, 10))
    
    for i in range(rows*cols):
        ax = fig.add_subplot(rows, cols, i+1)
        ax.imshow(weights[i].view(5, 10).cpu().numpy(), cmap = 'bone')
        #plt.title(str(train_target[i]))
        ax.axis('off')
        
def get_pca(data, data_test=None, n_components = 2):
    pca = decomposition.PCA()
    
    pca.n_components = n_components
    pca_data = pca.fit_transform(data)
    
    if data_test is not None: 
        pca_data_test = pca.transform(data_test)
        return pca_data, pca_data_test 
    
    return pca_data

def get_tensors_old(data, batch_size):
    data['label'] = data['label'].str.replace('ClassA', '1')
    data['label'] = data['label'].str.replace('ClassB', '0')
    data['label'] = data['label'].str.replace('Noise', '0.5')
    target = torch.tensor(data['label'].values.astype(np.float32))
    x = torch.tensor(target.drop('label', axis = 1).values.astype(np.float32)) 
    x = f.normalize(x)
    xy = data_utils.TensorDataset(x, target) 
    xy = data_utils.DataLoader(dataset = xy, batch_size = batch_size)

    print('Shape tensor: ', list(xy.size()))
    return data, x, y, xy

def get_tensors(data, batch_size):
    print('____get_tensor_function____')
    data.loc[data.label=='ClassA','label'] = 1
    data.loc[data.label=='ClassB','label'] = 0
    data.loc[data.label=='Noise','label'] = 0.5
    target = torch.tensor(data['label'].values.astype(np.float32))
    x = torch.tensor(data.drop('label', axis = 1).values.astype(np.float32)) 
    x = f.normalize(x)
    xy_ = data_utils.TensorDataset(x, target) 
    xy = data_utils.DataLoader(dataset = xy_, batch_size = batch_size)
    print('shape tensor: ', (x.size()))
    return data, x, target, xy

def generate_samples(samples, train_dataset, epsilon, option = 2): 
    number_columns = train_dataset.shape[1]
    samples1 = samples*2
    data_prior = pd.DataFrame(0, index=np.arange(1), columns=train_dataset.columns)

    
        #option 1
    if option == 1:
        for i in range(samples1):
            new_data = pd.DataFrame(0, index=np.arange(1), columns=train_dataset.columns) 
            new_data.columns = train_dataset.columns
            new_data['PeriodLS']= (np.random.uniform(0.2-epsilon,1.0+epsilon))#-minimum_period)/(maximum_period-minimum_period)
            new_data['label'] = 'Noise'
            frames = [data_prior, new_data]
            data_prior = pd.concat(frames, ignore_index=True)

    if option==2:
        #option 2
        for i in range(samples):
            new_data = pd.DataFrame(0, index=np.arange(1), columns=train_dataset.columns) 
            new_data.columns = train_dataset.columns
            new_data['PeriodLS']=(np.random.uniform(0.2-epsilon,0.2))#-minimum_period)/(maximum_period-minimum_period)
            new_data['label'] = 'Noise'
            frames = [data_prior, new_data]
            data_prior = pd.concat(frames, ignore_index=True)


        for i in range(samples):    
            new_data = pd.DataFrame(0, index=np.arange(1), columns=train_dataset.columns) 
            new_data.columns = train_dataset.columns
            new_data['PeriodLS']=(np.random.uniform(1.0,1.0+epsilon))
            new_data['label'] = 'Noise'
            frames = [data_prior, new_data]
            data_prior = pd.concat(frames, ignore_index=True)


    #option 3
    if option==3:
        for i in range(samples):    
            new_data = pd.DataFrame(0, index=np.arange(1), columns=train_dataset.columns) #pd.DataFrame([train_dataset.sample(1000).mean()]).T
            new_data['PeriodLS']= 1.0
            new_data['label'] = 'Noise'
            frames = [data_prior, new_data]
            data_prior = pd.concat(frames, ignore_index=True)


        for i in range(samples):    
            new_data = pd.DataFrame(0, index=np.arange(1), columns=train_dataset.columns) 
            new_data.columns = train_dataset.columns
            new_data['PeriodLS']= 0.2
            new_data['label'] = 'Noise'
            frames = [data_prior, new_data]
            data_prior = pd.concat(frames, ignore_index=True)
                
    return data_prior

def get_tsne(data, data_test = None, n_components = 2, n_curves = None):
    if n_curves is not None:
        data = data[:n_curves]
    tsne = manifold.TSNE(n_components = n_components, random_state = 0)
    tsne_data = tsne.fit_transform(data)
    
    if data_test is not None: 
        tsne_data_test = tsne.fit_transform(data_test)
        return tsne_data, tsne_data_test  
    
    return tsne_data

def get_representations(net, iterator, device):

    net.eval()
    outputs = []
    intermediates = []
    intermediates2 = []
    labels = []

    with torch.no_grad():
        for (x, y) in iterator:
            x = x.to(device)
            y_pred, h2, h1 = net(x)
            outputs.append(y_pred.cpu())
            intermediates.append(h1.cpu())
            intermediates2.append(h2.cpu())
            labels.append(y)
        
    outputs = torch.cat(outputs, dim = 0)
    intermediates = torch.cat(intermediates, dim = 0)
    intermediates2 = torch.cat(intermediates2, dim = 0)
    labels = torch.cat(labels, dim = 0)

    return outputs, intermediates, intermediates2, labels

def plot_representations(data, labels, ax, n_curves = None):
    if n_curves is not None:
        data = data[:n_curves]
        labels = labels[:n_curves]
    #fig = plt.figure(figsize = (10, 10))
    #ax = fig.add_subplot(111)
    scatter = ax.scatter(data[:, 0], data[:, 1], c = labels, alpha =0.5)
    handles, labels = scatter.legend_elements()
    legend = ax.legend(handles = handles, labels = labels)


def generate_samples_2D(samples, train_dataset):

    data_prior = pd.DataFrame(0, index=np.arange(1), columns=train_dataset.columns)

    class_filtered_upper = train_dataset[(train_dataset.label=='ClassA') & (train_dataset.PeriodLS>0.9)]
    class_filtered_lower = train_dataset[(train_dataset.label=='ClassA') & (train_dataset.PeriodLS<0.3)]

    mean_upper = (class_filtered_upper[['Amplitude', 'PeriodLS']].mean())
    cov_upper =  (class_filtered_upper[['Amplitude', 'PeriodLS']].cov())

    samples_upper = pd.DataFrame(np.random.multivariate_normal(mean_upper, cov_upper, samples), columns=['Amplitude', 'PeriodLS'])


    mean_lower = (class_filtered_lower[['Amplitude', 'PeriodLS']].mean())
    cov_lower =  (class_filtered_lower[['Amplitude', 'PeriodLS']].cov())

    samples_lower = pd.DataFrame(np.random.multivariate_normal(mean_lower, cov_lower, samples), columns=['Amplitude', 'PeriodLS'])


    new_data_upper = pd.DataFrame(0, index=np.arange(samples), columns=train_dataset.columns) 
    new_data_upper.columns = train_dataset.columns
    new_data_upper['PeriodLS']= samples_upper['PeriodLS']
    new_data_upper['Amplitude']= samples_upper['Amplitude']
    new_data_upper['label'] = 'Noise'
    
    new_data_lower = pd.DataFrame(0, index=np.arange(samples), columns=train_dataset.columns) 
    new_data_lower.columns = train_dataset.columns
    new_data_lower['PeriodLS']= samples_upper['PeriodLS']
    new_data_lower['Amplitude']= samples_upper['Amplitude']
    new_data_lower['label'] = 'Noise'
    
    
    frames = [new_data_lower, new_data_upper]
    data_prior = pd.concat(frames, ignore_index=True)


                
    return data_prior