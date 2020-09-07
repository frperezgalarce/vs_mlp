import sys
import seaborn as sns
import scipy
import numpy as np
import pandas as pd

from scipy.stats import norm
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.decomposition import PCA

pal = sns.light_palette((200, 75, 60), input="husl", as_cmap=True)

CLASS_COLORS = ['#66c2a5',
                '#fc8d62',
                '#8da0cb',
                '#e78ac3',
                '#a6d854',
                '#ffd92f',
                '#e5c494',
                '#b3b3b3']


def plot_gmm(X, mu, W, ax=None):
    assert mu.shape[0] == W.shape[0]
    dims = mu.shape[0]

    if ax:
        ax.scatter(X[:, 0], X[:, 1],
                   alpha=0.5,
                   edgecolor='black', linewidth=0.15)
    else:
        plt.scatter(X[:, 0], X[:, 1],
                    alpha=0.5,
                    edgecolor='black', linewidth=0.15)
    min_x, min_y = np.amin(X, axis=0)
    max_x, max_y = np.amax(X, axis=0)
    x, y = np.mgrid[min_x:max_x:0.1, min_y:max_y:0.1]
    z = np.zeros(x.shape + (2,))
    z[:, :, 0] = x;
    z[:, :, 1] = y
    for i in range(mu.shape[0]):
        f_z = scipy.stats.multivariate_normal.pdf(z, mu[i, :], W[i, :])
        if ax:
            ax.contour(x, y, f_z, antialiased=True, cmap=pal)
        else:
            plt.contour(x, y, f_z, antialiased=True, cmap=pal)


def clean_data(data):
    try:
        data = data.loc[:, ~data.columns.str.contains('^Unnamed')]
        data = data.drop(['Pred', 'Pred2', 'h', 'e', 'u', 'ID'], axis=1)
    except:
        print('---')
    names = data.columns
    scaler = preprocessing.StandardScaler()
    data = scaler.fit_transform(data)
    data = pd.DataFrame(data, columns=names)
    return data


def kl_divergence(p, q):
    epsilon = 0.00001
    p = p + epsilon
    q = q + epsilon
    kl = p * np.log(p / q)
    return kl


def pca_reduction(data, components=2):
    pca = PCA(n_components=components)
    pca.fit(data)
    data = pca.transform(data)
    data = pd.DataFrame(data)
    return data


def plot_gmm_obs(X, C, title='', ax=None):
    xlabel = 'x'
    ylabel = 'y'
    components = np.max(C) + 1
    for k in range(components):
        obs_of_k_component = np.where(C == k)[0]
        if ax:
            ax.scatter(X[obs_of_k_component, 0], X[obs_of_k_component, 1],
                       facecolor=CLASS_COLORS[k], alpha=0.5,
                       edgecolor='black', linewidth=0.15)
        else:
            plt.scatter(X[obs_of_k_component, 0], X[obs_of_k_component, 1],
                        facecolor=CLASS_COLORS[k], alpha=0.5,
                        edgecolor='black', linewidth=0.15)
    if ax:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    else:
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
