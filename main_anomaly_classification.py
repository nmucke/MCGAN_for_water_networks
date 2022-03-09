import pdb
import torch.nn as nn
import torch
from data_handling.gan_dataloaders import get_dataloader
import models.GAN_models as GAN_models
from utils.load_checkpoint import load_checkpoint
from transforms.transform_data import transform_data
from utils.seed_everything import seed_everything
from training.training_GAN import TrainGAN
from utils.graph_utils import get_incidence_mat
import networkx as nx
torch.set_default_dtype(torch.float32)
import pickle
import os
import wntr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.svm import SVC 
from sklearn.utils import shuffle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis



if __name__ == "__main__":
    target_columns = ['leak']
    feature_columns = ['flow_rate_obs_error',
                       'head_obs_error',
                       'critic_score_leak',
                       'critic_score_no_leak',
                       'reservoir_demand_diff',
                       'flow_rate_std',
                       'head_std']

    # Train data
    data_no_leak = pd.read_pickle("training_classification_data_no_leak.pkl")
    data_with_leak = pd.read_pickle("training_classification_data_with_leak.pkl")
    data = pd.concat([data_no_leak, data_with_leak], ignore_index=True)
    data = shuffle(data)

    train_targets = data[target_columns]
    train_features = data[feature_columns]

    # Test data
    data_no_leak = pd.read_pickle("test_classification_data_no_leak.pkl")
    data_with_leak = pd.read_pickle("test_classification_data_with_leak.pkl")
    data = pd.concat([data_no_leak, data_with_leak], ignore_index=True)
    data = shuffle(data)

    test_targets = data[target_columns]
    test_features = data[feature_columns]

    names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        "Gaussian Process",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA",
        ]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
    ]

    for name, clf in zip(names, classifiers):
        
        clf.fit(train_features.to_numpy(), train_targets.to_numpy().squeeze())
        score = clf.score(test_features.to_numpy(), test_targets.to_numpy().squeeze())

        print(f'{name}: {score:0.3f}')


    