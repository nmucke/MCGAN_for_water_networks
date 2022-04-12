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
import pdb
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
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier, Pool, cv

from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder

import ray

@ray.remote
def classifier_score(classifier, train_features, train_targets,
                     test_features, test_targets):

    classifier.fit(train_features.to_numpy(), train_targets.to_numpy().squeeze())
    score = classifier.score(test_features.to_numpy(),
                      test_targets.to_numpy().squeeze())

    return score

def get_leak_location_list(train=True,
                           num_samples=1000,
                           clustered=False):
    if train:
        data_path = 'data/training_data_with_leak/network_'
    else:
        data_path = 'data/test_data_with_leak/network_'
    leak_list = []
    if clustered:
        leak_intervals = [
            [1, 2, 3, 19, 18, 20],
            [4, 5, 6, 7, 8],
            [9, 13, 14, 15, 16, 17, 10, 11, 12, 28],
            [27, 26, 34, 25, 24, 23, 29],
            [33, 32, 31, 30],
            [21, 22]
        ]

        for i in range(num_samples):
            true_data_dict = nx.read_gpickle(data_path+str(i))
            leak_loc = true_data_dict['leak']['pipe']
            for j, leak_area in enumerate(leak_intervals):
                if leak_loc in leak_area:
                    leak_list.append(j+1)
    else:
        for i in range(num_samples):
            true_data_dict = nx.read_gpickle(data_path+str(i))
            leak_loc = true_data_dict['leak']['pipe']
            leak_list.append(leak_loc)
    leak_list = np.asarray(leak_list).reshape(-1, )

    return leak_list

if __name__ == "__main__":

    feature_columns_to_drop = ['Unnamed: 0']
    for i in range(7):
        string_flow = 'flow_rate_std_' + str(i)
        string_head = 'head_std_' + str(i)
        feature_columns_to_drop.append(string_flow)
        feature_columns_to_drop.append(string_head)

    data_with_leak = pd.read_csv("training_classification_data_with_leak_sensors.csv")
    data_with_leak = shuffle(data_with_leak)
    train_leak_features = data_with_leak.drop(columns=feature_columns_to_drop)

    data_with_leak = pd.read_csv("test_classification_data_with_leak_sensors.csv")
    data_with_leak = shuffle(data_with_leak)
    test_leak_features = data_with_leak.drop(columns=feature_columns_to_drop)

    train_leak_targets = get_leak_location_list(train=True,
                                                num_samples=train_leak_features.shape[0],
                                                clustered=True)

    test_leak_targets = get_leak_location_list(train=False,
                                                num_samples=test_leak_features.shape[0],
                                                clustered=True)

    leak_location_model = CatBoostClassifier(verbose=0)
    leak_location_model.fit(train_leak_features.to_numpy(), train_leak_targets)

    score = leak_location_model.score(test_leak_features.to_numpy(),
                                      test_leak_targets)

    preds = leak_location_model.predict(test_leak_features.to_numpy())
    cm = confusion_matrix(test_leak_targets, preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=[1, 2, 3, 4, 5, 6])
    disp.plot()
    plt.title(f'Accuracy={score:0.3f}')
    plt.savefig('confusion')
    plt.show()
    pdb.set_trace()

    target_columns = ['leak']

    # Train data
    data_no_leak = pd.read_csv("training_classification_data_no_leak_sensors.csv")
    data_with_leak = pd.read_csv("training_classification_data_with_leak_sensors.csv")
    data = pd.concat([data_no_leak, data_with_leak], ignore_index=True)
    data = shuffle(data)
    data = data.drop(columns=feature_columns_to_drop)

    train_targets = data[target_columns]
    train_features = data.loc[:, data.columns != target_columns[0]]

    #train_targets = train_targets[0:20]
    #train_features = train_features[0:20]

    # Test data
    data_no_leak = pd.read_csv("test_classification_data_no_leak_sensors.csv")
    data_with_leak = pd.read_csv("test_classification_data_with_leak_sensors.csv")
    data = pd.concat([data_no_leak, data_with_leak], ignore_index=True)
    data = shuffle(data)
    data = data.drop(columns=feature_columns_to_drop)

    test_targets = data[target_columns]
    test_features = data.loc[:, data.columns != target_columns[0]]

    '''
    names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        #"Gaussian Process",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA",
        "Gradient Boost",
        "CatBoost"
        ]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(gamma=2, C=1),
        #GaussianProcessClassifier(1.0 * RBF(1.0)),
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
        MLPClassifier(alpha=1, max_iter=100, hidden_layer_sizes=(32, 64, 64, 32), verbose=False, n_iter_no_change=50),
        AdaBoostClassifier(),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        GradientBoostingClassifier(),
        CatBoostClassifier(verbose=0)
    ]
    ray.init(num_cpus=len(classifiers))
    scores = ray.get([classifier_score.remote(classifier,
                                              train_features,
                                              train_targets,
                                              test_features,
                                              test_targets)
                     for classifier in classifiers])
    '''
    '''
    scores = [classifier_score(classifier,
                               train_features,
                               train_targets,
                               test_features,
                               test_targets)
                     for classifier in classifiers]
    '''
    '''
    for name, score in zip(names, scores):
        print(f'{name}: {score:0.3f}')

    best_model_idx = np.argmax(scores)
    best_score = scores[best_model_idx]
    best_model = classifiers[best_model_idx]
    '''
    best_model = CatBoostClassifier(verbose=0)
    best_model.fit(train_features.to_numpy(), train_targets.to_numpy().squeeze())
    score = best_model.score(test_features.to_numpy(),
                             test_targets.to_numpy().squeeze())
    preds = best_model.predict(test_features.to_numpy())
    cm = confusion_matrix(test_targets.to_numpy().squeeze(), preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=['No Leak', 'Leak'])
    disp.plot()
    #plt.title(f'{names[best_model_idx]}, accuracy={best_score:0.3f}')
    plt.title(f'accuracy={score:0.3f}')
    plt.savefig('confusion')
    plt.show()



    forest = GradientBoostingClassifier()
    forest.fit(train_features, train_targets)

    result = permutation_importance(forest, test_features.to_numpy(),
                                    test_targets.to_numpy().squeeze(),
                                    n_repeats=10, random_state=42, n_jobs=2)
    forest_importances = pd.Series(result.importances_mean,
                                   index=list(test_features.keys()))

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.savefig('feature_importance')
    plt.show()




    '''
    cbc = CatBoostClassifier(verbose=0)
    grid = {'max_depth': [3, 4, 5, 6, 7, 8],
            'n_estimators': [100, 150, 200, 250, 300, 350, 400]}
    gscv = GridSearchCV(estimator=cbc, param_grid=grid, scoring='accuracy',
                        cv=5, n_jobs=5)
    gscv.fit(train_features, train_targets)

    print(gscv.best_estimator_)
    print(gscv.best_score_)
    print(gscv.best_params_)
    pdb.set_trace()
    '''


    '''
    catboost = CatBoostClassifier()
    catboost.fit(train_features.to_numpy(), train_targets.to_numpy().squeeze())

    feature_importance = catboost.get_feature_importance(
            Pool(train_features, train_targets, cat_features=[1,2]))

    catboost.get_feature_importance(data=test_features.to_numpy())

    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)

    forest_importances = pd.Series(importances, index=feature_columns)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.show()
    '''
