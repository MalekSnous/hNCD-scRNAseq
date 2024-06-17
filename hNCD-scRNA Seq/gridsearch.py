
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.distributions import multivariate_normal as mn
import os
import random
from load_data import load_all_data

from algorithm_mapping import search_supervision, mapping, inv_mapping

from model import  CV_gridsearch, TEST

#%%   Cross Val Gridsearch here
save = False
PATH = os.getcwd()
dataset = 'half7'
dataset_liste = ['half7',
                 # 'binary4',
                 # 'binary6',
                 # 'Paul',
                 # 'Planaria'
                 ]
proportion_u = 0.1
method = 'hmodel'
t = 0
#%%
all_results = []
for dataset in dataset_liste:
    for proportion_u in [0.1,0.25,0.5]:

        all_results_t = []

        for t in range(5):

            (X, X_train, y_train, X_test, y_test,
             X_s_train, X_s_test, y_s_train, y_s_test,
             X_u_train, X_u_test, y_u_train, y_u_test,
             set_y_s, set_y_u, C, matrix_parents, matrix_ascendant_descendant) = load_all_data(PATH=PATH, dataset=dataset, t=t,
                                                                                               proportion_u=proportion_u)

            Xs, Xu = torch.cat((X_s_train, X_s_test), dim=0), torch.cat((X_u_train, X_u_test), dim=0)
            ys, yu = y_s_train + y_s_test, y_u_train + y_u_test
            d, c = X.shape[1], C.shape[0],
            set_y = list(range(c))
            n_components = c

            all_result_v  = CV_gridsearch(method=method,
                                          X_s_train=X_s_train,
                                          y_s_train=y_s_train,
                                          X_u_train=X_u_train,
                                          y_u_train=y_u_train,
                                          C=C,
                                          matrix_parents=matrix_parents,
                          matrix_ascendant_descendant=matrix_ascendant_descendant)
            all_result_v_mean = np.mean(all_result_v,axis=0)
            best_params_f1 = all_result_v_mean[:,0].argmax(axis=0)
            best_param_acc = all_result_v_mean[:,5].argmax(axis=0)

            array_test = TEST(method=method,
                              liste_hyperparams=[best_params_f1, best_param_acc],
                              X_s_train=X_s_train,
                              y_s_train=y_s_train,
                              X_u_train=X_u_train,
                              y_u_train=y_u_train,
                              X_s_test=X_s_test,
                              y_s_test=y_s_test,
                              X_u_test=X_u_test,
                              y_u_test=y_u_test,
                              C=C,
                              matrix_parents=matrix_parents,
                              matrix_ascendant_descendant=matrix_ascendant_descendant)

            all_results_t.append(array_test)
        # print('T : ', t, np.diag(array_test))

        array_results = np.array([all_results_t[i].tolist()[0] for i in range(5)])

        print(method, dataset, proportion_u)
        print(array_results.mean(axis=0))
        print(array_results.std(axis=0))


        all_results.append(all_results_t)

        if save :
            np.save(PATH + '/' + str(method) + '_' + str(dataset) + '_' + str(proportion_u), all_results[-1])
















































