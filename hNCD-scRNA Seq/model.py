import numpy as np
import pandas as pd
import umap
import torch
import torch.nn as nn
import torch.nn.functional as F


from  torch.distributions import multivariate_normal as mn



from sklearn.metrics import f1_score, normalized_mutual_info_score
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment as linear_assignment


import random
import itertools
from joblib import Parallel, delayed

from algorithm_mapping import breadth_first_search, search_supervision, search_supervision_multiple ,mapping, inv_mapping, bfs_parallel


############################################################"
def split_val(X, y, c, proportion_u=0.2):
    set_y_u = random.sample([lab for lab in set(y)], k=int(proportion_u * c))
    set_y_s = list(set(range(c)) - set(set_y_u))

    ys, yu = np.array(y)[np.isin(y, set_y_s)].tolist(), np.array(y)[np.isin(y, set_y_u)].tolist()
    Xs, Xu = X[np.isin(y, set_y_s)], X[np.isin(y, set_y_u)]

    return Xs, Xu, ys, yu


def partition(list_in, n):
    random.shuffle(list_in)
    return [list_in[i::n] for i in range(n)]


def split_val_cv(X, y, v, split_label):
    set_y_u = split_label[v]
    set_y_s = list(set(y) - set(set_y_u))
    ys, yu = np.array(y)[np.isin(y, set_y_s)].tolist(), np.array(y)[np.isin(y, set_y_u)].tolist()
    Xs, Xu = X[np.isin(y, set_y_s)], X[np.isin(y, set_y_u)]
    return Xs, Xu, ys, yu



def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(len(y_pred)):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return (sum([w[i, j] for i, j in zip(ind[0], ind[1])  ]) * 1.0 / y_pred.size).item()

def cluster_acc_hierarchical(y_true, y_pred, C):
    """
    Calculate sum of distance after clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        sum in [0,1]
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size

    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(len(y_pred)):
        w[y_pred[i], y_true[i]] += 1
    ind = linear_assignment(w.max() - w)
    return (sum([C[i, j] for i, j in zip(ind[0], ind[1])  ]) * 1.0 ).item()

def hierarchical_pairwise_error(y_true, y_pred, C):
    L = np.sum([[np.abs(C[i,j] - C[i_pred,j_pred]) for i,i_pred in zip(y_true, y_pred)] for j, j_pred in zip(y_true, y_pred)])
    return L/(len(y_pred))**2

def hierarcical_dist(y_true, y_pred, C):
    l_tot = 0
    return l_tot


def compute_matrix_parents(C, root_indice=0):
    c = C.shape[0]
    matrix_parents = torch.eye(c)
    list_parents = [[root_indice]]
    for label in range(1, c):
        liste_to_root = [label]
        dist_to_root = C[label, 0]
        # find parent
        new_label = label

        while C[new_label, root_indice] > 0:

            cols = torch.where(C[new_label] == 1)[0]
            # add parent to the list

            if len(cols) > 1:
                new_label = cols[np.argmin(C[root_indice, cols])].detach().tolist()
                liste_to_root.append(new_label)
            else:
                new_label = cols.detach().tolist()[0]
                liste_to_root.append(new_label)

            matrix_parents[label, new_label] = 1
        list_parents.append(liste_to_root)

    return matrix_parents
#%%  INIT  + CV GRIDSEARCH
####################""


def find_dict_params(method, matrix_parents,c):
    if method == 'hPhi' or 'hmethod':
        dict_params = {'lr': 1e-1, 'lambda_eps': 1e-2, 'lambdaa_u': 1e-1, 'matrix_parents': matrix_parents}
    if method == 'Phi':
        dict_params = {'lr': 1e-1, 'lambda_eps': 1e-2, 'lambdaa_u': 1e-1, 'matrix_parents': torch.eye(c)}
    if method == 'hGMM':
        dict_params = {'matrix_parents': matrix_parents, 'lambdaa': 5e2, 'lr': 5e-2, 'sub_epoch': 20
                       }
    if method == 'Autonovel':
        dict_params = {'lr': 1e-2, 'lambda': 1e-3, 'lambdaa_u': 1e1, 'topk': 4}
    if method == 'GMM':
        dict_params = {'matrix_parents': matrix_parents}
    if method == 'kmeans' :
        dict_params = {'matrix_parents': matrix_parents}

    return dict_params


def init_model(method, dict_params,d, c,set_y_s, set_y_u, C, matrix_parents = []):
    if method == 'hPhi' or method == 'hmodel':
        model = hPhi(matrix_parents=dict_params['matrix_parents'],
                        d=d,
                        C=C,
                        lr=dict_params['lr'],
                        lambda_eps=dict_params['lambda_eps'],
                        lambdaa_u=dict_params['lambdaa_u'])

    if method == 'hGMM':
        model = hGMM(matrix_parents=dict_params['matrix_parents'],
                          d=d,
                            C=C,
                          lambdaa=dict_params['lambdaa'],
                          lr=dict_params['lr'],
                          sub_epoch=dict_params['sub_epoch'])

    if method == 'Autonovel':
        model = Autonovel(d=d,
                          NS=len(set_y_s), NU=len(set_y_u),
                          hidden_layer_d=50, lambdaa=1e-3, lr=1e-2, topk=5, )

    if method == 'GMM':
        model = GMM(matrix_parents=dict_params['matrix_parents'], d=d)
    if method == 'kmeans' :
        model = kmeans(matrix_parents=dict_params['matrix_parents'], d=d)

    return model

#########################################################
#%%


def CV_gridsearch(method, X_s_train, y_s_train, X_u_train, y_u_train, C, matrix_parents,matrix_ascendant_descendant) :

    record = []
    set_y_s = list(set(y_s_train))
    all_split_shuffle_val = partition(list_in=set_y_s, n=5)
    c = C.shape[0]
    d = X_s_train.shape[1]

    for v in range(5):
        #print(v)

        X_ss_train, X_su_train, y_ss_train, y_su_train = split_val_cv(X_s_train, y_s_train,v=v, split_label=all_split_shuffle_val)

        # for training
        set_y_ss = list(set(y_ss_train))
        set_y_uu = list(set(y_su_train + y_u_train))
        Xss, yss = X_ss_train, y_ss_train
        Xuu, yuu = torch.cat((X_su_train, X_u_train), dim=0), [-1] * len(y_su_train) + [-1] * len(y_u_train)
        ns, nv = len(set_y_ss), len(set_y_uu)


        # for validation
        set_y_su = list(set(y_su_train))


        def gridsearch(dict_params):

            model = init_model(method=method,
                               dict_params=dict_params,
                               d=d,
                               c=c,
                               set_y_s=set_y_ss,
                               set_y_u=set_y_uu,
                               C=C,
                               matrix_parents = matrix_parents)

            model.fit(Xs=Xss,
                      Xu=Xuu,
                      ys=yss,
                      set_y_s= set_y_ss,
                      set_y_u=set_y_uu
                          )

            pred_u = model.predict(X=X_su_train, set_y=set_y_su )

            centroid_ss = [Xss[torch.where(torch.tensor(yss) == cluster)].mean(dim=0).detach().tolist()
                           for cluster in set_y_ss ]

            centroid_uu = [model.centroid_u[i].detach().tolist() for i in range(len(set_y_uu)) ]



            solution, solution_inv = search_supervision(set_y_s=set_y_ss,
                                                        set_y_u=set_y_uu,
                                                        C=C,
                                                        matrix_ascendant_descendant=matrix_ascendant_descendant,
                                                        centroid_s=centroid_ss,
                                                        centroid_u=centroid_uu)

            mapping_u_test = inv_mapping(solution_inv, model.predict_cluster(X=X_su_train, set_y=set_y_su))
            f1_search =  f1_score(y_true=y_su_train, y_pred=mapping_u_test, average='micro')

            solution, solution_inv = search_supervision_multiple(set_y_s=set_y_ss,
                                                        set_y_u=set_y_uu,
                                                        C=C,
                                                                 matrix_ascendant_descendant=matrix_ascendant_descendant,
                                                        centroid_s=centroid_ss,
                                                        centroid_u=centroid_uu)

            mapping_u_test = inv_mapping(solution_inv, model.predict_cluster(X=X_su_train, set_y=set_y_su))
            f1_search_multiple = f1_score(y_true=y_su_train, y_pred=mapping_u_test, average='micro')

            solution, solution_inv = breadth_first_search(
                                                        set_y_s=set_y_ss,
                                                        set_y_u=set_y_uu,
                                                        C=C,
                                                          matrix_ascendant_descendant= matrix_ascendant_descendant,
                                                        centroid_s=centroid_ss,
                                                        centroid_u=centroid_uu)

            mapping_u_test = inv_mapping(solution_inv, model.predict_cluster(X=X_su_train, set_y=set_y_su))
            f1_bfs = f1_score(y_true=y_su_train, y_pred=mapping_u_test, average='micro')

            solution, solution_inv = bfs_parallel(
                set_y_s=set_y_ss,
                set_y_u=set_y_uu,
                C=C,
                matrix_ascendant_descendant=matrix_ascendant_descendant,
                centroid_s=centroid_ss,
                centroid_u=centroid_uu)

            mapping_u_test = inv_mapping(solution_inv, model.predict_cluster(X=X_su_train, set_y=set_y_su))
            f1_bfs_multiple = f1_score(y_true=y_su_train, y_pred=mapping_u_test, average='micro')
            #
            #
            # # FOR this CV, juste select best ACC for mapping algo (like best clustering)
            # f1_search, f1_bfs = cluster_acc(y_true=y_su_train, y_pred=pred_u),cluster_acc(y_true=y_su_train, y_pred=pred_u)

            return ( f1_score( y_true=y_su_train, y_pred=pred_u, average='micro'),
                     f1_search ,  # f1 mapping supervision
                     f1_search_multiple,  # F1 mapping supervision multiple
                     f1_bfs, # F1 mapping BFS
                        f1_bfs_multiple,
                     cluster_acc(y_true=y_su_train, y_pred=pred_u),  #ACC
                     normalized_mutual_info_score(labels_true=y_su_train, labels_pred=pred_u),  # NMI
                     hierarcical_dist(y_true=y_su_train, y_pred=pred_u, C=C)

            )



        liste_params = find_dict_params(method, matrix_parents,c)


        model = init_model(method=method, dict_params=liste_params,d=d, c=c,C=C, set_y_s=set_y_ss, set_y_u=set_y_uu, matrix_parents=matrix_parents)
        grid = model.param_grid()

        keys, values = zip(*grid.items())
        gridsearch_parameter = [dict(zip(keys, v)) for v in itertools.product(*values)]

        dict_gridsearch_parameter = gridsearch_parameter

        all_result_v = Parallel(n_jobs=-1)(
            delayed(gridsearch)(combinaison) for combinaison in dict_gridsearch_parameter)
        record.append(all_result_v)



    return np.array(record)

def TEST(method, liste_hyperparams,
         X_s_train, y_s_train, X_u_train, y_u_train,
            X_s_test, y_s_test, X_u_test, y_u_test,
         C, matrix_parents,matrix_ascendant_descendant
         ) :

    set_y_s = list(set(y_s_train))
    set_y_u = list(set(y_u_train))
    c = C.shape[0]
    d = X_s_train.shape[1]

    def one_fit(dict_params):
        model = init_model(method=method,
                           dict_params=dict_params,
                           d=d,
                           c=c,
                           set_y_s=set_y_s,
                           set_y_u=set_y_u,
                           C=C,
                           matrix_parents=matrix_parents)

        model.fit(Xs=X_s_train,
                  Xu=X_u_train,
                  ys=y_s_train,
                  set_y_s=set_y_s,
                  set_y_u=set_y_u
                  )

        pred_u = model.predict(X=X_u_test, set_y=set_y_u)

        centroid_s = [X_s_train[torch.where(torch.tensor(y_s_train) == cluster)].mean(dim=0).detach().tolist()
                       for cluster in set_y_s]

        centroid_u = [model.centroid_u[i].detach().tolist() for i in range(len(set_y_u))]

        solution, solution_inv = search_supervision(set_y_s=set_y_s,
                                                    set_y_u=set_y_u,
                                                    C=C,
                                                    matrix_ascendant_descendant=matrix_ascendant_descendant,
                                                    centroid_s=centroid_s,
                                                    centroid_u=centroid_u)

        mapping_u_test = inv_mapping(solution_inv, model.predict_cluster(X=X_u_test, set_y=set_y_u))
        f1_search = f1_score(y_true=y_u_test, y_pred=mapping_u_test, average='micro')

        solution, solution_inv = search_supervision_multiple(set_y_s=set_y_s,
                                                    set_y_u=set_y_u,
                                                    C=C,
                                                             matrix_ascendant_descendant=matrix_ascendant_descendant,
                                                    centroid_s=centroid_s,
                                                    centroid_u=centroid_u)

        mapping_u_test = inv_mapping(solution_inv, model.predict_cluster(X=X_u_test, set_y=set_y_u))
        f1_search_multiple = f1_score(y_true=y_u_test, y_pred=mapping_u_test, average='micro')

        solution, solution_inv = breadth_first_search(
                                                      set_y_s=set_y_s,
                                                      set_y_u=set_y_u,
                                                      C=C,
                                                      matrix_ascendant_descendant=matrix_ascendant_descendant,
                                                      centroid_s=centroid_s,
                                                      centroid_u=centroid_u)

        mapping_u_test = inv_mapping(solution_inv, model.predict_cluster(X=X_u_test, set_y=set_y_u))
        f1_bfs = f1_score(y_true=y_u_test, y_pred=mapping_u_test, average='micro')

        solution, solution_inv =  bfs_parallel(
                                                      set_y_s=set_y_s,
                                                      set_y_u=set_y_u,
                                                      C=C,
                                                      matrix_ascendant_descendant=matrix_ascendant_descendant,
                                                      centroid_s=centroid_s,
                                                      centroid_u=centroid_u)

        mapping_u_test = inv_mapping(solution_inv, model.predict_cluster(X=X_u_test, set_y=set_y_u))
        f1_bfs_multiple = f1_score(y_true=y_u_test, y_pred=mapping_u_test, average='micro')

        results = (f1_score(y_true=y_u_test, y_pred=pred_u, average='micro'),
                f1_search,  # f1 mapping supervision
                f1_search_multiple, # f1 mapping multiple
                f1_bfs,  # F1 mapping BFS
                f1_bfs_multiple,
                cluster_acc(y_true=y_u_test, y_pred=pred_u),  # ACC
                normalized_mutual_info_score(labels_true=y_u_test, labels_pred=pred_u),  # NMI
                hierarcical_dist(y_true=y_u_test, y_pred=pred_u, C=C)

                )
        if method == 'hmodel':
            model.hPhi = False
            model.hGMM = True
            model.fit(Xs=X_s_train,
                      Xu=X_u_train,
                      ys=y_s_train,
                      set_y_s=set_y_s,
                      set_y_u=set_y_u
                      )

            pred_u = model.predict(X=X_u_test, set_y=set_y_u)

            centroid_s = [X_s_train[torch.where(torch.tensor(y_s_train) == cluster)].mean(dim=0).detach().tolist()
                          for cluster in set_y_s]

            centroid_u = [model.centroid_u[i].detach().tolist() for i in range(len(set_y_u))]

            solution, solution_inv = search_supervision(set_y_s=set_y_s,
                                                        set_y_u=set_y_u,
                                                        C=C,
                                                        matrix_ascendant_descendant=matrix_ascendant_descendant,
                                                        centroid_s=centroid_s,
                                                        centroid_u=centroid_u)

            mapping_u_test = inv_mapping(solution_inv, model.predict_cluster(X=X_u_test, set_y=set_y_u))
            f1_search = f1_score(y_true=y_u_test, y_pred=mapping_u_test, average='micro')

            solution, solution_inv = search_supervision_multiple(set_y_s=set_y_s,
                                                                 set_y_u=set_y_u,
                                                                 C=C,
                                                                 matrix_ascendant_descendant=matrix_ascendant_descendant,
                                                                 centroid_s=centroid_s,
                                                                 centroid_u=centroid_u)

            mapping_u_test = inv_mapping(solution_inv, model.predict_cluster(X=X_u_test, set_y=set_y_u))
            f1_search_multiple = f1_score(y_true=y_u_test, y_pred=mapping_u_test, average='micro')

            solution, solution_inv = breadth_first_search(
                set_y_s=set_y_s,
                set_y_u=set_y_u,
                C=C,
                matrix_ascendant_descendant=matrix_ascendant_descendant,
                centroid_s=centroid_s,
                centroid_u=centroid_u)

            mapping_u_test = inv_mapping(solution_inv, model.predict_cluster(X=X_u_test, set_y=set_y_u))
            f1_bfs = f1_score(y_true=y_u_test, y_pred=mapping_u_test, average='micro')

            solution, solution_inv = bfs_parallel(
                set_y_s=set_y_s,
                set_y_u=set_y_u,
                C=C,
                matrix_ascendant_descendant=matrix_ascendant_descendant,
                centroid_s=centroid_s,
                centroid_u=centroid_u)

            mapping_u_test = inv_mapping(solution_inv, model.predict_cluster(X=X_u_test, set_y=set_y_u))
            f1_bfs_multiple = f1_score(y_true=y_u_test, y_pred=mapping_u_test, average='micro')

            results_bis = (f1_score(y_true=y_u_test, y_pred=pred_u, average='micro'),
                       f1_search,  # f1 mapping supervision
                       f1_search_multiple,  # f1 mapping multiple
                       f1_bfs,  # F1 mapping BFS
                       f1_bfs_multiple,
                       cluster_acc(y_true=y_u_test, y_pred=pred_u),  # ACC
                       normalized_mutual_info_score(labels_true=y_u_test, labels_pred=pred_u),  # NMI
                       hierarcical_dist(y_true=y_u_test, y_pred=pred_u, C=C)

                       )
            results = results + results_bis
        return results


    liste_params = find_dict_params(method, matrix_parents, c)

    model = init_model(method=method, dict_params=liste_params, d=d, c=c, C=C, set_y_s=set_y_s, set_y_u=set_y_u,
                       matrix_parents=matrix_parents)
    grid = model.param_grid()
    keys, values = zip(*grid.items())
    gridsearch_parameter = [dict(zip(keys, v)) for v in itertools.product(*values)]
    dict_best_hyperparameters = [gridsearch_parameter[best] for best in liste_hyperparams]

    all_results_test = Parallel(n_jobs=-1)(
        delayed(one_fit)(combinaison) for combinaison in dict_best_hyperparameters)

    return np.array(all_results_test)




########################################"

#%%

class kmeans(nn.Module):

    def __init__(self, matrix_parents, d=50, C=torch.tensor([1]),):
        super().__init__()
        self.matrix_parents = matrix_parents
        self.C =C
        self.c = matrix_parents.shape[0]
        self.d = d


    def fit(self, Xs, Xu, ys, set_y_s=[], set_y_u=[], max_iter=100):


        ### INIT centroid
        self.set_y_s = set_y_s
        self.set_y_u = set_y_u
        self.set_s_present = list(set(ys))
        self.epoch = 1

        self.model = KMeans(n_clusters=len(self.set_y_u), random_state=0)

        centroid_s = [
            Xs[torch.tensor(ys) == cluster].mean(dim=0).detach().tolist()
            if cluster in self.set_s_present
            else (Xs[torch.tensor(ys) == self.set_s_present[self.C[cluster][self.set_s_present].argmin()]].mean(dim=0)
                  + torch.randn(1, self.d)).detach().tolist()[0]

            for cluster in self.set_y_s]

        # les centroids non presents on les places proches des centroids supervisés proches

        self.centroid_s = torch.tensor(centroid_s)

        self.model.fit(X=Xu)
        self.centroid_u = torch.FloatTensor(self.model.cluster_centers_)

        self.new_Phi = torch.zeros((self.c, self.d))
        self.new_Phi[self.set_y_s] =  self.centroid_s
        self.new_Phi[self.set_y_u] = self.centroid_u

    def predict(self, X, set_y=[]):
        mat_phi = self.new_Phi
        if len(set_y) > 0:
            mat_phi = mat_phi[set_y]
        y_pred = torch.argmin(torch.cdist(X, mat_phi), dim=1).tolist()
        if len(set_y) > 0:
            y_pred = np.array(set_y)[y_pred].tolist()
        return y_pred

    def predict_cluster(self, X, set_y=[]):
        mat_phi = self.new_Phi
        if len(set_y) > 0:
            mat_phi = mat_phi[set_y]
        y_pred = torch.argmin(torch.cdist(X, mat_phi), dim=1).tolist()

        return y_pred

    def param_grid(self):
        param_grid = {
                      'matrix_parents': [self.matrix_parents]
                      }
        return param_grid

    def tiny_param_grid(self):
        param_grid = {
                      'matrix_parents': [self.matrix_parents]
                      }
        return param_grid

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        return { }



class GMM(nn.Module):

    def __init__(self, matrix_parents, d=50, C=torch.tensor([1]),cov_type='diag'):
        super().__init__()
        self.matrix_parents = matrix_parents
        self.C =C
        self.c = matrix_parents.shape[0]
        self.d = d
        self.cov_type = cov_type


    def fit(self, Xs, Xu, ys, set_y_s=[], set_y_u=[], max_iter=100):


        ### INIT centroid
        self.set_y_s = set_y_s
        self.set_y_u = set_y_u
        self.set_s_present = list(set(ys))
        self.epoch = 1

        self.model = GaussianMixture(n_components=len(set_y_u), random_state=0, covariance_type=self.cov_type)


        centroid_s = [
            Xs[torch.tensor(ys) == cluster].mean(dim=0).detach().tolist()
            if cluster in self.set_s_present
            else (Xs[torch.tensor(ys) == self.set_s_present[self.C[cluster][self.set_s_present].argmin()]].mean(dim=0)
                  + torch.randn(1, self.d)).detach().tolist()[0]

            for cluster in self.set_y_s]

        # les centroids non presents on les places proches des centroids supervisés proches

        self.centroid_s = torch.tensor(centroid_s)

        self.model.fit(X=Xu)
        self.centroid_u = torch.FloatTensor(self.model.means_)

        self.new_Phi = torch.zeros((self.c, self.d))
        self.new_Phi[self.set_y_s] =  self.centroid_s
        self.new_Phi[self.set_y_u] = self.centroid_u

    def predict(self, X, set_y=[]):
        mat_phi = self.new_Phi
        if len(set_y) > 0:
            mat_phi = mat_phi[set_y]
        y_pred = torch.argmin(torch.cdist(X, mat_phi), dim=1).tolist()
        if len(set_y) > 0:
            y_pred = np.array(set_y)[y_pred].tolist()
        return y_pred

    def predict_cluster(self, X, set_y=[]):
        mat_phi = self.new_Phi
        if len(set_y) > 0:
            mat_phi = mat_phi[set_y]
        y_pred = torch.argmin(torch.cdist(X, mat_phi), dim=1).tolist()

        return y_pred

    def param_grid(self):
        param_grid = {
                      'matrix_parents': [self.matrix_parents]
                      }
        return param_grid

    def tiny_param_grid(self):
        param_grid = {
                      'matrix_parents': [self.matrix_parents]
                      }
        return param_grid

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        return { }


class hPhi(nn.Module):

    def __init__(self, matrix_parents, d=50, C=torch.tensor([1]), lr=5e-1, lambda_eps=1e-2, lambdaa_u=1e1):
        super().__init__()
        self.matrix_parents = matrix_parents
        self.C =C
        self.c = matrix_parents.shape[0]
        self.d = d

        self.lr = lr
        self.lambdaa_u = lambdaa_u
        self.lambda_eps = lambda_eps

    def forward(self):
        return self.matrix_parents @ self.Epsilon

    def init_centroid_u(self,  set_y_u, nb_neighbor = 3):

        index_voisinage = self.C[set_y_u][:,self.set_s_present]
        topk_voisinage = index_voisinage.topk(dim=1,k=nb_neighbor, largest=False).indices


        self.centroid_u = torch.tensor([self.centroid_s[topk_voisinage[i]].mean(dim=0).detach().tolist()
                      for i in range(len(set_y_u))
                      ])  + torch.rand(len(set_y_u),self.d)

    def fit(self, Xs, Xu, ys, set_y_s=[], set_y_u=[], max_iter=100):

        ### INIT centroid
        self.set_y_s = set_y_s
        self.set_y_u = set_y_u
        self.set_s_present = list(set(ys))
        self.epoch = 1

        centroid_s = [
            Xs[torch.tensor(ys) == cluster].mean(dim=0).detach().tolist()
            if cluster in self.set_s_present
            else (Xs[torch.tensor(ys) == self.set_s_present[self.C[cluster][self.set_s_present].argmin()]].mean(dim=0)
                  + torch.randn(1, self.d)).detach().tolist()[0]

            for cluster in self.set_y_s]

        # les centroids non presents on les places proches des centroids supervisés proches

        self.centroid_s = torch.tensor(centroid_s)

        #### PROTOTYPE
        self.phi = torch.randn((self.c, self.d))
        self.phi[self.set_y_s] = self.centroid_s


        self.init_centroid_u(set_y_u, nb_neighbor = 3)
        self.phi[self.set_y_u] = self.centroid_u


        self.epsilon = self.matrix_parents.inverse() @ self.phi
        self.eps_s = self.epsilon[self.set_y_s]
        self.epsilon = self.epsilon.requires_grad_(True)

        # self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lambda_eps)
        self.optimizer = torch.optim.Adam([self.epsilon], lr=self.lr, weight_decay=self.lambda_eps)

        for epoch in range(max_iter):
            self.new_Phi = self.matrix_parents @ self.epsilon
            loss_s = torch.norm(Xs - self.new_Phi[ys], dim=1).mean()

            index_y_pseudo_label_u = torch.argmin(torch.cdist(Xu, self.new_Phi[self.set_y_u]), dim=1).tolist()
            # y_pseudo_label_u = torch.tensor(self.set_y_u)[index_y_pseudo_label_u].tolist()

            loss_u = torch.norm(Xu - self.new_Phi[self.set_y_u][torch.tensor(index_y_pseudo_label_u)], dim=1).mean()

            loss = loss_s + self.lambdaa_u * loss_u  # + self.lambda_eps*loss_eps
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # with torch.no_grad():
            #     self.epsilon[set_y_s] =   self.eps_s

            #self.optimizer.weight_decay= self.lambda_eps/(5*self.epoch)

        self.new_Phi = self.matrix_parents @ self.epsilon

        self.centroid_u = self.new_Phi[self.set_y_u]

    def predict(self, X, set_y=[]):
        mat_phi = self.new_Phi
        if len(set_y) > 0:
            mat_phi = mat_phi[set_y]
        y_pred = torch.argmin(torch.cdist(X, mat_phi), dim=1).tolist()
        if len(set_y) > 0:
            y_pred = np.array(set_y)[y_pred].tolist()
        return y_pred

    def predict_cluster(self, X, set_y=[]):
        mat_phi = self.new_Phi
        if len(set_y) > 0:
            mat_phi = mat_phi[set_y]
        y_pred = torch.argmin(torch.cdist(X, mat_phi), dim=1).tolist()

        return y_pred

    def param_grid(self):
        param_grid = { 'lambda_eps':np.logspace(-3,4, 10),
                      'lr':[5e-2 ],
                      'lambdaa_u':np.logspace(-1,4, 5),
                      'matrix_parents': [self.matrix_parents]
                      }
        return param_grid

    def tiny_param_grid(self):
        param_grid = {"lambdaa": [1.e-4, 1.e-3],  # 3
                      'lr': [ 5e-2 ],
                      'sub_epoch': [20],
                      'matrix_parents': [self.matrix_parents]
                      }
        return param_grid

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        return { }

class hGMM(nn.Module):
    def __init__(self, matrix_parents, d=50,lambdaa = 1e-3, lr=5e-1, sub_epoch=20, C=torch.tensor([1]), cov_type='diag'):
        super().__init__()

        self.matrix_parents = matrix_parents
        self.c = matrix_parents.shape[0]
        self.n_components = self.c
        self.d = d
        self.lambdaa = lambdaa
        self.lr = lr
        self.sub_epoch = sub_epoch
        self.epoch_abs = 0
        self.C = C
        self.cov_type = cov_type

    def init_centroid_u(self, set_y_u, nb_neighbor=3):

        index_voisinage = self.C[set_y_u][:, self.set_s_present]
        topk_voisinage = index_voisinage.topk(dim=1, k=nb_neighbor, largest=False).indices

        self.centroid_u = torch.tensor([self.centroid_s[topk_voisinage[i]].mean(dim=0).detach().tolist()
                                        for i in range(len(set_y_u))
                                        ]) + torch.rand(len(set_y_u), self.d)

    def fit(self, Xs, Xu, ys, set_y_s=[], set_y_u=[], max_iter=20, centroid_u=[]):

        nu = Xu.shape[0]

        ### INIT centroid
        self.set_y_s = set_y_s
        self.set_y_u = set_y_u
        self.set_s_present = list(set(ys))

        centroid_s = [
            Xs[torch.tensor(ys) == cluster].mean(dim=0).detach().tolist()
            if cluster in self.set_s_present
            else
            (Xs[torch.tensor(ys) == self.set_s_present[self.C[cluster][self.set_s_present].argmin()]].mean(dim=0)
             + torch.randn(1, self.d)).detach().tolist()[0]
            for cluster in self.set_y_s
        ]

        # les centroids non presents on les places proches des centroids supervisés proches

        self.centroid_s = torch.tensor(centroid_s)
        #### PROTOTYPE
        self.phi = torch.randn((self.c, self.d))
        self.phi[self.set_y_s] = self.centroid_s

        if len(centroid_u) ==0:
            self.init_centroid_u(set_y_u, nb_neighbor=3)
            self.phi[self.set_y_u] = self.centroid_u
        else :
            self.centroid_u = torch.tensor(centroid_u)
            self.phi[self.set_y_u] = self.centroid_u

        self.epsilon = self.matrix_parents.inverse() @ self.phi
        self.eps_s = self.epsilon[self.set_y_s]
        self.epsilon = self.epsilon.requires_grad_(True)

        # self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lambda_eps)
        self.optimizer = torch.optim.Adam([self.epsilon], lr=self.lr, weight_decay=self.lambdaa)



        # OPTIM
        ###################################################


        self.mu = self.matrix_parents @ self.epsilon
        self.sigma = [torch.eye(self.d) for ci in range(self.c)]


        self.gmm = [mn.MultivariateNormal(self.matrix_parents[k] @ self.epsilon, self.sigma[k]) for k in
                    range(self.n_components)]
        self.pi = torch.Tensor([1 / self.n_components] * self.n_components)
        self.pi[self.set_y_s] = torch.tensor([ys.count(i)/(len(ys) + nu) for i in self.set_y_s])
        self.pi[self.set_y_u] = torch.tensor([len(self.set_y_u) / len(self.set_y_s + self.set_y_u) * (1/len(self.set_y_u) )] * len(self.set_y_u))


        n = Xu.shape[0]
        self.W = torch.zeros(n, self.n_components)


        # if len(yu) == 0 :
        #     y = [-1] * n
        # mask_labelled = torch.Tensor(y) > -1
        # mask_unlabelled = torch.Tensor(y) == -1
        # self.W[mask_labelled] = torch.eye(self.n_components)[torch.tensor(y)[mask_labelled].tolist(),:]

        for epoch in range(max_iter):

            # E STEP
            for k in set_y_u:

                prior = self.pi[k]
                likelihood = self.gmm[k].log_prob(Xu).exp()  +1e-16 # probability that X is taken from normal(means, cov)
                self.W[:, k] = prior * likelihood
                #self.record_k = k

            self.log_likelihood = self.W.sum(dim=1).log().sum()

            # normalize over all possible cluster assignments
            self.W = self.W.T.div(self.W.sum(dim=1)).T
            self.W = self.W.detach()

            #self.resp = self.W

            # M STEP for Epsilon
            for sub_epoch in range(self.sub_epoch):
                # with torch.autograd.set_detect_anomaly(True) :
                self.Q_loss = - sum([self.W[:, k] @
                                (self.gmm[k].log_prob(Xu)  # + weights[k].log()
                                 ) - (self.lambdaa ) * torch.norm(self.epsilon[k])
                                for k in range(self.n_components)])

                self.optimizer.zero_grad()
                self.Q_loss.backward()
                self.optimizer.step()

                #  self.update()
                self.mu = self.matrix_parents @ self.epsilon
                self.mu[self.set_y_s] = self.centroid_s
                self.gmm = [mn.MultivariateNormal(self.mu[k], self.sigma[k]) for k in range(self.n_components)]

            #######################"

            self.resp_weights = self.W.sum(dim=0)
            self.pi[self.set_y_u] = self.resp_weights[self.set_y_u] / Xu.shape[0]
            self.pi = self.pi.detach()

            #############################################
            ##### COV

            # weights

            if self.cov_type == 'full':

                with torch.no_grad():
                    # covariance
                    for k in set_y_u :
                        diff = (Xu - self.mu[k]).T
                        weighted_sum = torch.matmul(self.W[:, k] * diff, diff.T)
                        self.sigma[k] = weighted_sum /  (1 + self.resp_weights[k] )
                        self.gmm = [mn.MultivariateNormal(self.mu[k], self.sigma[k]) for k in range(self.n_components)]


        self.epoch_abs +=1

        self.centroid_u = self.mu[self.set_y_u]


    def predict(self, X, set_y=[]):


        if len(set_y) == 0 :
           all_proba = torch.Tensor(
                [self.gmm[k].log_prob(X).exp().detach().tolist() for k in range(self.n_components)]).T
           pred = torch.argmax(all_proba, dim=1).tolist()
        else :
            all_proba = torch.Tensor(
                [self.gmm[k].log_prob(X).exp().detach().tolist() for k in set_y]).T
            pred = torch.tensor(set_y)[torch.argmax(all_proba, dim=1).tolist()].tolist()
        return pred

    def predict_cluster(self, X, set_y=[]):
        if len(set_y) == 0 :
           all_proba = torch.Tensor(
                [self.gmm[k].log_prob(X).exp().detach().tolist() for k in range(self.n_components)]).T
           pred = torch.argmax(all_proba, dim=1).tolist()
        else :
            all_proba = torch.Tensor(
                [self.gmm[k].log_prob(X).exp().detach().tolist() for k in set_y]).T
            pred = torch.argmax(all_proba, dim=1).tolist()
        return pred

    def param_grid(self):
        param_grid = { 'lambdaa':np.logspace(-3,4, 16),
                      'lr':[5e-1 ],
                      'sub_epoch':[20],
                      'matrix_parents': [self.matrix_parents]
                      }
        return param_grid

    def tiny_param_grid(self):
        param_grid = {"lambdaa": [1.e-4, 1.e-3],  # 3
                      'lr': [ 5e-2 ],
                      'sub_epoch': [20],
                      'matrix_parents': [self.matrix_parents]
                      }
        return param_grid

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        return { }







class hmodel(nn.Module):


    def __init__(self, matrix_parents, d=50, C=torch.tensor([1]), lr=5e-1, lambda_eps=1e-2, lambdaa_u=1e1):
        super().__init__()
        self.matrix_parents = matrix_parents
        self.C =C
        self.c = matrix_parents.shape[0]
        self.d = d

        self.lr = lr
        self.lambdaa_u = lambdaa_u
        self.lambda_eps = lambda_eps

        self.hPhi = True
        self.hGMM = False
        self.update_cov = False


    def init_centroid_u(self,  set_y_u, nb_neighbor = 3):

        index_voisinage = self.C[set_y_u][:,self.set_s_present]
        topk_voisinage = index_voisinage.topk(dim=1,k=nb_neighbor, largest=False).indices

        self.centroid_u = torch.tensor([self.centroid_s[topk_voisinage[i]].mean(dim=0).detach().tolist()
                      for i in range(len(set_y_u))
                      ])  + torch.rand(len(set_y_u),self.d)

    def fit(self, Xs, Xu, ys, set_y_s=[], set_y_u=[], max_iter=100):
        if self.hPhi:
            self.fit_hPhi(Xs, Xu, ys, set_y_s, set_y_u, max_iter=100)

        if self.hGMM:
            self.fit_hGMM(Xs, Xu, ys, set_y_s, set_y_u, max_iter=20)

    def predict(self, X, set_y):
        if self.hPhi:
            pred = self.predict_hPhi(X=X, set_y=set_y)

        if self.hGMM:
            pred = self.predict_hGMM(X=X, set_y=set_y)
        return pred

    def predict_cluster(self, X, set_y):
        if self.hPhi:
            pred = self.predict_cluster_hPhi(X=X, set_y=set_y)

        if self.hGMM:
            pred = self.predict_cluster_hGMM(X=X, set_y=set_y)
        return pred



    def fit_hPhi(self, Xs, Xu, ys, set_y_s=[], set_y_u=[], max_iter=100):

        ### INIT centroid
        self.set_y_s = set_y_s
        self.set_y_u = set_y_u
        self.set_s_present = list(set(ys))
        self.epoch = 1

        centroid_s = [
            Xs[torch.tensor(ys) == cluster].mean(dim=0).detach().tolist()
            if cluster in self.set_s_present
            else (Xs[torch.tensor(ys) == self.set_s_present[self.C[cluster][self.set_s_present].argmin()]].mean(dim=0)
                  + torch.randn(1, self.d)).detach().tolist()[0]

            for cluster in self.set_y_s]
        self.centroid_s = torch.tensor(centroid_s)

        #### PROTOTYPE
        self.phi = torch.randn((self.c, self.d))
        self.phi[self.set_y_s] = self.centroid_s

        self.init_centroid_u(set_y_u, nb_neighbor = 3)
        self.phi[self.set_y_u] = self.centroid_u

        self.epsilon = self.matrix_parents.inverse() @ self.phi
        self.eps_s = self.epsilon[self.set_y_s]
        self.epsilon = self.epsilon.requires_grad_(True)

        # self.optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=lambda_eps)
        self.optimizer = torch.optim.Adam([self.epsilon], lr=self.lr, weight_decay=self.lambda_eps)

        for epoch in range(max_iter):
            self.new_Phi = self.matrix_parents @ self.epsilon
            loss_s = torch.norm(Xs - self.new_Phi[ys], dim=1).mean()

            index_y_pseudo_label_u = torch.argmin(torch.cdist(Xu, self.new_Phi[self.set_y_u]), dim=1).tolist()
            # y_pseudo_label_u = torch.tensor(self.set_y_u)[index_y_pseudo_label_u].tolist()

            loss_u = torch.norm(Xu - self.new_Phi[self.set_y_u][torch.tensor(index_y_pseudo_label_u)], dim=1).mean()

            loss = loss_s + self.lambdaa_u * loss_u  # + self.lambda_eps*loss_eps
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.new_Phi = self.matrix_parents @ self.epsilon
        self.centroid_u_Phi = self.new_Phi[self.set_y_u]




    def predict_hPhi(self, X, set_y=[]):
        mat_phi = self.new_Phi
        if len(set_y) > 0:
            mat_phi = mat_phi[set_y]
        y_pred = torch.argmin(torch.cdist(X, mat_phi), dim=1).tolist()
        if len(set_y) > 0:
            y_pred = np.array(set_y)[y_pred].tolist()
        return y_pred

    def predict_cluster_hPhi(self, X, set_y=[]):
        mat_phi = self.new_Phi
        if len(set_y) > 0:
            mat_phi = mat_phi[set_y]
        y_pred = torch.argmin(torch.cdist(X, mat_phi), dim=1).tolist()
        return y_pred

    def fit_hGMM(self, Xs, Xu, ys, set_y_s, set_y_u, max_iter=20):

        self.set_y_s = set_y_s
        self.set_y_u = set_y_u


        self.sigma_s = torch.eye(self.d)
        self.sigma_s[range(len(self.sigma_s)), range(len(self.sigma_s))] = torch.diag((Xs.T @ Xs) / (Xs.shape[0]))
        self.sigma_s = [self.sigma_s for i in range(len(self.set_y_s))]


        lr, lambdaa_eps = 1e-1, self.lambda_eps
        self.ncomp = len(self.set_y_u)
        self.pi = torch.FloatTensor([1 / self.ncomp] * self.ncomp)

        self.mu0 = self.centroid_u_Phi

        self.sigma0 = torch.eye(self.d)
        self.sigma0[range(len(self.sigma0)), range(len(self.sigma0))] = torch.diag((Xu.T @ Xu) / (Xu.shape[0]))
        self.sigma0 = [self.sigma0 for i in range(self.ncomp)]

        self.W = torch.zeros(Xu.shape[0], self.ncomp)
        self.sigma = self.sigma0
        self.mu = self.mu0
        self.mu_full = self.new_Phi

        self.mu_set_s = self.mu_full[self.set_y_s]
        self.eps_all = torch.inverse(self.matrix_parents) @ self.mu_full
        self.eps_all = self.eps_all.detach()
        self.eps_s = self.eps_all[self.set_y_s]
        # eps = eps_all[set_y_u]
        self.eps_all[self.set_y_s] = self.eps_s.detach()
        self.eps_all.requires_grad_()

        self.optimizer = torch.optim.Adam([self.eps_all], lr=lr, weight_decay=lambdaa_eps)

        self.mu_all = self.matrix_parents @ self.eps_all

        self.gmm = [mn.MultivariateNormal((self.matrix_parents @ self.eps_all)[set_y_u][k], self.sigma0[k]) for k in
               range(self.ncomp)]


        for iter in range(5):

            for k in range(self.ncomp):
                prior = self.pi[k]
                likelihood = self.gmm[k].log_prob(Xu).exp() + 1e-16  # probability that X is taken from normal(means, cov)
                self.W[:, k] = prior * likelihood
                # record_k = k

            self.log_likelihood = self.W.sum(dim=1).log().sum()
            # print(log_likelihood)

            # normalize over all possible cluster assignments
            self.W = self.W.T.div(self.W.sum(dim=1)).T
            self.W = self.W.detach()

            # PI
            #######################"

            self.resp_weights = self.W.sum(dim=0)
            self.pi = self.resp_weights / Xu.shape[0]
            self.pi = self.pi.detach()

            # MU
            ##############################
            # mu = (W.T @ Xu).T.div(W.sum(dim=0)).T

            for sub_epoch in range(50):
                # with torch.autograd.set_detect_anomaly(True) :
                self.Q_loss = - sum([self.W[:, k] @
                                (self.gmm[k].log_prob(Xu)  # + weights[k].log()
                                 )  # - (lambdaa) * torch.norm(self.epsilon[k])
                                for k in range(self.ncomp)])
                #
                self.optimizer.zero_grad()
                self.Q_loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    self.eps_all[self.set_y_s] = self.eps_s

                self.gmm = [mn.MultivariateNormal((self.matrix_parents @ self.eps_all)[self.set_y_u][k], self.sigma[k]) for k in
                       range(self.ncomp)]

            ##########################################"""
            # Sigma
            ###############################

            self.mu_all = self.matrix_parents @ self.eps_all
            self.mu = self.mu_all[set_y_u]

            if self.update_cov:
                self.mu_var = self.mu.detach()
                for k in range(self.ncomp):
                    diff = (Xu - self.mu_var[k]).T
                    self.weighted_sum = torch.matmul(self.W[:, k] * diff, diff.T)
                    self.sigma[k] = self.weighted_sum / (1 + self.resp_weights[k])
                    self.sigma[k][range(len(self.sigma[k])), range(len(self.sigma[k]))] = torch.diag(self.sigma[k]) + 1

            with torch.no_grad():
                self.eps_all[set_y_s] = self.eps_s
                self.mu_all = self.matrix_parents @ self.eps_all
                self.mu_all[set_y_s] = self.mu_set_s
                self.eps_all = torch.inverse(self.matrix_parents) @ self.mu_all
                self.eps_all.requires_grad_()
                self.optimizer = torch.optim.Adam([self.eps_all], lr=lr, weight_decay=lambdaa_eps)

            self.mu_all = self.matrix_parents @ self.eps_all
            self.mu = self.mu_all[self.set_y_u]

            self.gmm = [mn.MultivariateNormal(self.mu[k], self.sigma[k]) for k in
                   range(len(set_y_u))]

            self.mu_all[set_y_u] = self.mu.detach()

        self.gmm_all = []
        for i in range(self.c):
            if i in self.set_y_s :
                index_set_i = self.set_y_s.index(i)
                self.gmm_all.append(mn.MultivariateNormal(self.new_Phi[index_set_i], self.sigma_s[index_set_i]))
            else :
                index_set_i = self.set_y_u.index(i)
                self.gmm_all.append(mn.MultivariateNormal(self.mu[index_set_i], self.sigma[index_set_i]))



    def predict_hGMM(self, X, set_y=[]):

        if len(set_y) == 0:
            all_proba = torch.Tensor(
                [self.gmm_all[k].log_prob(X).exp().detach().tolist() for k in range(self.n_components)]).T
            pred = torch.argmax(all_proba, dim=1).tolist()
        else:
            all_proba = torch.Tensor(
                [self.gmm_all[k].log_prob(X).exp().detach().tolist() for k in set_y]).T
            pred = torch.tensor(set_y)[torch.argmax(all_proba, dim=1).tolist()].tolist()
        return pred

    def predict_cluster_hGMM(self, X, set_y=[]):
        if len(set_y) == 0:
            all_proba = torch.Tensor(
                [self.gmm_all[k].log_prob(X).exp().detach().tolist() for k in range(self.n_components)]).T
            pred = torch.argmax(all_proba, dim=1).tolist()
        else:
            all_proba = torch.Tensor(
                [self.gmm_all[k].log_prob(X).exp().detach().tolist() for k in set_y]).T
            pred = torch.argmax(all_proba, dim=1).tolist()
        return pred


    def param_grid(self):
        param_grid = { 'lambda_eps':np.logspace(-3,4, 10),
                      'lr':[5e-2 ],
                      'lambdaa_u':np.logspace(-1,4, 5),
                      'matrix_parents': [self.matrix_parents]
                      }
        return param_grid

    def tiny_param_grid(self):
        param_grid = {"lambdaa": [1.e-4, 1.e-3],  # 3
                      'lr': [ 5e-2 ],
                      'sub_epoch': [20],
                      'matrix_parents': [self.matrix_parents]
                      }
        return param_grid

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        return { }










######################################"

### Autonovel


class BCE(nn.Module):
    eps = 1e-7  # Avoid calculating log(0). Use the small value of float16.

    def forward(self, prob1, prob2, simi):
        # simi: 1->similar; -1->dissimilar; 0->unknown(ignore)
        assert len(prob1) == len(prob2) == len(simi), 'Wrong input size:{0},{1},{2}'.format(str(len(prob1)),
                                                                                            str(len(prob2)),
                                                                                            str(len(simi)))
        P = prob1.mul_(prob2)
        P = P.sum(1)
        P.mul_(simi).add_(simi.eq(-1).type_as(P))
        neglogP = -P.add_(BCE.eps).log_()
        return neglogP.mean()


def PairEnum(x, mask=None):
    # Enumerate all pairs of feature in x
    assert x.ndimension() == 2, 'Input dimension must be 2'
    x1 = x.repeat(x.size(0), 1)
    x2 = x.repeat(1, x.size(0)).view(-1, x.size(1))
    if mask is not None:
        xmask = mask.view(-1, 1).repeat(1, x.size(1))
        # dim 0: #sample, dim 1:#feature
        x1 = x1[xmask].view(-1, x.size(1))
        x2 = x2[xmask].view(-1, x.size(1))
    return x1, x2


class Autonovel(nn.Module):
    def __init__(self, d=50, hidden_layer_d=50, lambdaa=1e-3, lr=1e-2, topk=5, NS=25, NU=25):
        super(Autonovel, self).__init__()

        self.d = d
        self.lambdaa = lambdaa
        self.lr = lr
        self.hidden_layer_d = hidden_layer_d
        self.head1 = nn.Linear(self.d, self.hidden_layer_d)
        self.head2 = nn.Linear(self.hidden_layer_d, self.hidden_layer_d)

        self.labelled_class, self.unlabelled_class = NS, NU
        self.classif1 = nn.Linear(self.hidden_layer_d, NS)
        self.classif2 = nn.Linear(self.hidden_layer_d, NU)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.lambdaa)
        self.criterion1 = nn.CrossEntropyLoss()
        self.criterion2 = BCE()

        self.topk = topk
        self.epoch_abs = 0

        return

    def forward(self, X):
        z = torch.tanh(self.head1(X))
        out = torch.tanh(self.head2(z))
        out1 = self.classif1(z)
        out2 = self.classif2(z)
        return out1, out2, out

    def fit(self,  Xs, Xu, ys, set_y_s=[], set_y_u=[], max_epochs=101):

        # mask_labelled = torch.Tensor(y) > -1
        # mask_unlabelled = torch.Tensor(y) == -1
        # X_s, X_u = X[mask_labelled], X[mask_unlabelled]
        # y_s = torch.tensor(y)[mask_labelled]
        X_s = Xs
        X_u = Xu
        y_s = ys
        liste_indice_lb = self.re_index(y_s)

        for epoch in range(max_epochs):

            self.epoch_abs +=1
            ### CE ###
            proj_s, _, _ = self.forward(X_s)
            self.loss_ce = self.criterion1(proj_s, torch.tensor(liste_indice_lb))

            #### BCE ###
            _, proj_u, zu = self.forward(X_u)

            prob2_ulb = F.softmax(proj_u, dim=1)
            #
            rank_feat = zu.detach()  # (feat[~mask_lb]).detach()
            rank_idx = torch.argsort(rank_feat, dim=1, descending=True)
            rank_idx1, rank_idx2 = PairEnum(rank_idx)
            rank_idx1, rank_idx2 = rank_idx1[:, :self.topk], rank_idx2[:, :self.topk]
            rank_idx1, _ = torch.sort(rank_idx1, dim=1)
            rank_idx2, _ = torch.sort(rank_idx2, dim=1)
            #
            rank_diff = rank_idx1 - rank_idx2
            rank_diff = torch.sum(torch.abs(rank_diff), dim=1)
            target_ulb = torch.ones_like(rank_diff).float()
            target_ulb[rank_diff > 0] = -1

            prob1_ulb, _ = PairEnum(prob2_ulb)  # PairEnum(prob2[~mask_lb])
            _, prob2_ulb = PairEnum(prob2_ulb)  # PairEnum(prob2_bar[~mask_lb])
            self.loss_bce = self.criterion2(prob1_ulb, prob2_ulb, target_ulb)

            loss = self.loss_ce + self.loss_bce
            self.optimizer.zero_grad()
            loss.backward()

            self.optimizer.step()


        self.pred_u_train = self.predict(X_u)
        self.centroid_u = [X_u[torch.tensor(self.pred_u_train) == cluster].mean(dim=0) for cluster in range(self.unlabelled_class)]

        self.centroid_s = [X_s[torch.tensor(ys) == cluster].mean(dim=0) for cluster in set_y_s]

    def predict(self, X, set_y=[], unsupervised=True):

        proj_test_s, proj_test_u, proj_all = self.forward(X)

        y_pred = torch.argmin(torch.cdist(proj_test_u, torch.eye(self.unlabelled_class)), dim=1).tolist()
        if unsupervised == False:
            y_pred = torch.argmin(torch.cdist(proj_test_s, torch.eye(self.labelled_class)), dim=1).tolist()

        return y_pred

    def predict_cluster(self, X, set_y=[], unsupervised=True):

        proj_test_s, proj_test_u, proj_all = self.forward(X)

        y_pred = torch.argmin(torch.cdist(proj_test_u, torch.eye(self.unlabelled_class)), dim=1).tolist()
        if unsupervised == False:
            y_pred = torch.argmin(torch.cdist(proj_test_s, torch.eye(self.labelled_class)), dim=1).tolist()

        return y_pred

    def re_index(self, y):
        #
        #set_y = [c for c in range(len(list(set(y))))]
        #liste_indice_lb_test = [set_y.index(i) for i in y]
        #liste_indice_lb_test = [list(set(y)).index(i) for i in y]
        try :
            yliste = y.detach().tolist()
        except :
            yliste = y
        sety = list(set(yliste))
        return [sety.index(i) for i in y]


    def param_grid(self):
        param_grid = {"lambda":  [1e-3] ,#[1e-4, 1e-3, 1e-2, 1e-3],  # 3
                      'lr': [1e-2],
                      'lambdaa_u':[1] ,# [ 5e-1, 1e0],
                      'topk' : [5]
                      }
        return param_grid

    def tiny_param_grid(self):
        param_grid = {"lambdaa": [1.e-4, 1.e-3],  # 3

                      }
        return param_grid

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def get_params(self, deep=True):
        return {
            "lambdaa": self.lambdaa_,  # lambda regression ridge
            "lambdaa_solution_regression": self.lambdaa_solution_regression,  # pour P la loss mu*Lxsi - (1-mu)*L_reg
            "distance": self.distance
        }







