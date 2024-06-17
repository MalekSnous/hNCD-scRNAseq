
import numpy as np

import torch
from model import Regular_EM, EM_Gmm_nn, Phi_reg, Autonovel, hierarchical_pairwise_error


def find_dict_params(method, matrix_hierarchy, matrix_flat):
    if method == 'Phi':
        dict_params = {'lr': 5e-1, 'lambda_eps': 1e-2, 'lambdaa_u': 1e-1, 'matrix_parents': matrix_hierarchy}
    if method == 'Phi_flat':
        dict_params = {'lr': 1e-1, 'lambda_eps': 1e-2, 'lambdaa_u': 1e-1, 'matrix_parents': matrix_flat}

    if method == 'GMM':
        dict_params = {'matrix_parents': matrix_hierarchy, 'matrix_parents_true': matrix_hierarchy,
                       'lambdaa': 5e2, 'lr': 5e-1, 'sub_epoch': 20
                       }
    if method == 'Autonovel':
        dict_params = {'lr': 1e-2, 'lambda': 1e-3, 'lambdaa_u': 1e1, 'topk': 4}

    if method == 'GMM_flat':
        dict_params = {}

    return dict_params


def init_model(method, dict_params, matrix_parents_true, d ,X_init=[], y_init=[], NS=[],NU=[]):
    if method == 'Phi' or method == 'Phi_flat':
        model = Phi_reg(matrix_parents=dict_params['matrix_parents'],
                        matrix_parents_true=matrix_parents_true,
                        d=d,
                        C=torch.tensor([1]),
                        lr=dict_params['lr'],
                        lambda_eps=dict_params['lambda_eps'],
                        lambdaa_u=dict_params['lambdaa_u'])

    if method == 'GMM':
        pre_model = Phi_reg(matrix_parents=dict_params['matrix_parents'],
                            matrix_parents_true=matrix_parents_true,
                            d=d,
                            C=torch.tensor([1]),
                            lr=dict_params['lr'],
                            lambda_eps=dict_params['lambdaa'],
                            lambdaa_u=1)

        pre_model.fit(X=X_init,
                      y=y_init,
                      max_epochs=101)

        mu_soluce = pre_model.new_Phi

        model = EM_Gmm_nn(matrix_parents=dict_params['matrix_parents'],
                          matrix_parents_true=matrix_parents_true,
                          d=d,
                          lambdaa=dict_params['lambdaa'],
                          lr=dict_params['lr'],
                          sub_epoch=dict_params['sub_epoch'])
        # mu_init = X_train.mean(dim=0) + mu_random
        model.init(mu_init=mu_soluce)

    if method == 'Autonovel':
        model = Autonovel(d=d,
                          NS=NS,
                          NU=NU,
                          hidden_layer_d=50,
                          lambdaa=1e-3,
                          lr=1e-2,
                          topk=5, )

    if method == 'GMM_flat':

        model = Regular_EM(d=d, n_components=matrix_parents_true.shape[0])

        dict_params_init = find_dict_params('Phi_flat')
        pre_model = Phi_reg(matrix_parents=dict_params_init['matrix_parents'],
                            matrix_parents_true=matrix_parents_true,
                            d=d,
                            C=torch.tensor([1]),
                            lr=dict_params_init['lr'],
                            lambda_eps=dict_params_init['lambda_eps'],
                            lambdaa_u=1)
        pre_model.fit(X=X_init,
                      y=y_init,
                      max_epochs=51)


        mu_soluce = pre_model.new_Phi
        model.init(mu_init=mu_soluce)

    return model