import os

import torch

def mapping(solution, y_pred,set_y_u):
    return [set_y_u[solution[str(i)]] for i in y_pred]


def inv_mapping(solution, y_pred):
    return [solution[i] for i in y_pred]

def update_nb_voisin_supervised(set_s, set_pseudo_s, set_U, C):
    c = C.shape[0]
    dic_nb_voisin={}
    for node in set_U:
        nb_voisin_sup = 0
        # Search of supervised neighboor
        for value in range(c):
            if C[node, value] == 1 :
                if value in set_s or value in set_pseudo_s:
                    nb_voisin_sup +=1

        dic_nb_voisin[node] = nb_voisin_sup

    return  dic_nb_voisin

def find_nodes(dict_nb_neighboor,missing_number_supervised, dic_nb_voisin_all):
    liste_node = []
    for element in dict_nb_neighboor:
        if dic_nb_voisin_all[element]   - dict_nb_neighboor[element]    == missing_number_supervised :
            liste_node.append(element)
    return liste_node


def find_nodes_one_priorirty(dict_nb_neighboor,missing_number_supervised, dic_nb_voisin_all):

    liste_node_candidate = []
    for element in dict_nb_neighboor:
        if dic_nb_voisin_all[element] - dict_nb_neighboor[element] == missing_number_supervised:
            liste_node_candidate.append(element)

    ### SELECT MOST PROBABLE NODE
    # ici, choix, on prend les noeuds qui ont le plus de voisins ! (genre pas les feuills qui en ont que un

    liste_node_candidate_max = []
    try :
        nb_neighboor_max = max([dic_nb_voisin_all[i] for i in liste_node_candidate])

        for element in liste_node_candidate:
            if dic_nb_voisin_all[element] == nb_neighboor_max:
                liste_node_candidate_max.append(element)
    except:
        liste_node_candidate_max = liste_node_candidate
    return liste_node_candidate_max
def find_node_criteria(dict_current, miss_nb_sup, C, dic_nb_voisin_all):
    liste_node_candidate = find_nodes(dict_nb_neighboor=dict_current,
                                      missing_number_supervised=miss_nb_sup,
                                      dic_nb_voisin_all=dic_nb_voisin_all)
    ### SELECT MOST PROBABLE NODE
    # ici, choix, on prend les noeuds qui ont le plus de voisins ! (genre pas les feuills qui en ont que un
    nb_neighboor_max = max([dic_nb_voisin_all[i] for i in liste_node_candidate])
    liste_node_candidate_max = []
    for element in liste_node_candidate:
        if dic_nb_voisin_all[element] == nb_neighboor_max:
            liste_node_candidate_max.append(element)

    # le plus proche de la racine ave cpréférence pour la largeur #probleme du argmin
    node = liste_node_candidate_max[C[0,liste_node_candidate_max].argmin()]

    return node

#%%


def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))


def pairwise_dist(node,  # Node a assigner
                  set_Z,  # Intersection, either fully supervised or pseudo supervised
                  centroid_Z,  # Associated centroid of set_Z
                  centroid_U,

                  C,
                  matrix_ascendant_descendant,
                  ):  # Centroid for candidate

    set_y = list(set(range(C.shape[0])))

    all_pwdist_ponderate_pseudosup = torch.zeros((1, len(centroid_U)))
    if len(set_Z) > 0 :
        liste_ancetre = np.array(set_y)[matrix_ascendant_descendant[node] > 0].tolist()
        liste_descendant = np.array(set_y)[matrix_ascendant_descendant[:, node] > 0].tolist()

        liste_ancetre = intersection(liste_ancetre, set_Z)
        liste_descendant = intersection(liste_descendant, set_Z)

        index_ancestre = [set_Z.index(element) for element in liste_ancetre]
        index_descenant = [set_Z.index(element) for element in liste_descendant]

        centroid_ancestor = torch.tensor(centroid_Z)[index_ancestre]
        centroid_descendant = torch.tensor(centroid_Z)[index_descenant]

        pwdist_ancestor = torch.cdist(centroid_ancestor, torch.tensor(centroid_U))
        pwdist_descendant = torch.cdist(centroid_descendant, torch.tensor(centroid_U))
        ponderate_pwdist_ancestor = torch.div(pwdist_ancestor.T, C[node, liste_ancetre]).T
        ponderate_pwdist_descendant = torch.div(pwdist_descendant.T, C[node, liste_descendant]).T

        all_pwdist_ponderate_pseudosup = torch.cat((ponderate_pwdist_ancestor, ponderate_pwdist_descendant), dim=0)
    return all_pwdist_ponderate_pseudosup



#%%
def search_supervision(set_y_s, set_y_u,
                       C,
                       centroid_s,
                       centroid_u,
                       matrix_ascendant_descendant
                       ):

    ### INIT
    c = C.shape[0]
    dic_nb_voisin_all = {}
    for node in range(c):
        dic_nb_voisin_all[node] = C[node].tolist().count(1)

    solution = {}
    solution_inv = {}

    set_y_s_pseudo = []
    centroid_s_pseudo = []

    centroid_u_pseudo = centroid_u.copy()
    set_y_u_pseudo = set_y_u.copy()



    index_cluster_candidate = [i for i in range(len(set_y_u_pseudo))]


    #####
    nb_boucle = 0
    while len(set_y_u_pseudo) > 0 :
        # Find full supervised neibhoorhoog
        dict_current = update_nb_voisin_supervised(set_s=set_y_s, set_pseudo_s=set_y_s_pseudo, set_U=set_y_u_pseudo, C=C)
        try :
            node = find_node_criteria(miss_nb_sup=0, dict_current=dict_current, C=C, dic_nb_voisin_all=dic_nb_voisin_all)
        except :
            try :
                node = find_node_criteria(miss_nb_sup=1, dict_current=dict_current, C=C, dic_nb_voisin_all=dic_nb_voisin_all)
            except :
                try :
                    node = find_node_criteria(miss_nb_sup=2, dict_current=dict_current, C=C, dic_nb_voisin_all=dic_nb_voisin_all)
                except:
                    #print(('3eme niveau'))
                    node = find_node_criteria(miss_nb_sup=3, dict_current=dict_current, C=C, dic_nb_voisin_all=dic_nb_voisin_all)

        #print(node)
        #### NODE DEFINI PAR LISTE DE PRIORITE
        ### ASSIGN CLUSTER TO THIS NODE
        liste_all_node_cluster_value = np.array([parallel_crit(node=node,
                                                               C=C,
                                                               set_y_s=set_y_s,
                                                               set_y_s_pseudo=set_y_s_pseudo,
                                                               centroid_s=centroid_s,
                                                               centroid_s_pseudo=centroid_s_pseudo,
                                                               centroid_u_pseudo=centroid_u_pseudo,
                                                               matrix_ascendant_descendant=matrix_ascendant_descendant)

                                                ])

        index_argmin = liste_all_node_cluster_value[:, 2].argmin()

        node, index_cluster = int(liste_all_node_cluster_value[index_argmin, 0]), int(
            liste_all_node_cluster_value[index_argmin, 1])

        index_argmin = liste_all_node_cluster_value[:, 2].argmin()

        node, index_cluster = int(liste_all_node_cluster_value[index_argmin, 0]), int(
            liste_all_node_cluster_value[index_argmin, 1])

        cluster_candidate = index_cluster_candidate[index_cluster]
        # index_pseudo_u = index_cluster_pseudo[set_y_u.index(cluster_candidate)]
        solution[node] = cluster_candidate
        solution_inv[cluster_candidate] = node

        # print(cluster_candidate)

        # update y_s_pseudo
        set_y_s_pseudo.append(node)
        centroid_s_pseudo.append(centroid_u_pseudo[index_argmin])

        # remove y_u_pseudo
        set_y_u_pseudo.remove(node)
        centroid_u_pseudo.remove(centroid_u_pseudo[index_argmin])
        index_cluster_candidate.remove(cluster_candidate)

        # # Ici on prend en compte les pseudo supervisés aussi
        # liste_neighboor_sup = list(set(torch.where(C[node] < 3)[0].tolist()) & set(set_y_s))
        # liste_neighboor_pseudo_s = list(set(torch.where(C[node] < 3)[0].tolist()) & set(set_y_s_pseudo))
        #
        # ### CLUSTER NEIGHBORHOOD
        # centroid_s_neighboor = torch.tensor(centroid_s)[[set_y_s.index(element)for element in liste_neighboor_sup ]]
        # centroid_pseudo_s_neighboor = torch.tensor(centroid_s_pseudo)[[set_y_s_pseudo.index(element)for element in liste_neighboor_pseudo_s ]]
        #
        # pwdist = torch.cdist(
        #
        #     torch.cat((centroid_s_neighboor, centroid_pseudo_s_neighboor), dim=0),
        #     torch.tensor( centroid_u_pseudo)
        #                      )
        #
        #
        # ### UPDATE SUPERVISION WITH PSEUDO S
        #
        # index_argmin = pwdist.sum(dim=0).argmin()
        # cluster_candidate = index_cluster_candidate[index_argmin]
        #
        # #index_pseudo_u = index_cluster_pseudo[set_y_u.index(cluster_candidate)]
        #
        # solution[node] = cluster_candidate
        # solution_inv[cluster_candidate] =  node
        #
        # # update y_s_pseudo
        # set_y_s_pseudo.append(node)
        # centroid_s_pseudo.append(centroid_u_pseudo[index_argmin])
        #
        # # remove y_u_pseudo
        # set_y_u_pseudo.remove(node)
        # centroid_u_pseudo.remove(centroid_u_pseudo[index_argmin])
        # index_cluster_candidate.remove(index_cluster_candidate[index_argmin])
        # nb_boucle +=1

        if nb_boucle > len(set_y_u) + 1:
            break

    return solution, solution_inv





#%%
def parallel_node_attribution(node, C, set_y_s, set_y_s_pseudo, centroid_s,  centroid_s_pseudo , centroid_u_pseudo):
    liste_neighboor_sup = list(set(torch.where(C[node] < 3)[0].tolist()) & set(set_y_s))
    liste_neighboor_pseudo_s = list(set(torch.where(C[node] < 3)[0].tolist()) & set(set_y_s_pseudo))

    ### CLUSTER NEIGHBORHOOD
    centroid_s_neighboor = torch.tensor(centroid_s)[[set_y_s.index(element) for element in liste_neighboor_sup]]
    centroid_pseudo_s_neighboor = torch.tensor(centroid_s_pseudo)[
        [set_y_s_pseudo.index(element) for element in liste_neighboor_pseudo_s]]

    pwdist = torch.cdist(

        torch.cat((centroid_s_neighboor, centroid_pseudo_s_neighboor), dim=0),
        torch.tensor(centroid_u_pseudo)
    )

    weighted_pwdist_sum  = (pwdist / pwdist.shape[0]).sum(dim=0)

    cluster_index_candidat, value_attribution = weighted_pwdist_sum.argmin().detach().tolist(), weighted_pwdist_sum.min().detach().tolist()

    return [node, cluster_index_candidat, value_attribution]



def search_supervision_multiple(set_y_s, set_y_u,
                       C,
                       centroid_s,
                       centroid_u,
                                matrix_ascendant_descendant
                       ):

    ### INIT
    c = C.shape[0]
    dic_nb_voisin_all = {}
    for node in range(c):
        dic_nb_voisin_all[node] = C[node].tolist().count(1)

    solution = {}
    solution_inv = {}

    set_y_s_pseudo = []
    centroid_s_pseudo = []

    centroid_u_pseudo = centroid_u.copy()
    set_y_u_pseudo = set_y_u.copy()

    index_cluster_candidate = [i for i in range(len(set_y_u_pseudo))]

    #####
    nb_boucle = 0
    while len(set_y_u_pseudo) > 0 :

        dict_current = update_nb_voisin_supervised(set_s=set_y_s, set_pseudo_s=set_y_s_pseudo, set_U=set_y_u_pseudo,
                                                   C=C)

        liste_node = find_nodes_one_priorirty(dict_nb_neighboor=dict_current,
                                missing_number_supervised=0,
                                dic_nb_voisin_all=dic_nb_voisin_all)
        if len(liste_node) == 0:
            liste_node = find_nodes_one_priorirty(dict_nb_neighboor=dict_current,
                                    missing_number_supervised=1,
                                    dic_nb_voisin_all=dic_nb_voisin_all)
            if len(liste_node) == 0:
                liste_node = find_nodes_one_priorirty(dict_nb_neighboor=dict_current,
                                        missing_number_supervised=2,
                                        dic_nb_voisin_all=dic_nb_voisin_all)
                if len(liste_node) == 0:
                    liste_node = find_nodes_one_priorirty(dict_nb_neighboor=dict_current,
                                            missing_number_supervised=3,
                                            dic_nb_voisin_all=dic_nb_voisin_all)


        ### Compute potential assignement for each node and each cluster
        liste_all_node_cluster_value = np.array([parallel_crit(node=node,
                                                               C=C,
                                                               set_y_s=set_y_s,
                                                               set_y_s_pseudo=set_y_s_pseudo,
                                                               centroid_s=centroid_s,
                                                               centroid_s_pseudo=centroid_s_pseudo,
                                                               centroid_u_pseudo=centroid_u_pseudo,
                                                               matrix_ascendant_descendant=matrix_ascendant_descendant)

                                                 for node in liste_node])

        index_argmin = liste_all_node_cluster_value[:, 2].argmin()

        node, index_cluster = int(liste_all_node_cluster_value[index_argmin, 0]), int(
            liste_all_node_cluster_value[index_argmin, 1])

        index_argmin = liste_all_node_cluster_value[:, 2].argmin()

        node, index_cluster = int(liste_all_node_cluster_value[index_argmin, 0]), int(
            liste_all_node_cluster_value[index_argmin, 1])

        cluster_candidate = index_cluster_candidate[index_cluster]
        # index_pseudo_u = index_cluster_pseudo[set_y_u.index(cluster_candidate)]
        solution[node] = cluster_candidate
        solution_inv[cluster_candidate] = node

        #print(cluster_candidate)

        # update y_s_pseudo
        set_y_s_pseudo.append(node)
        centroid_s_pseudo.append(centroid_u_pseudo[index_argmin])

        # remove y_u_pseudo
        set_y_u_pseudo.remove(node)
        centroid_u_pseudo.remove(centroid_u_pseudo[index_argmin])
        index_cluster_candidate.remove(cluster_candidate)




        nb_boucle += 1

        if nb_boucle > len(set_y_u) + 1:
            break

    return solution, solution_inv



#%%







#%%


import numpy as np




#%%
def breadth_first_search(set_y_s,
                         set_y_u,

                        C,
                       centroid_s,
                       centroid_u,
                        matrix_ascendant_descendant,
                         lambdaa_pseudo_sup=1):
    c = C.shape[0]
    dic_nb_voisin_all = {}
    for node in range(c):
        dic_nb_voisin_all[node] = C[node].tolist().count(1)

    solution = {}
    solution_inv = {}

    set_y_s_pseudo = []
    centroid_s_pseudo = []


    centroid_u_pseudo = centroid_u.copy()
    set_y_u_pseudo = set_y_u.copy()

    index_cluster_pseudo = [i for i in range(len(set_y_u_pseudo))]


    for node in set_y_u:

        all_pwdist_ponderate_sup = pairwise_dist(node=node, set_Z=set_y_s, centroid_Z=centroid_s,
                                                 centroid_U=centroid_u_pseudo,
                                                 C=C,
                                                 matrix_ascendant_descendant= matrix_ascendant_descendant,

                                                 )

        all_pwdist_ponderate_pseudosup = lambdaa_pseudo_sup * pairwise_dist(node=node, set_Z=set_y_s_pseudo, centroid_Z=centroid_s_pseudo,
                                                           centroid_U=centroid_u_pseudo,

                                                                                C=C,
                                                                                matrix_ascendant_descendant=matrix_ascendant_descendant,
                                                                                )




        if all_pwdist_ponderate_pseudosup.shape[0] < 1:
            all_pwdist_ponderate_pseudosup = torch.zeros(all_pwdist_ponderate_sup.shape)

        all_pwdist = torch.cat((all_pwdist_ponderate_sup,  all_pwdist_ponderate_pseudosup), dim=0)
        best_index_current_cluster = all_pwdist.sum(dim=0).argmin().detach().tolist()
        cluster_attribution = index_cluster_pseudo[best_index_current_cluster]

        #mappin: node -> index_cluster

        # SOLUTION

        solution[str(node)] = cluster_attribution
        solution_inv[cluster_attribution] = node

        # UPDATE PSEUDO SET
        set_y_s_pseudo.append(set_y_u_pseudo[best_index_current_cluster])
        centroid_s_pseudo.append(centroid_u_pseudo[best_index_current_cluster])

        index_cluster_pseudo.remove(cluster_attribution)
        set_y_u_pseudo.remove(set_y_u_pseudo[best_index_current_cluster])
        centroid_u_pseudo.remove(centroid_u_pseudo[best_index_current_cluster])


    return solution, solution_inv

#%%
def parallel_crit(node,
                  centroid_s, centroid_s_pseudo ,
                  centroid_u_pseudo,
                  matrix_ascendant_descendant,
                  C,
                  set_y_s, set_y_s_pseudo
                  ):
    all_pwdist_ponderate_sup = pairwise_dist(node=node, set_Z=set_y_s, centroid_Z=centroid_s,
                                             centroid_U=centroid_u_pseudo,
                                             C=C,
                                             matrix_ascendant_descendant=matrix_ascendant_descendant,

                                             )

    all_pwdist_ponderate_pseudosup = 1 * pairwise_dist(node=node, set_Z=set_y_s_pseudo,
                                                                        centroid_Z=centroid_s_pseudo,
                                                                        centroid_U=centroid_u_pseudo,

                                                                        C=C,
                                                                        matrix_ascendant_descendant=matrix_ascendant_descendant,
                                                                        )

    if all_pwdist_ponderate_pseudosup.shape[0] < 1:
        all_pwdist_ponderate_pseudosup = torch.zeros(all_pwdist_ponderate_sup.shape)

    all_pwdist = torch.cat((all_pwdist_ponderate_sup, all_pwdist_ponderate_pseudosup), dim=0)

    weighted_pwdist_sum = (all_pwdist / all_pwdist.shape[0]).sum(dim=0)

    cluster_index_candidat, value_attribution = weighted_pwdist_sum.argmin().detach().tolist(), weighted_pwdist_sum.min().detach().tolist()


    return [node, cluster_index_candidat, value_attribution]


def bfs_parallel(set_y_s,
                         set_y_u,

                        C,
                       centroid_s,
                       centroid_u,
                        matrix_ascendant_descendant,
                         lambdaa_pseudo_sup=1):
    c = C.shape[0]
    dic_nb_voisin_all = {}
    for node in range(c):
        dic_nb_voisin_all[node] = C[node].tolist().count(1)

    solution = {}
    solution_inv = {}

    set_y_s_pseudo = []
    centroid_s_pseudo = []

    centroid_u_pseudo = centroid_u.copy()
    set_y_u_pseudo = set_y_u.copy()

    index_cluster_candidate = [i for i in range(len(set_y_u_pseudo))]

    hauteur, nb_boucle = 0, 0


    while len(set_y_u_pseudo) > 0 :



        liste_node = intersection(set_y_u_pseudo, torch.tensor(list(range(c)) )[C[0] == hauteur].tolist() )
        if len(liste_node) > 0 :

            liste_all_node_cluster_value = np.array([parallel_crit(node=node,
                                                                               C=C,
                                                                               set_y_s=set_y_s,
                                                                               set_y_s_pseudo=set_y_s_pseudo,
                                                                               centroid_s=centroid_s,
                                                                               centroid_s_pseudo=centroid_s_pseudo,
                                                                               centroid_u_pseudo=centroid_u_pseudo,
                                                                   matrix_ascendant_descendant=matrix_ascendant_descendant)

                                                     for node in liste_node])

            index_argmin = liste_all_node_cluster_value[:, 2].argmin()

            node, index_cluster = int(liste_all_node_cluster_value[index_argmin, 0]), int(
                liste_all_node_cluster_value[index_argmin, 1])



            cluster_candidate = index_cluster_candidate[index_cluster]
            # index_pseudo_u = index_cluster_pseudo[set_y_u.index(cluster_candidate)]
            solution[node] = cluster_candidate
            solution_inv[cluster_candidate] = node

            # print(cluster_candidate)
            centroid_affected = centroid_u_pseudo[index_argmin]


            # update y_s_pseudo
            set_y_s_pseudo.append(node)
            centroid_s_pseudo.append(centroid_affected)

            # remove y_u_pseudo
            set_y_u_pseudo.remove(node)
            centroid_u_pseudo.remove(centroid_affected)
            index_cluster_candidate.remove(cluster_candidate)


        else :
                hauteur +=1



        nb_boucle += 1
        # if nb_boucle > len(set_y_u) + 1:
        #     break
    #print(len(set_y_u_pseudo))

    return solution, solution_inv

#%%

#%%
if __name__ == '__main__':
    PATH = os.getcwd()
    dataset = 'binary4'
    t=0
    from load_data import load_all_data
    #%%
    path_data = PATH + '/datasets/'+str((dataset))+'/'
    (X, X_train, y_train, X_test, y_test,
     X_s_train, X_s_test, y_s_train, y_s_test,
     X_u_train, X_u_test, y_u_train, y_u_test,
     set_y_s, set_y_u, C, matrix_parents, matrix_ascendant_descendant) = load_all_data(PATH=PATH, dataset=dataset, t=0, proportion_u=0.1)
    Xs , Xu = torch.cat((X_s_train, X_s_test), dim=0), torch.cat((X_u_train, X_u_test), dim=0)
    ys, yu = y_s_train + y_s_test, y_u_train + y_u_test
    d, c = X.shape[1], C.shape[0],
    set_y = list(range(c))
    n_components = c

    from sklearn.cluster import KMeans
    model = KMeans(n_clusters=len(set_y_u)).fit(Xu)
    pred_u_train = model.labels_
    pred_u_test = model.predict(Xu)
    # model = GaussianMixture(n_components=len(set_y_u), random_state=0).fit(X_u_train)
    # pred_u_train = model.predict(X_u_train)
    centroid_s = [X[torch.where(torch.tensor(y_train) == cluster)].mean(dim=0).detach().tolist() for cluster in set_y_s]
    centroid_u = [Xu[pred_u_train == cluster].mean(dim=0).detach().tolist() for cluster in range(len(set_y_u))]
    #%%
    solution, solution_inv = bfs_parallel(
        set_y_s=set_y_s,
        set_y_u=set_y_u,
        C=C,
        matrix_ascendant_descendant=matrix_ascendant_descendant,
        centroid_s=centroid_s,
        centroid_u=centroid_u)

    solution, solution_inv = breadth_first_search(
        set_y_s=set_y_s,
        set_y_u=set_y_u,
        C=C,
        matrix_ascendant_descendant=matrix_ascendant_descendant,
        centroid_s=centroid_s,
        centroid_u=centroid_u)


    solution, solution_inv = search_supervision(
        set_y_s=set_y_s,
        set_y_u=set_y_u,
        C=C,
        matrix_ascendant_descendant=matrix_ascendant_descendant,
        centroid_s=centroid_s,
        centroid_u=centroid_u)

    solution, solution_inv = search_supervision_multiple(
        set_y_s=set_y_s,
        set_y_u=set_y_u,
        C=C,
        matrix_ascendant_descendant=matrix_ascendant_descendant,
        centroid_s=centroid_s,
        centroid_u=centroid_u)


#f1_bfs_multiple = f1_score(y_true=y_u_test, y_pred=mapping_u_test, average='micro')
#%%
#
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd
# import torch
# import torch.nn as nn
# from torch.distributions import multivariate_normal as mn
# import os
# import random
# from load_data import load_all_data
#
#
# #%%
# proportion_u = 0.5  # 0.1, 0.25, 0.5
#
# dataset = 'binary4'  #'binary4'  # 'binary_6'  # Planaria, Paul, binary_6
# PATH = os.getcwd()
# path_file_abs = PATH
# random.seed = 1312
# method = 'hPhi'
#
# #%%
#
# all_results_t = []
# t = 4
# path_data = PATH + '/datasets/'+str((dataset))+'/'
#
# (X, X_train, y_train, X_test, y_test,
#          X_s_train, X_s_test, y_s_train, y_s_test,
#          X_u_train, X_u_test, y_u_train, y_u_test,
#          set_y_s, set_y_u, C, matrix_parents, matrix_ascendant_descendant) = load_all_data(PATH=PATH, dataset=dataset, t=t, proportion_u=proportion_u)
#
# centroid_s = [X_s_train[torch.where(torch.tensor(y_s_train) == cluster)].mean(dim=0).detach().tolist()
#                        for cluster in set_y_s]
# c = C.shape[0]
# from model import hPhi
# import itertools
#
# model =  hPhi(matrix_parents=matrix_parents,
#                         d=50,
#                         C=C,
#                      )
#
#
# model.fit(Xs=X_s_train,
#                   Xu=X_u_train,
#                   ys=y_s_train,
#                   set_y_s=set_y_s,
#                   set_y_u=set_y_u)
#
# pred_u = model.predict(X=X_u_test, set_y=set_y_u)
#
# #%%
# from sklearn.metrics import f1_score
# print(f1_score(y_u_test, pred_u, average='micro'))
# centroid_u = model.centroid_u.detach().tolist()
#
# #%%
#
#
# solution, solution_inv = search_supervision_multiple(set_y_s=set_y_s,
#                                                      set_y_u=set_y_u,
#                                                      C=C,
#                                                      centroid_s=centroid_s,
#                                                      centroid_u=centroid_u)
# #%%
# mapping_u_test = inv_mapping(solution_inv, model.predict_cluster(X=X_u_train, set_y=set_y_u))
# f1_search_multiple = f1_score(y_true=y_u_train, y_pred=mapping_u_test, average='micro')
#%%
#
#
#
# #%%  DEBUG
#
# c = C.shape[0]
# dic_nb_voisin_all = {}
# for node in range(c):
#     dic_nb_voisin_all[node] = C[node].tolist().count(1)
#
# solution = {}
# solution_inv = {}
#
# set_y_s_pseudo = []
# centroid_s_pseudo = []
#
# centroid_u_pseudo = centroid_u.copy()
# set_y_u_pseudo = set_y_u.copy()
#
# index_cluster_candidate = [i for i in range(len(set_y_u_pseudo))]
#
# #####
# nb_boucle = 0
# #%%
# # Find full supervised neibhoorhoog
# dict_current = update_nb_voisin_supervised(set_s=set_y_s, set_pseudo_s=set_y_s_pseudo, set_U=set_y_u_pseudo, C=C)
#
# liste_node = find_nodes(dict_nb_neighboor=dict_current,
#                         missing_number_supervised=0,
#                         dic_nb_voisin_all=dic_nb_voisin_all)
# if len(liste_node) == 0:
#     liste_node = find_nodes(dict_nb_neighboor=dict_current,
#                             missing_number_supervised=1,
#                             dic_nb_voisin_all=dic_nb_voisin_all)
#     if len(liste_node) == 0:
#         liste_node = find_nodes(dict_nb_neighboor=dict_current,
#                                 missing_number_supervised=2,
#                                 dic_nb_voisin_all=dic_nb_voisin_all)
#         if len(liste_node) == 0:
#             liste_node = find_nodes(dict_nb_neighboor=dict_current,
#                                     missing_number_supervised=3,
#                                     dic_nb_voisin_all=dic_nb_voisin_all)
#
#
# # print(node)
#
# ### Compute potential assignement for each node and each cluster
#
# # Ici on prend en compte les pseudo supervisés aussi
#
# liste_all_node_cluster_value = np.array([parallel_node_attribution(node=node,
#                                                                    C=C,
#                                                                    set_y_s=set_y_s,
#                                                                    set_y_s_pseudo=set_y_s_pseudo,
#                                                                    centroid_s=centroid_s,
#                                                                    centroid_s_pseudo=centroid_s_pseudo,
#                                                                    centroid_u_pseudo=centroid_u_pseudo)
#
#                                          for node in liste_node])
#
# index_argmin = liste_all_node_cluster_value[:, 2].argmin()
#
# node, index_cluster = int(liste_all_node_cluster_value[index_argmin, 0]), int(
#     liste_all_node_cluster_value[index_argmin, 1])
#
# cluster_candidate = index_cluster_candidate[index_cluster]
# # index_pseudo_u = index_cluster_pseudo[set_y_u.index(cluster_candidate)]
# solution[node] = cluster_candidate
# solution_inv[cluster_candidate] = node
#
# # update y_s_pseudo
# set_y_s_pseudo.append(node)
# centroid_s_pseudo.append(centroid_u_pseudo[index_argmin])
#
# # remove y_u_pseudo
# set_y_u_pseudo.remove(node)
# centroid_u_pseudo.remove(centroid_u_pseudo[index_argmin])
# index_cluster_candidate.remove(index_cluster_candidate[index_argmin])
# nb_boucle += 1


#%%

# set_y = list(set(range(c)))
# #%%  DEV ALGO BFS
#
# lambdaa_pseudo_sup = 1
# dic_nb_voisin_all = {}
# for node in range(c):
#     dic_nb_voisin_all[node] = C[node].tolist().count(1)
#
# solution = {}
# solution_inv = {}
#
# set_y_s_pseudo = []
# centroid_s_pseudo = []
#
# centroid_u_pseudo = centroid_u.copy()
# set_y_u_pseudo = set_y_u.copy()
#
# index_cluster_pseudo = [i for i in range(len(set_y_u_pseudo))]
#
# for node in set_y_u:
#
#     all_pwdist_ponderate_sup = pairwise_dist(node=node,
#                                              set_Z=set_y_s,
#                                              centroid_Z=centroid_s,
#                                              centroid_U=centroid_u_pseudo,
#                                              C=C,
#                                              matrix_ascendant_descendant=matrix_ascendant_descendant,
#
#                                              )
#
#     all_pwdist_ponderate_pseudosup = lambdaa_pseudo_sup * pairwise_dist(node=node,
#                                                                         set_Z=set_y_s_pseudo,
#                                                                         centroid_Z=centroid_s_pseudo,
#                                                                         centroid_U=centroid_u_pseudo,
#
#                                                                         C=C,
#                                                                         matrix_ascendant_descendant=matrix_ascendant_descendant,
#                                                                         )
#
#
#     if all_pwdist_ponderate_pseudosup.shape[0] < 1:
#         all_pwdist_ponderate_pseudosup = torch.zeros(all_pwdist_ponderate_sup.shape)
#
#     all_pwdist = torch.cat((all_pwdist_ponderate_sup, all_pwdist_ponderate_pseudosup), dim=0)
#     best_index_current_cluster = all_pwdist.sum(dim=0).argmin().detach().tolist()
#     cluster_attribution = index_cluster_pseudo[best_index_current_cluster]
#
#     # mappin: node -> index_cluster
#
#     # SOLUTION
#
#     solution[str(node)] = cluster_attribution
#     solution_inv[cluster_attribution] = node
#
#     # UPDATE PSEUDO SET
#     set_y_s_pseudo.append(set_y_u_pseudo[best_index_current_cluster])
#     centroid_s_pseudo.append(centroid_u_pseudo[best_index_current_cluster])
#
#     index_cluster_pseudo.remove(index_cluster_pseudo[best_index_current_cluster])
#     set_y_u_pseudo.remove(set_y_u_pseudo[best_index_current_cluster])
#     centroid_u_pseudo.remove(centroid_u_pseudo[best_index_current_cluster])
#
# #%%





#
#
#
#
#
# #%%
# #%%
# solution, solution_inv = breadth_first_search(
#                                                       set_y_s=set_y_s,
#                                                       set_y_u=set_y_u,
#                                                       C=C,
#                                                       matrix_ascendant_descendant=matrix_ascendant_descendant,
#                                                       centroid_s=centroid_s,
#                                                       centroid_u=centroid_u)
#
#
# #%%
# mapping_u_test = inv_mapping(solution_inv, model.predict_cluster(X=X_u_test, set_y=set_y_u))
# f1_bfs = f1_score(y_true=y_u_test, y_pred=mapping_u_test, average='micro')