
import numpy as np
import torch
import os
import random
from sklearn import preprocessing



#%%

# from sklearn.feature_selection import SelectFromModel
# from sklearn.svm import LinearSVC
#
# PATH = os.getcwd()
#
#
# for dataset in ['half7','binary4','binary6','Paul', 'Planaria']:
#     for proportion_u in [0.1,0.25,0.5]:
#
#
#         random.seed = 1312
#         path_data = PATH + '/datasets/' + str((dataset)) + '/'
#
#         X = np.load(path_data + 'sample/X.npy', allow_pickle=True)
#         y = np.load(path_data + 'sample/y.npy', allow_pickle=True).tolist()
#         X = np.vstack(X).astype(np.float64)
#         lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
#         model_fs = SelectFromModel(lsvc, prefit=True, max_features=10)
#
#         X = torch.FloatTensor(model_fs.transform(X))
#
#         np.save(path_data + 'sample/X_fs_10.npy',X)

#%%



def load_all_data(PATH, dataset, t, proportion_u, print_=False):

    random.seed = 1312
    path_data = PATH + '/datasets/' + str((dataset)) + '/'

    X = np.load(path_data + 'sample/X_fs_10.npy', allow_pickle=True)
    y = np.load(path_data + 'sample/y.npy', allow_pickle=True).tolist()
    mat_dist = np.load(path_data + 'mat_dist.npy')

    if dataset in ['Paul', 'Planaria']:
        y_names = np.load(path_data + str(dataset) + '_names.npy', allow_pickle=True).tolist()

    if dataset == 'Paul':
        X, y = X[:-1], y[:-1]
        new_y = []
        root_indice = y_names.index('root')
        for element in y:
            if element == root_indice:
                new_y.append(0)
            if element == 0:
                new_y.append(root_indice)
            if element != 0 and element != root_indice:
                new_y.append(element)

    if dataset == 'Planaria':
        new_y = []
        root_indice = y_names.index('neoblast 1')
        for element in y:
            if element == root_indice:
                new_y.append(0)
            if element == 0:
                new_y.append(root_indice)
            if element != 0 and element != root_indice:
                new_y.append(element)

    if dataset in ['binary3', 'binary4', 'binary6','half7']:
        y = np.array(y).astype(('int')).tolist()
        root_indice = 0

    C = torch.Tensor(mat_dist)
    if dataset in ['Paul', 'Planaria']:

        C, old_dict = reordonate_matrix_dist(mat_dist, y_names, root_indice)
        C = torch.Tensor(C)
        y = [old_dict[element] for element in y ]

    X = np.vstack(X).astype(np.float64)

    X = torch.FloatTensor(X)
    if print_:
        print('X shape : ', X.shape, 'len y : ', len(y))
    c = C.shape[0]

    matrix_parents = torch.eye(c)
    list_parents = [[root_indice]]
    #C = torch.tensor(C)
    for label in range(1, c):
        liste_to_root = [label]
        dist_to_root = C[label, 0]

        new_label = label
        r = 1
        while C[new_label, root_indice] > 0:

            cols = torch.where(C[new_label] == 1)[0]


            if len(cols) > 1:
                new_label = cols[np.argmin(C[root_indice, cols])].detach().tolist()
                liste_to_root.append(new_label)
            else:
                new_label = cols.detach().tolist()[0]
                liste_to_root.append(new_label)

            matrix_parents[label, new_label] = 1 #  Variante / (2 ** r)

            r += 1
        list_parents.append(liste_to_root)

    matrix_ascendant_descendant = matrix_parents - torch.eye(c)



    path_t = path_data + 'ncd/' + str(proportion_u) + '/' + str(t) + '/'
    set_y_u = np.load(path_data + 'ncd/' + str(proportion_u) + '/set_u_' + str(t) + '.npy').tolist()
    set_y_s = list(set(list(range(c))) - set(set_y_u))

    set_y_s = sorted(set_y_s)
    set_y_u = sorted(set_y_u)

    train_s, test_s, = (np.load(path_t + 'train_s.npy', allow_pickle=True).tolist(),
                        np.load( path_t + 'test_s.npy', allow_pickle=True).tolist())
    train_u, test_u = (np.load(path_t + 'train_u.npy', allow_pickle=True).tolist(),
                       np.load(path_t + 'test_u.npy', allow_pickle=True).tolist())

    X_s_train, X_s_test, y_s_train, y_s_test = (torch.FloatTensor(X[train_s]),
                                                torch.FloatTensor(X[test_s]),
                                                np.array(y)[train_s].tolist(),
                                                np.array(y)[test_s].tolist())

    X_u_train, X_u_test, y_u_train, y_u_test = (torch.FloatTensor(X[train_u]),
                                                torch.FloatTensor(X[test_u]),
                                                np.array(y)[train_u].tolist(),
                                                np.array(y)[test_u].tolist())


    X_s_train, X_u_train = torch.split(
        torch.FloatTensor(preprocessing.StandardScaler().fit_transform(torch.cat((X_s_train, X_u_train), dim=0))),
        [X_s_train.size(0), X_u_train.size(0)])
    X_s_test, X_u_test = torch.split(
        torch.FloatTensor(preprocessing.StandardScaler().fit_transform(torch.cat((X_s_test, X_u_test), dim=0))),
        [X_s_test.size(0), X_u_test.size(0)])
    X_train = torch.cat((X_s_train, X_u_train), dim=0)
    y_train = y_s_train + [-1] * X_u_train.shape[0]
    X_test = torch.cat((X_s_test, X_u_test), dim=0)
    y_test = y_s_test + [-1] * X_u_test.shape[0]

    return  X, X_train, y_train, X_test, y_test, X_s_train, X_s_test, y_s_train, y_s_test, X_u_train, X_u_test, y_u_train, y_u_test, set_y_s,set_y_u, C, matrix_parents, matrix_ascendant_descendant





def reordonate_matrix_dist(mat_dist, y_names, indice_root) :
    c = mat_dist.shape[0]

    old_dict = {}
    dict_generation = {}  # clé : génération value : tous les noeuds de cette generation
    all_nodes_ordered = []

    max_generation = int(max(mat_dist[indice_root]).tolist())
    new_index = 0
    for generation in range(max_generation + 1):
        liste_generation = []
        for node in range(c):
            if mat_dist[indice_root, node] == generation:
                liste_generation.append(node)
                old_dict[node] = new_index
                new_index +=1

        dict_generation[generation] = liste_generation
        all_nodes_ordered = all_nodes_ordered + liste_generation

    ####
    new_matrix = np.zeros(np.shape(mat_dist))
    for row, node in enumerate(all_nodes_ordered):
        other_nodes = all_nodes_ordered[row + 1:]
        value = mat_dist[node, other_nodes]
        new_matrix[row, row + 1:] = value

    return new_matrix + new_matrix.T , old_dict





#%%


