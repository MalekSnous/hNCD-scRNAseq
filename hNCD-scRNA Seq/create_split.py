


import numpy as np
import os
import random
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from sklearn.model_selection import train_test_split

import networkx as nx



for lenght in [3,4,6]:

    path = '/home/malek/hNCD-git/datasets/binary' + str(lenght)

    mat_dist = np.load(path + '/mat_dist.npy')


    c = mat_dist.shape[0]
    proj_tree = MDS(n_components=2, dissimilarity='precomputed').fit_transform(mat_dist)

    G = nx.DiGraph()
    for node in range(c):
        G.add_node(node, pos=proj_tree[node].tolist())
        #G.add_node(node, pos=centroid_all[node])
    for u in range(c):
        for v in range(u, c):
            if mat_dist[u, v] == 1:
                # G.add_edge(u,v)
                # G.add_edge(v,u)
                G.add_edges_from([(u, v, {"weight": 1})])
                G.add_edges_from([(v, u, {"weight": 1})])

    try :
        pos = nx.get_node_attributes(G, 'pos')
    except :
        pos = nx.spring_layout(G)

    #%%
    def print_graph(color_array_ncd):
        fig = plt.plot()
        nx.draw_networkx_nodes(G, pos, node_color=color_array_ncd, node_size=100)
        nx.draw_networkx_edges(G, pos, edge_color="grey", style='dashed', arrows=None)
        nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif")
        plt.axis("off")
        return fig


    fig0 = print_graph(['r' for i in range(c)])


    plt.show()


    for proportion_u in [0.1,0.25,0.5]:
        path_ncd = path + '/ncd/' + str(proportion_u)
        os.makedirs(path_ncd, exist_ok=True)
        set_y = list(range(c))

        for t in range(10):
            set_y_u = random.sample(list(range(1,c)), k=int(proportion_u*c))
            np.save(path_ncd+'/set_u_'+str(t)+'.npy', set_y_u)
            #fig = plt.figure()
            color_y = ['r' if i not in set_y_u else 'g' for i in range(c)]
            fig = print_graph(color_y)

            plt.scatter(proj_tree[:,0], proj_tree[:,1], color=color_y)
            plt.savefig(path_ncd+'/proj_mds'+str(t)+'.jpg')
            #print(set_y_u)




#%%
for lenght in [3, 4, 6]:
    path = '/home/malek/hNCD-git/datasets/binary' + str(lenght)


    X = np.load(path + '/sample/X.npy', allow_pickle=True)
    y = np.load(path + '/sample/y.npy', allow_pickle=True).tolist()



    for proportion_u in [0.1,0.25,0.5]:
        path_ncd = path + '/ncd/' + str(proportion_u)
        set_y = list(range(c))

        for t in range(10):
            set_y_u = sorted(np.load(path_ncd+'/set_u_'+str(t)+'.npy').tolist())
            set_y_s = list(set(set_y) - set(set_y_u))
            set_y_u = [str(element) for element in set_y_u]

            set_y_s = [str(element) for element in set_y_s]
            path_t = path_ncd + '/'+str(t)
            os.makedirs(path_t, exist_ok=True)

            indice_s = np.array(range(len(y)))[np.isin(y, set_y_s)].tolist()
            indice_u = np.array(range(len(y)))[np.isin(y, set_y_u)].tolist()

            train_s, test_s = train_test_split(indice_s, stratify=np.array(y)[indice_s].tolist(), test_size=0.2)
            train_u, test_u = train_test_split(indice_u, stratify=np.array(y)[indice_u].tolist(), test_size=0.2)

            np.save(path_t+'/train_s', train_s)
            np.save(path_t + '/test_s', test_s)
            np.save(path_t + '/train_u', train_u)
            np.save(path_t + '/test_u', test_u)


#%%

#%%  BRANCHES

lenght = 7


path = '/home/malek/hNCD-git/datasets/half' + str(lenght)

mat_dist = np.load(path + '/mat_dist.npy')


c = mat_dist.shape[0]
proj_tree = MDS(n_components=2, dissimilarity='precomputed').fit_transform(mat_dist)

G = nx.DiGraph()
for node in range(c):
    G.add_node(node, pos=proj_tree[node].tolist())
    #G.add_node(node, pos=centroid_all[node])
for u in range(c):
    for v in range(u, c):
        if mat_dist[u, v] == 1:
            # G.add_edge(u,v)
            # G.add_edge(v,u)
            G.add_edges_from([(u, v, {"weight": 1})])
            G.add_edges_from([(v, u, {"weight": 1})])

try :
    pos = nx.get_node_attributes(G, 'pos')
except :
    pos = nx.spring_layout(G)

#%%
def print_graph(color_array_ncd):
    fig = plt.plot()
    nx.draw_networkx_nodes(G, pos, node_color=color_array_ncd, node_size=100)
    nx.draw_networkx_edges(G, pos, edge_color="grey", style='dashed', arrows=None)
    nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif")
    plt.axis("off")
    return fig


fig0 = print_graph(['r' for i in range(c)])


plt.show()


for proportion_u in [0.1,0.25,0.5]:
    path_ncd = path + '/ncd/' + str(proportion_u)
    os.makedirs(path_ncd, exist_ok=True)
    set_y = list(range(c))

    for t in range(10):
        set_y_u = random.sample(list(range(1,c)), k=int(proportion_u*c))
        np.save(path_ncd+'/set_u_'+str(t)+'.npy', set_y_u)
        #fig = plt.figure()
        color_y = ['r' if i not in set_y_u else 'g' for i in range(c)]
        fig = print_graph(color_y)

        plt.scatter(proj_tree[:,0], proj_tree[:,1], color=color_y)
        plt.savefig(path_ncd+'/proj_mds'+str(t)+'.jpg')
        #print(set_y_u)




#%%



X = np.load(path + '/sample/X.npy', allow_pickle=True)
y = np.load(path + '/sample/y.npy', allow_pickle=True).tolist()



for proportion_u in [0.1,0.25,0.5]:
    path_ncd = path + '/ncd/' + str(proportion_u)
    set_y = list(range(c))

    for t in range(10):
        set_y_u = sorted(np.load(path_ncd+'/set_u_'+str(t)+'.npy').tolist())
        set_y_s = list(set(set_y) - set(set_y_u))
        set_y_u = [str(element) for element in set_y_u]

        set_y_s = [str(element) for element in set_y_s]
        path_t = path_ncd + '/'+str(t)
        os.makedirs(path_t, exist_ok=True)

        indice_s = np.array(range(len(y)))[np.isin(y, set_y_s)].tolist()
        indice_u = np.array(range(len(y)))[np.isin(y, set_y_u)].tolist()

        train_s, test_s = train_test_split(indice_s, stratify=np.array(y)[indice_s].tolist(), test_size=0.2)
        train_u, test_u = train_test_split(indice_u, stratify=np.array(y)[indice_u].tolist(), test_size=0.2)

        np.save(path_t+'/train_s', train_s)
        np.save(path_t + '/test_s', test_s)
        np.save(path_t + '/train_u', train_u)
        np.save(path_t + '/test_u', test_u)





#%%
from load_data import reordonate_matrix_dist

#%% PAUL

dataset = 'Paul'
path = '/home/malek/hNCD-git/datasets/Paul'

X = np.load(path + '/sample/X.npy', allow_pickle=True)
y = np.load(path + '/sample/y.npy', allow_pickle=True).tolist()
mat_dist = np.load(path + '/mat_dist.npy')
c = mat_dist.shape[0]
y_names = np.load(path + '/'+str(dataset) + '_names.npy', allow_pickle=True).tolist()



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

### CORRECTION METTR ELA RACINE EN PREMIERE COLONNE

C, old_dict = reordonate_matrix_dist(mat_dist, y_names, root_indice)
y = [old_dict[element] for element in new_y ]


proj_tree = MDS(n_components=2, dissimilarity='precomputed').fit_transform(C)

#%%

G = nx.DiGraph()
for node in range(c):
    G.add_node(node, pos=proj_tree[node].tolist(), name=y_names[node])

for u in range(c):
    for v in range(u, c):
        if C[u, v] == 1:

            G.add_edges_from([(u, v, {"weight": 1})])
            G.add_edges_from([(v, u, {"weight": 1})])

try:
    pos = nx.get_node_attributes(G, 'pos')
except:
    pos = nx.spring_layout(G)

def print_graph(color_array_ncd):
    fig = plt.plot()
    nx.draw_networkx_nodes(G, pos, node_color=color_array_ncd, node_size=100, )
    nx.draw_networkx_edges(G, pos, edge_color="grey", style='dashed', arrows=None)
    nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif")
    plt.axis("off")
    return fig


fig0 = print_graph(['r' for i in range(c)])

plt.show()
#%%
for proportion_u in [0.1, 0.25, 0.5]:
    path_ncd = path + '/ncd/' + str(proportion_u)
    os.makedirs(path_ncd, exist_ok=True)
    set_y = list(set(y))

    for t in range(10):
        set_y_u = random.sample(list(range(1, max(set_y))), k=int(proportion_u * c))
        np.save(path_ncd + '/set_u_' + str(t) + '.npy', set_y_u)
        # fig = plt.figure()
        color_y = ['r' if i not in set_y_u else 'g' for i in range(c)]
        fig = print_graph(color_y)

        plt.scatter(proj_tree[:, 0], proj_tree[:, 1], color=color_y)
        plt.savefig(path_ncd + '/proj_mds' + str(t) + '.jpg')
        # print(set_y_u)


#%%
for proportion_u in [0.1,0.25,0.5]:
    path_ncd = path + '/ncd/' + str(proportion_u)
    set_y = list(set(y))

    for t in range(10):
        set_y_u = sorted(np.load(path_ncd+'/set_u_'+str(t)+'.npy').tolist())
        set_y_s = list(set(set_y) - set(set_y_u))
        #set_y_u = [str(element) for element in set_y_u]

        #set_y_s = [str(element) for element in set_y_s]
        path_t = path_ncd + '/'+str(t)
        os.makedirs(path_t, exist_ok=True)

        indice_s = np.array(range(len(y)))[np.isin(y, set_y_s)].tolist()
        indice_u = np.array(range(len(y)))[np.isin(y, set_y_u)].tolist()

        train_s, test_s = train_test_split(indice_s, stratify=np.array(y)[indice_s].tolist(), test_size=0.2)
        train_u, test_u = train_test_split(indice_u, stratify=np.array(y)[indice_u].tolist(), test_size=0.2)

        np.save(path_t+'/train_s', train_s)
        np.save(path_t + '/test_s', test_s)
        np.save(path_t + '/train_u', train_u)
        np.save(path_t + '/test_u', test_u)



########################""


#%% PLANARIA




dataset = 'Planaria'
path = '/home/malek/hNCD-git/datasets/Planaria'

X = np.load(path + '/sample/X.npy', allow_pickle=True)
y = np.load(path + '/sample/y.npy', allow_pickle=True).tolist()
mat_dist = np.load(path + '/mat_dist.npy')
c = mat_dist.shape[0]
y_names = np.load(path + '/'+str(dataset) + '_names.npy', allow_pickle=True).tolist()


new_y = []

root_indice = y_names.index('neoblast 1')
for element in y:
    if element == root_indice:
        new_y.append(0)
    if element == 0:
        new_y.append(root_indice)
    if element != 0 and element != root_indice:
        new_y.append(element)
### CORRECTION METTR ELA RACINE EN PREMIERE COLONNE
C, old_dict = reordonate_matrix_dist(mat_dist, y_names, root_indice)
y = [old_dict[element] for element in new_y ]


proj_tree = MDS(n_components=2, dissimilarity='precomputed').fit_transform(C)

#%%

G = nx.DiGraph()
for node in range(c):
    G.add_node(node, pos=proj_tree[node].tolist(), name=y_names[node])

for u in range(c):
    for v in range(u, c):
        if C[u, v] == 1:

            G.add_edges_from([(u, v, {"weight": 1})])
            G.add_edges_from([(v, u, {"weight": 1})])

try:
    pos = nx.get_node_attributes(G, 'pos')
except:
    pos = nx.spring_layout(G)

def print_graph(color_array_ncd):
    fig = plt.plot()
    nx.draw_networkx_nodes(G, pos, node_color=color_array_ncd, node_size=100, )
    nx.draw_networkx_edges(G, pos, edge_color="grey", style='dashed', arrows=None)
    nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif")
    plt.axis("off")
    return fig


fig0 = print_graph(['r' for i in range(c)])

plt.show()
#%%
for proportion_u in [0.1, 0.25, 0.5]:
    path_ncd = path + '/ncd/' + str(proportion_u)
    os.makedirs(path_ncd, exist_ok=True)
    set_y = list(set(y))

    for t in range(10):
        set_y_u = random.sample(list(range(1, max(set_y))), k=int(proportion_u * c))
        np.save(path_ncd + '/set_u_' + str(t) + '.npy', set_y_u)
        # fig = plt.figure()
        color_y = ['r' if i not in set_y_u else 'g' for i in range(c)]
        fig = print_graph(color_y)

        plt.scatter(proj_tree[:, 0], proj_tree[:, 1], color=color_y)
        plt.savefig(path_ncd + '/proj_mds' + str(t) + '.jpg')
        # print(set_y_u)


#%%
for proportion_u in [0.1,0.25,0.5]:
    path_ncd = path + '/ncd/' + str(proportion_u)
    set_y = list(set(y))

    for t in range(10):
        set_y_u = sorted(np.load(path_ncd+'/set_u_'+str(t)+'.npy').tolist())
        set_y_s = list(set(set_y) - set(set_y_u))
        #set_y_u = [str(element) for element in set_y_u]

        #set_y_s = [str(element) for element in set_y_s]
        path_t = path_ncd + '/'+str(t)
        os.makedirs(path_t, exist_ok=True)

        indice_s = np.array(range(len(y)))[np.isin(y, set_y_s)].tolist()
        indice_u = np.array(range(len(y)))[np.isin(y, set_y_u)].tolist()

        train_s, test_s = train_test_split(indice_s, stratify=np.array(y)[indice_s].tolist(), test_size=0.2)
        train_u, test_u = train_test_split(indice_u, stratify=np.array(y)[indice_u].tolist(), test_size=0.2)

        np.save(path_t+'/train_s', train_s)
        np.save(path_t + '/test_s', test_s)
        np.save(path_t + '/train_u', train_u)
        np.save(path_t + '/test_u', test_u)



