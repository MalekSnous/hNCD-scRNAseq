import os
import anndata as ad
from scanpy import pp
import numpy as np
from numpy import random
import scipy as sp
import matplotlib.cm as cm
import matplotlib.pyplot as plt

#set random seed for reproducability

#%%


#%%
from prosstt import tree
from prosstt import simulation as sim
from prosstt import sim_utils as sut
from scanpy.tl import umap
from prosstt import tree


#%%

PATH = os.getcwd()
rseed = 1312
random.seed(rseed)
save = True


dataset_number = 1
topology_list = ['linear', 'binary', 'half']
topology = topology_list[dataset_number]
lengh_tree_liste = [5, 6, 10] # [254,8,21] papier plos
lenght_tree = lengh_tree_liste[dataset_number]



topology = 'half'
lenght_tree = 7   #


topology = 'half'
lenght_tree = 14   #120 label








topology = 'binary'
lenght_tree = 3    #15 label

topology = 'binary'
lenght_tree = 4    #31 label

topology = 'binary'
lenght_tree = 6  #127 label



topology = 'binary'
lenght_tree = 3    #31 label

topology = 'half'
lenght_tree = 6   #



#
value = 0.1
modules = 100  # nb de programme genetique
#modules = 50

draw = True
multi_draw = False


nb_genes = 100

value_list = [0.1,0.5,1.0]  # alpha in NB sampling


print('module:', modules, 'lenght of tree :', lenght_tree, 'nb_genes :', nb_genes, 'alpha_value :',value,  'topology :', topology)

#%%

number_tree = topology
nfactor = 4  # nb_sample_per_node = nfactor*10
name_tree = '_' + str(modules) + '_' + str(value)
name_file_to_save = PATH + '/data/final_Tree/' + str(topology) + '/' + str(lenght_tree) + '/'
if save:
    os.makedirs(name_file_to_save, exist_ok=True)


def create_tree(lenght=5, topology='binary'):
    list_edge = []
    dict_edges = {}

    if topology == 'binary':
        list_of_nodes = [element for element in range(sum([2 ** i for i in range(lenght + 1)]))]

        # init
        dict_edges[list_of_nodes[0]] = [list_of_nodes[1], list_of_nodes[2]]
        list_edge.append([str(list_of_nodes[0]), str(list_of_nodes[1])])
        list_edge.append([str(list_of_nodes[0]), str(list_of_nodes[2])])

        for t in range(1, lenght):

            nodes_parent = list_of_nodes[2 ** (t - 1) - 1: 2 ** (t) - 1]
            nodes = list_of_nodes[2 ** t - 1: 2 ** (t + 1) - 1]
            nodes_child = list_of_nodes[2 ** (t + 1) - 1: 2 ** (t + 2) - 1]
            print(nodes_parent, nodes, nodes_child)

            for indice, node in enumerate(nodes):
                liste_descendant = [nodes_child[2 * indice], nodes_child[2 * indice + 1]]
                liste_ascendante = [nodes_parent[int(indice / 2)]]
                dict_edges[node] = liste_descendant + liste_ascendante
                list_edge.append([str(node), str(nodes_child[2 * indice])])
                list_edge.append([str(node), str(nodes_child[2 * indice + 1])])

        for indice, node in enumerate(nodes):
            dict_edges[nodes_child[2 * indice]] = [node]
            dict_edges[nodes_child[2 * indice + 1]] = [node]

    if topology == 'linear':
        list_of_nodes = [i for i in range(lenght + 1)]

        # init
        dict_edges[list_of_nodes[0]] = [list_of_nodes[1]]
        list_edge.append([str(list_of_nodes[0]), str(list_of_nodes[1])])

        for t in range(1, lenght):
            nodes_parent = list_of_nodes[t - 1]
            nodes = list_of_nodes[t]
            nodes_child = list_of_nodes[t + 1]

            liste_of_neighboor = [nodes_parent, nodes_child]
            dict_edges[nodes] = liste_of_neighboor
            list_edge.append([str(nodes), str(nodes_child)])

        dict_edges[t] = [list_of_nodes[t - 1], list_of_nodes[t + 1]]
        dict_edges[t + 1] = [list_of_nodes[t]]

    if topology == 'half':
        list_of_nodes = [element for element in range(sum([1 + i for i in range(lenght + 1)]))]
        dict_edges = {}
        list_edge = []
        # init
        dict_edges[list_of_nodes[0]] = [list_of_nodes[1], list_of_nodes[2]]
        list_edge.append([str(list_of_nodes[0]), str(list_of_nodes[1])])
        list_edge.append([str(list_of_nodes[0]), str(list_of_nodes[2])])

        for t in range(1, lenght):
            nodes_parent = list_of_nodes[sum([1 + i for i in range(t - 1)]): sum([1 + i for i in range(t)])]
            nodes = list_of_nodes[sum([1 + i for i in range(t)]): sum([1 + i for i in range(t + 1)])]
            nodes_child = list_of_nodes[sum([1 + i for i in range(t + 1)]): sum([1 + i for i in range(t + 2)])]

            # print(nodes_parent, nodes, nodes_child)

            for indice, node in enumerate(nodes):
                if indice == 0:
                    liste_descendant = [nodes_child[indice], nodes_child[indice + 1]]
                    liste_ascendante = [nodes_parent[indice]]
                    list_edge.append([str(node), str(nodes_child[2 * indice])])
                    list_edge.append([str(node), str(nodes_child[2 * indice + 1])])
                if indice != 0:
                    liste_descendant = [nodes_child[indice + 1]]
                    if indice == 1:
                        liste_ascendante = [nodes_parent[int(indice / 2)]]
                    else:
                        liste_ascendante = [nodes_parent[indice - 1]]
                    list_edge.append([str(node), str(nodes_child[indice + 1])])

                dict_edges[node] = liste_descendant + liste_ascendante

        for indice, node in enumerate(nodes):
            if indice == 0:
                dict_edges[nodes_child[indice]] = [node]
            dict_edges[nodes_child[indice + 1]] = [node]

    return list_edge, dict_edges


list_edge, dict_edges = create_tree(lenght=lenght_tree, topology=topology)

#%%

top = list_edge
branches = np.unique(np.array(top).flatten())
time = {b: nfactor*10 for b in branches}  #ce qui revient au nombre de cell généré car ça définit la densité dans le sampling
time = {b: 20 for b in branches}
G = nb_genes
t = tree.Tree(topology=top, G=G, time=time, num_branches=len(branches), branch_points=1, modules=modules)


uMs, Ws, Hs = sim.simulate_lineage(t, intra_branch_tol=-1, inter_branch_tol=0)
gene_scale = sut.simulate_base_gene_exp(t, uMs)
t.add_genes(uMs, gene_scale)

mya = np.min([0.05, 1 / t.modules])
uMs, Ws, Hs = sim.simulate_lineage(t, a=mya, intra_branch_tol=-1, inter_branch_tol=0)
gene_scale = sut.simulate_base_gene_exp(t, uMs)
t.add_genes(uMs, gene_scale)

alpha = np.array([value]*t.G)
beta = np.array([1.5] * t.G)

X, labs, brns, scalings = sim.sample_whole_tree(t, n_factor=nfactor, alpha=alpha, beta=beta)
X = (X.transpose() / scalings).transpose()

br_names, indices = np.unique(brns, return_inverse=True)

#%%
#

#%%
if draw :
    data = ad.AnnData(np.log(X + 1))
    pp.neighbors(data, use_rep='X')
    umap(data)
    dm = data.obsm["X_umap"]
    fig = plt.figure()
    dm = data.obsm["X_umap"]
    br_names, indices = np.unique(brns, return_inverse=True)
    plt.scatter(dm[:, 0], dm[:, 1], c=labs, alpha=0.2)
    plt.show()
    if save :
        plt.savefig(name_file_to_save+'X_Tree'+str(name_tree)+'.jpg')
        plt.savefig(name_file_to_save + 'X_Tree' + str(name_tree) + '.pdf')

#%%

for dim in [5,10,15,20,25,30,40,50] :
#dim= 20
    feature_selection = random.sample(list(range(100)), dim)
    X_selection = X[:, feature_selection]
    data = ad.AnnData(np.log(X_selection + 1))
    pp.neighbors(data, use_rep='X')
    umap(data)
    dm = data.obsm["X_umap"]

    fig = plt.figure()
    plt.scatter(dm[:, 0], dm[:, 1], c=labs, alpha=0.8)
    fig.savefig(name_file_to_save + 'X_Tree' + str(dim) + '.pdf')

    plt.show()
#%%
if save:
    plt.savefig(name_file_to_save + 'X_Tree' + str(name_tree) + '.jpg')
    plt.savefig(name_file_to_save + 'X_Tree' + str(name_tree) + '.pdf')
#%%


set_names = br_names.tolist()
X, y = X, brns

import networkx as nx
def create_C(dict_edges):
    G = nx.Graph(dict_edges)
    nodes = list(G.nodes)
    c = len(nodes)
    mat_dist = np.zeros((c, c))
    for i in range(c):
        for j in range(i + 1, c):
            mat_dist[i, j] = len(nx.shortest_path(G, source=i, target=j)) -1
    return mat_dist + mat_dist.T

mat_dist = create_C(dict_edges)

#%%

print(X.shape)
print(len(y))
print(mat_dist.shape)
#print(nfactor)

#nb sample per label, it should be close to 1200 for partial label train test split
print([list(y).count(str(c)) for c in range(10)])
print(list(y).count('0'))

print(name_file_to_save)


save = True
name_tree = '_'+str(modules)+'_'+str(value)+'_nfactor_'+str(nfactor)
name_tree

if save :
    os.makedirs(name_file_to_save, exist_ok=True)
    np.save(name_file_to_save+'X_Tree'+str(name_tree), X)
    np.save(name_file_to_save+'y_Tree'+str(name_tree), y)
    np.save(name_file_to_save+'Tree'+str(name_tree)+'_mat_dist', mat_dist)
    np.save(name_file_to_save+'Tree'+str(name_tree)+'_pseudotime', labs)