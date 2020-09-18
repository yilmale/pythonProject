import networkx as nx
import random

from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami
import numpy as np
import matplotlib.pyplot as plt

def update(g,r):
    count = 0
    elms = list(g.nodes.keys())
    while (count <= r):
        e = elms[random.randint(0,len(elms)-1)]
        elms.remove(e)
        g.remove_node(e)
        count += 1
    return g

def simulate_ER(n = 10, p=0.5, f=0.3, replications = 30,):
    outcomes = []
    Sf=0
    for i in range(replications):
        random.seed()
        removeCnt = int(n*f)
        er = update(nx.erdos_renyi_graph(int(n), p),removeCnt)
        largest_cc = max(nx.connected_components(er), key=len)
        Sf=len(largest_cc)
        outcomes.append(Sf/n)
    return sum(outcomes)/len(outcomes)


def simulate_PreferentialAttachment(n=10, m=5, f=0.3, replications = 30):
    outcomes = []
    Sf = 0
    for i in range(int(replications)):
        random.seed()
        removeCnt = int(n * f)
        pa = update(nx.barabasi_albert_graph(n, m),removeCnt)
        largest_cc = max(nx.connected_components(pa), key=len)
        Sf = len(largest_cc)
        outcomes.append(Sf / n)
    return sum(outcomes) / len(outcomes)

def simulate_WattsStrogatz(n=10, k=3, p=0.5, f=0.3, replications = 30):
    outcomes = []
    Sf = 0
    for i in range(int(replications)):
        random.seed()
        removeCnt = int(n * f)
        pa = update(nx.watts_strogatz_graph(n,k,p),removeCnt)
        largest_cc = max(nx.connected_components(pa), key=len)
        Sf = len(largest_cc)
        outcomes.append(Sf / n)
    return sum(outcomes) / len(outcomes)



problem = {
    'num_vars': 5,
    'names': ['n', 'p', 'm', 'f', 'k'],
    'bounds': [[30, 100],
               [0.1, 0.9],
               [2, 29],
               [0.1, 0.7],
               [5,10]]
}

param_values = saltelli.sample(problem, 50)
Y = np.zeros([param_values.shape[0]])
Z = np.zeros([param_values.shape[0]])
W = np.zeros([param_values.shape[0]])

for i, X in enumerate(param_values):
    Y[i] = simulate_ER(X[0],X[1],X[3],replications=30)
    #Z[i] = simulate_PreferentialAttachment(int(X[0]),int(X[2]),X[3],replications=30)
    #W[i] = simulate_PreferentialAttachment(int(X[0]), int(X[4]), X[1], replications=30)

Si = sobol.analyze(problem,Y)
print(Si['S1'])

print("x1-x2:", Si['S2'][0,1])
print("x1-x3:", Si['S2'][0,2])
print("x2-x3:", Si['S2'][1,2])


