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



problem = {
    'num_vars': 3,
    'names': ['n', 'p', 'f'],
    'bounds': [[10, 100],
               [0.1, 0.9],
               [0.1, 0.7]]
}

param_values = saltelli.sample(problem, 50)
Y = np.zeros([param_values.shape[0]])

for i, X in enumerate(param_values):
    Y[i] = simulate_ER(X[0],X[1],X[2],replications=30)

Si = sobol.analyze(problem,Y)
print(Si['S1'])

for v in Si.items():
    print(v[1])






