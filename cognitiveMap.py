from __future__ import (absolute_import, print_function, division,
                        unicode_literals)
import matplotlib.pyplot as plt

import networkx as nx
import random
from ema_workbench import (Model, RealParameter, MultiprocessingEvaluator, CategoricalParameter,
                           IntegerParameter, ScalarOutcome, ArrayOutcome, Constant, ema_logging,
                           perform_experiments)
from ema_workbench.em_framework.evaluators import MC

from ema_workbench.analysis import feature_scoring
from ema_workbench.analysis import prim

from ema_workbench.analysis import scenario_discovery_util as sdutil

import pandas as pd
import seaborn as sns

import numpy as np

default_INACTIVE = 0.0
default_ACTIVE = 1.0
default = 0.01
decay = 0.05
MAX = 1.0
MIN = -1.0
Delta = 0.005
Threshold = 0.30
RED = "#ff474c"
BLUE = "#95d0fc"


def updateActivations(G):
    delta = 0.0
    for n in G.nodes:
        d = updateNodeActivation(G,n)
        delta = max(delta,d)
    return(delta)

def updateNodeActivation(G,n):
    beforeUpdate = G.nodes[n]['activation']
    if G.nodes[n]['clamped'] == False:
        net = 0.0
        adj = G.adj[n]
        for m in adj.keys():
            net = net + (adj[m]['weight']*G.nodes[m]['activation'])
        if (net > 0):
            G.nodes[n]['activation'] = min(1.0,(G.nodes[n]['activation']*(1-decay)) + \
                                       (net*(MAX-G.nodes[n]['activation'])))
        else:
            G.nodes[n]['activation'] = max(-1.0,(G.nodes[n]['activation']*(1 - decay)) + \
                                       (net*(G.nodes[n]['activation']-MIN)))
    diff = abs(G.nodes[n]['activation']-beforeUpdate)
    return diff

def getActivations(G):
    nodeActivations = {}
    for n in G.nodes:
        nodeActivations[n] = G.nodes[n]['activation']
    return nodeActivations

def coherenceMaximizer(G):
    diff = 1.0
    iterations = 200
    while (iterations > 0) and (diff > Delta):
        diff = updateActivations(G)
        iterations = iterations - 1
    return getActivations(G)

def setEdgeColors(G):
    edgeColor = []
    for e in list(G.edges):
        src = e[0]
        tgt = e[1]
        if G.adj[src][tgt]['weight'] > 0:
            edgeColor.append(BLUE)
        else:
            edgeColor.append(RED)
    return edgeColor

def setNodeColors(G):
    colorCode = []
    for n in G.nodes:
        if (G.nodes[n]['activation'] > Threshold):
            colorCode.append(RED)
        else:
            colorCode.append(BLUE)
    return colorCode

def createPairs(nodes,w):
    if (len(nodes) == 1):
        pairs = []
    else:
        h = nodes.pop(0)
        remainingNodes = nodes.copy()
        pairs = createPairs(nodes,w)
        ps=[]
        for n in remainingNodes:
            ps.append((h,n,w))
        pairs = ps + pairs
    return pairs

def addEdge(G,nodes,target,ew):
    for n in nodes:
        G.add_edge(n,target,weight=ew/len(nodes))
    pairs=createPairs(nodes,ew/len(nodes))
    for i,j,w in pairs:
        G.add_edge(i,j,weight=w)


def simulate_CM(a = 0.0, b=0.0):
    G = nx.Graph()
    G.clear()
    G.add_node("A", activation=a, clamped=True)
    G.add_node("B", activation=b, clamped=True)
    G.add_node("C", activation=default, clamped=False)
    G.add_node("D", activation=default, clamped=False)
    G.add_node("E", activation=default, clamped=False)

    addEdge(G, ["C", "D"], "A", 1.0)
    G.add_edge("E", "B", weight=1.0)
    activations = coherenceMaximizer(G)
    c = activations["C"]
    d = activations["D"]
    e = activations["E"]
    return {'c': c, 'd': d, 'e': e}


if __name__ == '__main__':
    ema_logging.LOG_FORMAT = '[%(name)s/%(levelname)s/%(processName)s] %(message)s'
    ema_logging.log_to_stderr(ema_logging.INFO)

    model = Model('SimulateCM', function=simulate_CM)  # instantiate the model
    # specify uncertainties
    model.uncertainties = [RealParameter("a", 0.1, 1.0)]

    model.levers = [RealParameter("b", 0.0, 0.01)]

    # specify outcomes
    model.outcomes = [ScalarOutcome('c'),
                      ScalarOutcome('d'),
                      ScalarOutcome('e')]

    #model.constants = [Constant('replications', 10)]

    n_scenarios = 10
    n_policies = 10

    res = perform_experiments(model, n_scenarios, n_policies)

    experiments, outcomes = res

    print(experiments)
    print(outcomes)

''' 
edgeColor = setEdgeColors(G)
finalColorCode = setNodeColors(G)

plt.figure(figsize=[15,10])
plt.subplot(121)
nx.draw(G, with_labels=True, node_color=initialColorCode,
        edge_color= edgeColor, pos = nx.planar_layout(G), font_weight='bold',
        node_size=[1000,300,300,300,300])
plt.subplot(122)
nx.draw(G, with_labels=True, node_color=finalColorCode,
        edge_color= edgeColor, pos = nx.planar_layout(G), font_weight='bold')
plt.show()

fm = FM('ParameterFeatureTree',[
          Mandatory([FM('context',
                        [Mandatory(FM('X',Alternative([P('A'),P('B')]))),
                        Optional([P('b'),P('q'),P('mean')])])]),
          Mandatory([FM('design',
                        [Optional([P('delta'),P('stddev')])])]),
          Mandatory([FM('outcomes',
                        [Optional([P('max_P'),P('utility'),P('inertia'),P('reliability')])])]),
          ]
         )
'''
