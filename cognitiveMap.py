import networkx as nx
import matplotlib.pyplot as plt
import random

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
        d = updateNodeActivation(n)
        delta = max(delta,d)
    return(delta)

def updateNodeActivation(n):
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



G = nx.Graph()

G.add_node("A", activation=default_ACTIVE, clamped=True)
G.add_node("B", activation=default_INACTIVE, clamped=True)
G.add_node("C", activation=default, clamped=False)
G.add_node("D", activation=default, clamped=False)
G.add_node("E", activation=default, clamped=False)

G.add_edge("A","C",weight=1.0)
G.add_edge("B","C",weight=-1.0)
G.add_edge("C","E",weight=1.0)
G.add_edge("B","D",weight=1.0)
G.add_edge("D","E",weight=-1.0)

print('Initial activations: ', getActivations(G))

edgeColor = setEdgeColors(G)

initialColorCode =setNodeColors(G)

activations = coherenceMaximizer(G)
print(activations)


finalColorCode = setNodeColors(G)


plt.subplot(121)
nx.draw(G, with_labels=True, node_color=initialColorCode,
        edge_color= edgeColor, pos = nx.planar_layout(G), font_weight='bold')
plt.subplot(122)
nx.draw(G, with_labels=True, node_color=finalColorCode,
        edge_color= edgeColor, pos = nx.planar_layout(G), font_weight='bold')
plt.show()



