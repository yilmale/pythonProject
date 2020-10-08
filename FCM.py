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



class FCM:
    def __init__(self):
        self.G = nx.Graph()
        self.G.add_node("A", activation=default_ACTIVE, clamped=True)
        self.G.add_node("B", activation=default_INACTIVE, clamped=True)
        self. G.add_node("C", activation=default, clamped=False)
        self.G.add_node("D", activation=default, clamped=False)
        self.G.add_node("E", activation=default, clamped=False)

        self.G.add_edge("A", "C", weight=1.0)
        self.G.add_edge("B", "C", weight=-1.0)
        self.G.add_edge("C", "E", weight=1.0)
        self.G.add_edge("B", "D", weight=1.0)
        self.G.add_edge("D", "E", weight=-1.0)

    def updateActivations(self):
        delta = 0.0
        for n in self.G.nodes:
            d = self.updateNodeActivation(n)
            delta = max(delta,d)
        return delta

    def updateNodeActivation(self,n):
        beforeUpdate = self.G.nodes[n]['activation']
        if self.G.nodes[n]['clamped'] == False:
            net = 0.0
            adj = self.G.adj[n]
            for m in adj.keys():
                net = net + (adj[m]['weight'] * self.G.nodes[m]['activation'])
            if (net > 0):
                self.G.nodes[n]['activation'] = min(1.0, (self.G.nodes[n]['activation'] * (1 - decay)) + \
                                               (net * (MAX - self.G.nodes[n]['activation'])))
            else:
                self.G.nodes[n]['activation'] = max(-1.0, (self.G.nodes[n]['activation'] * (1 - decay)) + \
                                               (net * (self.G.nodes[n]['activation'] - MIN)))
        diff = abs(self.G.nodes[n]['activation'] - beforeUpdate)
        return diff

    def getActivations(self):
        nodeActivations = {}
        for n in self.G.nodes:
            nodeActivations[n] = self.G.nodes[n]['activation']
        return nodeActivations

    def setEdgeColors(self):
        edgeColor = []
        for e in list(self.G.edges):
            src = e[0]
            tgt = e[1]
            if self.G.adj[src][tgt]['weight'] > 0:
                edgeColor.append(BLUE)
            else:
                edgeColor.append(RED)
        return edgeColor

    def setNodeColors(self):
        colorCode = []
        for n in self.G.nodes:
            if (self.G.nodes[n]['activation'] > Threshold):
                colorCode.append(RED)
            else:
                colorCode.append(BLUE)
        return colorCode


fcm = FCM()
print('Initial activations: ', fcm.getActivations())

plt.figure(figsize=[15,10])
plt.subplot(121)
nx.draw(fcm.G, with_labels=True, pos = nx.planar_layout(fcm.G), font_weight='bold')
plt.show()