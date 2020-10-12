import networkx as nx
import matplotlib.pyplot as plt
import random


default_INACTIVE = 0.0
default_ACTIVE = 1.0
default = 0.01
MAX = 1.0
MIN = -1.0
Delta = 0.005
Threshold = 0.30
RED = "#ff474c"
BLUE = "#95d0fc"



class FCM:
    def __init__(self,nodes,dependencies,input):
        self.A_t={}
        self.G = nx.DiGraph()
        for n in nodes:
            self.G.add_node(n[0], activation=n[1], clamped=False)
            self.A_t[n[0]]=n[1]
        for e in dependencies:
            self.G.add_edge(e[0], e[1], weight=e[2])
        self.I = input

    def updateActivations(self):
        for n in self.G.nodes:
            self.A_t[n] = self.G.nodes[n]['activation']
        delta = 0.0
        for n in self.G.nodes:
            d = self.updateNodeActivation(n)
            delta = max(delta,d)
        return delta

    def hardThreshold(self,net,n):
        if net + self.I[n] > 0:
            return 1.0
        else:
            return 0.0

    def updateNodeActivation(self,n):
        beforeUpdate = self.A_t[n]
        if self.G.nodes[n]['clamped'] == False:
            net = 0.0
            adj = self.G.pred[n]
            for m in adj.keys():
                net = net + (adj[m]['weight'] * self.A_t[m])
            self.G.nodes[n]['activation'] = self.hardThreshold(net,n)
        diff = abs(self.G.nodes[n]['activation'] - beforeUpdate)
        return diff

    def updateA_t(self):
        for n in self.G.nodes:
            self.A_t[n] = self.G.nodes[n]['activation']

    def simulate_synchronous(self):
        for i in range(0,10):
            self.updateActivations()
            self.updateA_t()
            print(self.getActivations())

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



input = {'C1': 0.0, 'C2': 0.0, 'C3': 0.0, 'C4': 0.0, 'C5': 0.0}
nodes = [("C1", default_INACTIVE),("C2", default_INACTIVE),
         ("C3", default_INACTIVE),("C4", default_ACTIVE),("C5", default_INACTIVE)]

dependencies = [("C1", "C2", 1.0), ("C1", "C4", -1.0), ("C2", "C3", 1.0),
                        ("C2", "C5", -1.0),("C3", "C4", 1.0), ("C3", "C5", -1.0),
                        ("C4", "C1", 1.0), ("C4", "C3", -1.0),("C4", "C5", 1.0),
                        ("C5", "C1", -1.0), ("C5", "C2", 1.0),("C5", "C4", -1.0)]

fcm = FCM(nodes,dependencies,input)
print('Initial activations: ', fcm.getActivations())
print('Simulating....')
fcm.simulate_synchronous()


''' 
plt.figure(figsize=[15,10])
plt.subplot(121)
nx.draw(fcm.G, with_labels=True, pos = nx.planar_layout(fcm.G), font_weight='bold')
plt.show()
'''