import networkx as nx
import random

def update(g,r):
    count = 0
    elms = list(g.nodes.keys())
    while (count <= r):
        e = elms[random.randint(0,len(elms)-1)]
        elms.remove(e)
        g.remove_node(e)
        count += 1
    return g

def simulate_ER(n = 10, p=0.5, replications = 100,f=0.3):
    outcomes = []
    Sf=0
    for i in range(replications):
        random.seed()
        removeCnt = int(n*f)
        er = update(nx.erdos_renyi_graph(n, p),removeCnt)
        largest_cc = max(nx.connected_components(er), key=len)
        Sf=len(largest_cc)
        outcomes.append(Sf/n)
    return outcomes


out = simulate_ER()

print(sum(out)/len(out))