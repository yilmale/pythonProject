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

def update(g,r):
    count = 0
    elms = list(g.nodes.keys())
    while (count <= r):
        e = elms[random.randrange(0,len(elms)-1)]
        elms.remove(e)
        g.remove_node(e)
        count += 1
    return g

def simulate_ER(n = 10, p=0.5, replications = 10):
    mydensity = []
    Sf=0
    f = 0.2
    for i in range(replications):
        random.seed()
        removeCnt = int(n*f)
        er = update(nx.erdos_renyi_graph(n, p),removeCnt)
        largest_cc = max(nx.connected_components(er), key=len)
        Sf=len(largest_cc)
        gd = nx.density(er)
        mydensity.append(gd)
    cum = 0
    for d in mydensity:
        cum = cum+ d
    density = cum/replications
    return {'density': density}


if __name__ == '__main__':
    ema_logging.LOG_FORMAT = '[%(name)s/%(levelname)s/%(processName)s] %(message)s'
    ema_logging.log_to_stderr(ema_logging.INFO)

    model = Model('SimulateER', function=simulate_ER)  # instantiate the model
    # specify uncertainties
    model.uncertainties = [RealParameter("p", 0.1, 1.0)]

    model.levers = [IntegerParameter("n", 10, 100)]

    # specify outcomes
    model.outcomes = [ScalarOutcome('density')]

    model.constants = [Constant('replications', 10)]

    n_scenarios = 10
    n_policies = 10

    #res = perform_experiments(models, n_scenarios, n_policies)

    with MultiprocessingEvaluator(model) as evaluator:
        res = evaluator.perform_experiments(n_scenarios, n_policies,
                                            levers_sampling=MC)

    experiments, outcomes = res
    data= experiments[['n', 'p']]

#-----------------------FeatureScoring-------------------------------------------
    z = feature_scoring.get_feature_scores_all(x=data,y=outcomes)

    print(z)
    print(z.at['n','density'])
    print(z.at['p', 'density'])

    z1 = feature_scoring.F_REGRESSION(X=data,y=outcomes['density'])
    print(z1)

    x1 = outcomes['density']
    print(data)

#-------------------PRIM------------------------------------
    transformedY =[]
    for i in range(x1.shape[0]):
        if x1[i] < 0.8:
            transformedY.append(0)
        else:
            transformedY.append(1)

    y1 = np.array(transformedY)

    print(y1)

    prim_alg = prim.Prim(data, x1, threshold=0.9, peel_alpha=0.1, mode = sdutil.RuleInductionType.REGRESSION)
    box1 = prim_alg.find_box()
    box1.show_tradeoff()
    box1.inspect(style='table')
    box1.inspect(7,style='table')
    #plt.show()





