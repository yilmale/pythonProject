import math

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.optimize import brentq
from ema_workbench.analysis import prim

from ema_workbench.analysis import feature_scoring


from ema_workbench import (Model, RealParameter, ScalarOutcome, Constant,
                           ema_logging, MultiprocessingEvaluator)
from ema_workbench.em_framework.evaluators import MC


def lake_problem(
    b=0.42,          # decay rate for P in lake (0.42 = irreversible)
    q=2.0,           # recycling exponent
    mean=0.02,       # mean of natural inflows
    stdev=0.001,     # future utility discount rate
    delta=0.98,      # standard deviation of natural inflows
    alpha=0.4,       # utility from pollution
    nsamples=100,    # Monte Carlo sampling of natural inflows
        **kwargs):
    try:
        decisions = [kwargs[str(i)] for i in range(100)]
    except KeyError:
        decisions = [0, ] * 100

    Pcrit = brentq(lambda x: x**q / (1 + x**q) - b * x, 0.01, 1.5)
    nvars = len(decisions)
    X = np.zeros((nvars,))
    average_daily_P = np.zeros((nvars,))
    decisions = np.array(decisions)
    reliability = 0.0

    for _ in range(nsamples):
        X[0] = 0.0

        natural_inflows = np.random.lognormal(
            math.log(mean**2 / math.sqrt(stdev**2 + mean**2)),
            math.sqrt(math.log(1.0 + stdev**2 / mean**2)),
            size=nvars)

        for t in range(1, nvars):
            X[t] = (1 - b) * X[t - 1] + X[t - 1]**q / (1 + X[t - 1]**q) + \
                decisions[t - 1] + natural_inflows[t - 1]
            average_daily_P[t] += X[t] / float(nsamples)

        reliability += np.sum(X < Pcrit) / float(nsamples * nvars)

    max_P = np.max(average_daily_P)
    utility = np.sum(alpha * decisions * np.power(delta, np.arange(nvars)))
    inertia = np.sum(np.absolute(np.diff(decisions)) < 0.02) / float(nvars - 1)

    return max_P, utility, inertia, reliability


if __name__ == '__main__':
    ema_logging.log_to_stderr(ema_logging.INFO)

    # instantiate the model
    lake_model = Model('lakeproblem', function=lake_problem)
    lake_model.time_horizon = 100

    # specify uncertainties
    lake_model.uncertainties = [RealParameter('b', 0.1, 0.45),
                                RealParameter('q', 2.0, 4.5),
                                RealParameter('mean', 0.01, 0.05),
                                RealParameter('stdev', 0.001, 0.005),
                                RealParameter('delta', 0.93, 0.99)]

    # set levers, one for each time step
    lake_model.levers = [RealParameter(str(i), 0, 0.1) for i in
                         range(lake_model.time_horizon)]

    # specify outcomes
    lake_model.outcomes = [ScalarOutcome('max_P',),
                           ScalarOutcome('utility'),
                           ScalarOutcome('inertia'),
                           ScalarOutcome('reliability')]

    # override some of the defaults of the model
    lake_model.constants = [Constant('alpha', 0.41),
                            Constant('nsamples', 50)]

    # generate some random policies by sampling over levers
    n_scenarios = 10
    n_policies = 10

    with MultiprocessingEvaluator(lake_model) as evaluator:
        res = evaluator.perform_experiments(n_scenarios, n_policies,
                                            levers_sampling=MC)

    experiments, outcomes = res
    print(experiments)
    mydata = [experiments['b'], experiments['q'], experiments['delta']]
    mydata1 = np.array(mydata).T
    print(mydata1)
    df = pd.DataFrame(mydata1, columns=['b', 'q', 'delta'])
    print(df)
    # print(experiments.shape)
    # print(list(outcomes.keys()))
    print(outcomes)
    response = outcomes['utility']

    import rpy2.robjects as robjects

    from rpy2.robjects.packages import importr

    # import R's "base" package
    base = importr('base')

    # import R's "utils" package
    utils = importr('utils')

    packnames = ('ggplot2', 'prim', 'party', 'sandwich')

    import rpy2.robjects.packages as rpackages

    # R vector of strings
    from rpy2.robjects.vectors import StrVector

    # Selectively install what needs to be install.
    # We are fancy, just because we can.
    names_to_install = [x for x in packnames if not rpackages.isinstalled(x)]
    if len(names_to_install) > 0:
        utils.install_packages(StrVector(names_to_install))

    primPackage = importr('prim')

    r = robjects.r
    r('''
            f <- function() {
                data(quasiflow)
                qf <- quasiflow[1:1000,1:3]
                qf.label <- quasiflow[1:1000,4]
                thr <- c(0.25, -0.3)
                qf.prim <- prim.box(x=qf, y=qf.label, threshold=thr, threshold.type=0)
                jpeg('rplot.jpg')
                plot(qf.prim)
                dev.off()
            }
            ''')

    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects import pandas2ri

    with localconverter(robjects.default_converter + pandas2ri.converter):
        qf = robjects.conversion.py2rpy(df)

    print(qf)
    resp = robjects.FloatVector(response)
    #h = r.hist(resp)
    #hf = robjects.conversion.converter.rpy2py(h)
    #pf = pd.DataFrame(hf)
    #print(h)
    #print(pf)
    #print(pf.attrs)
    inputData = pd.read_csv('data.csv')
    print(inputData)

    with localconverter(robjects.default_converter + pandas2ri.converter):
        id = robjects.conversion.py2rpy(inputData)
    print(id)

    r('''
            testdt <- function(x) {
                library(party)
                png(file = "decision_tree.png")
                output = ctree(nativeSpeaker ~ age + shoeSize + score, data = x)
                plot(output)
                dev.off()
                output
            }
    ''')

    myF = r['testdt']
    out = myF(id)
    print(out)
    out

    rsort = robjects.r['sort']
    res = rsort(robjects.IntVector([1, 2, 3]), decreasing=True)
    print(res.r_repr())
    for x in res:
        print(x)

    #x = robjects.IntVector(range(10))
    #y = r.rnorm(10)
    #r.X11()
    #r.layout(r.matrix(robjects.IntVector([1, 2, 3, 2]), nrow=2, ncol=2))
    # r.plot(r.runif(10), y, xlab="runif", ylab="foo/bar", col="red")

    res = r.sort(robjects.IntVector([3, 5, 2]), decreasing=True)
    print(res.r_repr())

    v = robjects.FloatVector([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
    m = r['matrix'](v, nrow=2)
    print(m)

    data = pd.read_csv('data1.csv', index_col=False)
    x1 = data.iloc[:, 2:11]
    y1 = data.iloc[:, 15].values
    prim_alg = prim.Prim(x1, y1, threshold=0.8, peel_alpha=0.1)
    box1 = prim_alg.find_box()
    box1.show_tradeoff()
    #box1.show_pairs_scatter(i=4)
    print(box1.resample(21))
    box1.inspect(21)
    box1.inspect(21, style='table')

    print(box1.resample(15))
    box1.inspect(15)
    box1.inspect(15, style='table')
    #plt.show()

    # load data
    from ema_workbench import ema_logging, load_results
    fn = r'./1000 flu cases with policies.tar.gz'
    x, outcomes = load_results(fn)

    # we have timeseries so we need scalars
    y = {'deceased population': outcomes['deceased population region 1'][:, -1],
         'max. infected fraction': np.max(outcomes['infected fraction R1'], axis=1)}

    scores = feature_scoring.get_feature_scores_all(x, y)
    print(scores)
    sns.heatmap(scores, annot=True, cmap='viridis')
    plt.show()





