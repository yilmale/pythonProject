
import math


# more or less default imports when using
# the workbench
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.optimize import brentq

import prim


def get_antropogenic_release(xt, c1, c2, r1, r2, w1):
    '''

    Parameters
    ----------
    xt : float
         polution in lake at time t
    c1 : float
         center rbf 1
    c2 : float
         center rbf 2
    r1 : float
         ratius rbf 1
    r2 : float
         ratius rbf 2
    w1 : float
         weight of rbf 1

    Returns
    -------
    float

    note:: w2 = 1 - w1

    '''

    rule = w1*(abs(xt-c1)/r1)**3+(1-w1)*(abs(xt-c2)/r2)**3
    at1 = max(rule, 0.01)
    at = min(at1, 0.1)

    return at


def lake_model(b=0.42, q=2.0, mean=0.02,
               stdev=0.001, delta=0.98, alpha=0.4,
               nsamples=30, myears=100, c1=0.25,
               c2=0.25, r1=0.5, r2=0.5,
               w1=0.5, seed=None):
    '''runs the lake model for nsamples stochastic realisation using
    specified random seed.

    Parameters
    ----------
    b : float
        decay rate for P in lake (0.42 = irreversible)
    q : float
        recycling exponent
    mean : float
            mean of natural inflows
    stdev : float
            standard deviation of natural inflows
    delta : float
            future utility discount rate
    alpha : float
            utility from pollution
    nsamples : int, optional
    myears : int, optional
    c1 : float
    c2 : float
    r1 : float
    r2 : float
    w1 : float
    seed : int, optional
           seed for the random number generator

    Returns
    -------
    tuple

    '''
    np.random.seed(seed)
    Pcrit = brentq(lambda x: x**q/(1+x**q) - b*x, 0.01, 1.5)

    X = np.zeros((myears,))
    average_daily_P = np.zeros((myears,))
    reliability = 0.0
    inertia = 0
    utility = 0

    for _ in range(nsamples):
        X[0] = 0.0
        decision = 0.1

        decisions = np.zeros(myears,)
        decisions[0] = decision

        natural_inflows = np.random.lognormal(
            math.log(mean**2 / math.sqrt(stdev**2 + mean**2)),
            math.sqrt(math.log(1.0 + stdev**2 / mean**2)),
            size=myears)

        for t in range(1, myears):

            # here we use the decision rule
            decision = get_antropogenic_release(X[t-1], c1, c2, r1, r2, w1)
            decisions[t] = decision

            X[t] = (1-b)*X[t-1] + X[t-1]**q/(1+X[t-1]**q) + decision +\
                natural_inflows[t-1]
            average_daily_P[t] += X[t]/nsamples

        reliability += np.sum(X < Pcrit)/(nsamples*myears)
        inertia += np.sum(np.absolute(np.diff(decisions)
                                      < 0.02)) / (nsamples*myears)
        utility += np.sum(alpha*decisions*np.power(delta,
                                                   np.arange(myears))) / nsamples
    max_P = np.max(average_daily_P)
    return max_P, utility, inertia, reliability



from ema_workbench import (RealParameter, ScalarOutcome, Constant,
                           Model)

model = Model('lakeproblem', function=lake_model)


#specify uncertainties
model.uncertainties = [RealParameter('b', 0.1, 0.45),
                       RealParameter('q', 2.0, 4.5),
                       RealParameter('mean', 0.01, 0.05),
                       RealParameter('stdev', 0.001, 0.005),
                       RealParameter('delta', 0.93, 0.99)]

# set levers
model.levers = [RealParameter("c1", -2, 2),
                RealParameter("c2", -2, 2),
                RealParameter("r1", 0, 2),
                RealParameter("r2", 0, 2),
                RealParameter("w1", 0, 1)]

#specify outcomes
model.outcomes = [ScalarOutcome('max_P'),
                  ScalarOutcome('utility'),
                  ScalarOutcome('inertia'),
                  ScalarOutcome('reliability')]

# override some of the defaults of the model
model.constants = [Constant('alpha', 0.41),
                   Constant('nsamples', 30),
                   Constant('myears', 100)]



from ema_workbench import (MultiprocessingEvaluator, IpyparallelEvaluator, ema_logging,
                           perform_experiments, optimize)
ema_logging.log_to_stderr(ema_logging.INFO)

#with MultiprocessingEvaluator(model, n_processes=2, maxtasksperchild=4) as evaluator:
# results = evaluator.perform_experiments(scenarios=10, policies=10)

#import ipyparallel as ipp
#rc = ipp.Client()
#dview = rc[:]

#with IpyparallelEvaluator(model,rc) as evaluator:
# results = evaluator.perform_experiments(model, scenarios=10, policies=10)

results = perform_experiments(model, scenarios=10, policies=10)


experiments, outcomes = results
#print(experiments)
mydata = [experiments['b'],experiments['q'],experiments['delta']]
mydata1 = np.array(mydata).T
#print(mydata1)
df = pd.DataFrame(mydata1, columns=['b', 'q', 'delta'])
#print(df)
#print(experiments.shape)
#print(list(outcomes.keys()))
#print(outcomes)
response = outcomes['utility']

#p = prim.Prim(df, response, threshold=1.0, threshold_type=">")
#box = p.find_box()
#box.show_tradeoff()
#plt.show()

#print(response)



#from xcs import XCSAlgorithm
#from xcs.scenarios import MUXProblem, ScenarioObserver

#scenario = ScenarioObserver(MUXProblem(50000))

#algorithm = XCSAlgorithm()

import rpy2
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
h=r.hist(resp)
hf = robjects.conversion.converter.rpy2py(h)
pf = pd.DataFrame(hf)
print(h)
print(pf)
print(pf.attrs)

#print(prim_response)
#thr = robjects.FloatVector([1.0,2.0])
#rprim = r['prim.box']
#prim_res = rprim(x=qf,y=prim_response,threshold=thr)
#print(prim_res)


p = prim.Prim(df, response, threshold=1.0, threshold_type=">")
#box = p.find_box()
#box.show_tradeoff()
#plt.show()


from sklearn.datasets import load_iris
from sklearn import tree
X, y = load_iris(return_X_y=True)
print(X)
print(y)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, y)
tree.plot_tree(clf)


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
res = rsort(robjects.IntVector([1,2,3]), decreasing=True)
print(res.r_repr())
for x in res:
    print(x)


x = robjects.IntVector(range(10))
y = r.rnorm(10)
r.X11()
r.layout(r.matrix(robjects.IntVector([1, 2, 3, 2]), nrow=2, ncol=2))
#r.plot(r.runif(10), y, xlab="runif", ylab="foo/bar", col="red")

res = r.sort(robjects.IntVector([3,5,2]), decreasing=True)
print(res.r_repr())

v = robjects.FloatVector([1.1, 2.2, 3.3, 4.4, 5.5, 6.6])
m = r['matrix'](v, nrow = 2)
print(m)









