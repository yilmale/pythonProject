from __future__ import (absolute_import, print_function, division,
                        unicode_literals)


from ema_workbench import (Model, RealParameter, MultiprocessingEvaluator, CategoricalParameter,
                           IntegerParameter, ScalarOutcome, ArrayOutcome, Constant, ema_logging,
                           perform_experiments)
from ema_workbench.em_framework.evaluators import (MC,LHS,SOBOL)

from ema_workbench.analysis import feature_scoring
from ema_workbench.analysis import prim

from ema_workbench.analysis import scenario_discovery_util as sdutil

import pandas as pd
import seaborn as sns
import numpy as np

import matplotlib.pyplot as plt

def relativeRegret(data,outcomes,designParams,contextParams,outcomeParams):
    myres = {}
    design = []
    scenario = []
    designSet = set()
    contextSet = set()

    for ind in data.index:
        for i in designParams:
            design.append(data[i][ind])
        for j in contextParams:
            scenario.append(data[j][ind])
        elm = {(tuple(design), tuple(scenario)): outcomes[outcomeParams[0]][ind]}
        myres.update(elm)
        designSet.add(tuple(design))
        contextSet.add(tuple(scenario))
        design = []
        scenario = []

    measures = {}
    for sc in contextSet:
        m = []
        for d in designSet:
            o = myres[(d, sc)]
            m.append(o)
            measures[sc] = m

    maxs = {}
    for sc in contextSet:
        maxs[sc] = max(measures[sc])

    regret = {}
    for sc in contextSet:
        for d in designSet:
            rm = {(d, sc): (maxs[sc] - myres[(d, sc)]) / maxs[sc]}
            regret.update(rm)

    return (regret,contextSet,designSet)


def minmax(r,dsg,ctx):
    worstRegret = {}
    for d in dsg:
        rdata = []
        for sc in ctx:
            rdata.append(r[d, sc])
        elm = {d: rdata}
        worstRegret.update(elm)

    print(worstRegret)
    maxs = []
    maxvs = []
    for d, l in worstRegret.items():
        maxs.append(max(l))
        maxvs.append((d, max(l)))

    print(maxvs)
    minmax = min(maxs)
    minmaxKey=0
    for x in maxvs:
        if (x[1] == minmax):
            minmaxKey = x[0]

    return minmaxKey,minmax

'''
data = pd.read_csv('grainNov13.csv')

experiment = data[['DBODY','PC','DNOSE','LNOSE','THROAT','LNOZ',
                   'FN','TBURN','TRCR','TTR','TAILB2','TLE','TLEWINGLENGTH','LAUNCH']]

outcomes = data[['burnout','RangeBurn','TOF','thrust','Range','Apogee']]

print(experiment)

z = feature_scoring.get_feature_scores_all(x=experiment, y=outcomes)

print(z)

sns.heatmap(z, cmap='viridis', annot=True)
plt.show()
 '''


data = pd.read_csv('omegaExperiment.csv')
outcomes = pd.read_csv('omegaResults.csv')

data = data[['SCUDB.targetRange', 'SCUDB.targetAltitude', 'SCUDB.MassProperties.initialMass']]
outcomes = outcomes[['burnout', 'impact', 'apogeeAlt', 'apogeeTime']]

print(data)
print(outcomes)

designParams = ['SCUDB.MassProperties.initialMass']
contextParams = ['SCUDB.targetRange', 'SCUDB.targetAltitude']
outcomeParams = ['burnout']

r,ctx,dsg = relativeRegret(data,outcomes,designParams,contextParams,outcomeParams)
print(r)
print(ctx)
print(dsg)

a,b = minmax(r,dsg,ctx)
print(a,b)

'''
regretData = []
sns.set()
mat = np.random.rand(len(dsg),len(ctx))

i = 0
j = 0
for d in dsg:
    j=0
    for sc in ctx:
        mat[i,j]= r[d,sc]
        regretData.append(mat[i,j])
        j=j+1
    i=i+1

sns.set()
vM = max(regretData)
vm = min(regretData)
ax = sns.heatmap(mat, vmin=vm, vmax=vM)
plt.show()
'''



'''

z = feature_scoring.get_feature_scores_all(x=data, y=outcomes)

print(z)
print('initialMass -- burnout: ')
print(z.at['SCUDB.MassProperties.initialMass', 'burnout'])
print('initialMass -- impact: ')
print(z.at['SCUDB.MassProperties.initialMass', 'impact'])
print('initialMass -- apogeeAlt: ')
print(z.at['SCUDB.MassProperties.initialMass', 'apogeeAlt'])
print('initialMass -- apogeeAlt: ')
print(z.at['SCUDB.MassProperties.initialMass', 'apogeeAlt'])

print('targetAltitude -- burnout: ')
print(z.at['SCUDB.targetAltitude', 'burnout'])
print('targetAltitude -- impact: ')
print(z.at['SCUDB.targetAltitude', 'impact'])
print('targetAltitude -- apogeeAlt: ')
print(z.at['SCUDB.targetAltitude', 'apogeeAlt'])
print('targetAltitude -- apogeeAlt: ')
print(z.at['SCUDB.targetAltitude', 'apogeeAlt'])

print('targetRange -- burnout: ')
print(z.at['SCUDB.targetRange', 'burnout'])
print('targetRange -- impact: ')
print(z.at['SCUDB.targetRange', 'impact'])
print('targetRange -- apogeeAlt: ')
print(z.at['SCUDB.targetRange', 'apogeeAlt'])
print('targetRange -- apogeeAlt: ')
print(z.at['SCUDB.targetRange', 'apogeeAlt'])

sns.heatmap(z, cmap='viridis', annot=True)
plt.show()

'''
