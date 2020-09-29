from __future__ import (absolute_import, print_function, division,
                        unicode_literals)
from omegaPythonTools import pyomega
from omegaPythonTools import omiEditor
#import omegaJupyterNotebookTools as ojnt

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

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text


omiFilePath = 'C:\OMEGA\OMEGA\MODELS\SRBM\SCUD_Variants\scudb\scudb.omi'
omegaDirectory = 'C:\OMEGA\OMEGA'
executablePath = 'C:\OMEGA\OMEGA\omega.exe'
modelsDirectory = 'C:\OMEGA\OMEGA\MODELS'
PLAYER = 'SCUDB'
REPLICATIONS = 3


o = pyomega(omiFilename=omiFilePath, omegaDirectory=omegaDirectory)
o.printOutput = False


def setUpExperiment(**arg):
    inputMap = generateInputMap(model)
    for i in range(0, len(arg)):
        setVariable(inputMap[i],arg[inputMap[i]])

def setVariable(vname, val):
    o.omi.set(vname,val)

def collectOutputs(e,m):
    outputs = {}
    for i in outputMap.keys():
        f = outputMap[i]
        r = f(e,m)
        outputs[i] = r
    return outputs

def getBurnOut(e,m):
    return e.at[2, 'eventTime']

def getImpactTime(e,m):
    return m.at[0, 'simTime']

def getapogeeAlt(e,m):
    return m.at[0, 'Dynamics:apogeeAlt']

def getapogeeTime(e,m):
    return e.at[3, 'eventTime']

outputMap = {'burnout': getBurnOut, 'impact': getImpactTime,
                 'apogeeAlt': getapogeeAlt, 'apogeeTime': getapogeeTime}

def generateInputMap(m):
    inputMap = {}
    count = 0
    for i in m.uncertainties:
        inputMap[count] = i.variable_name[0]
        count = count + 1

    for i in m.levers:
        inputMap[count] = i.variable_name[0]
        count = count + 1
    return(inputMap)


def getFactors(mod):
    factors = []
    for i in mod.uncertainties:
        factors = factors + i.variable_name
    for i in mod.levers:
        factors = factors + i.variable_name
    return factors

def generateAnalysisFrame(mod,out):
    outFrame = {}
    frameSchema = []
    for u in mod.outcomes:
        outFrame[u.variable_name[0]] = out[u.variable_name[0]]
        frameSchema = frameSchema + u.variable_name
    return outFrame,frameSchema


def omegaDriver(**arg):
    o.reset()
    setUpExperiment(**arg)
    o.run()
    events = o.data.events[PLAYER]
    metrics = o.data.endOfRun[PLAYER]
    out = collectOutputs(events, metrics)
    return out


if __name__ == '__main__':

    ema_logging.LOG_FORMAT = '[%(name)s/%(levelname)s/%(processName)s] %(message)s'
    ema_logging.log_to_stderr(ema_logging.INFO)

    model = Model('omegaDriver', function=omegaDriver)  # instantiate the model
    # specify uncertainties

    model.uncertainties = [RealParameter("SCUDB.targetRange", 100000.0, 200000.0),
                           RealParameter("SCUDB.targetAltitude",15000.0,20000.0)]

    model.levers = [RealParameter("SCUDB.MassProperties.initialMass", 5000.0, 6000.0)]

    model.outcomes = [ScalarOutcome('burnout'),
                      ScalarOutcome('impact'),
                      ScalarOutcome('apogeeAlt'),
                      ScalarOutcome('apogeeTime')]

    #model.constants = [Constant('replications', 10)]

    n_contextScenarios = 3
    n_designPolicies = 3

    res = perform_experiments(model, n_contextScenarios, n_designPolicies, levers_sampling=LHS)

    experiments, outcomes = res

    data = experiments[getFactors(model)]

    frame,schema = generateAnalysisFrame(model,outcomes)
    y = pd.DataFrame(frame, columns=schema)

    print(data)
    print(y)

    o.omi.show()


''' 
e = omiEditor(omiFilePath)
e.show()

o = pyomega(omiFilename=omiFilePath, omegaDirectory= omegaDirectory)
o.run()
df = o.data.getData()
df.to_csv('outTest.csv')
df1 = o.data.timeHistory['SCUDB']
thrust = df1[['simTime','Propulsion:deliveredThrust']]

found = False
for ind in thrust.index:
    if (thrust['Propulsion:deliveredThrust'][ind] > 0.0):
        found = True
    elif (thrust['Propulsion:deliveredThrust'][ind] == 0) and (found == True):
       print('Burnout: ', thrust['simTime'][ind], thrust['Propulsion:deliveredThrust'][ind])
       break

print(thrust[['Propulsion:deliveredThrust']])

#x= o.omi.get('SCUDB.targetRange')
#print(x)
o.omi.set('SCUDB.targetRange', 100000.0)
#print('updated: ', o.omi.get('SCUDB.targetRange'))

#print(o.omi)

o.run()

print(o.data.endOfRun['SCUDB'])
metrics = o.data.endOfRun['SCUDB']
print(metrics.columns)
print('*******************************************************')
print(o.data.events['SCUDB'])

x=o.data.events['SCUDB']
print(x['eventTime'])
print(x.columns)


o.omi.set('SCUDB.targetRange', 100000.0)
o.run()
events = o.data.events['SCUDB']
print(events)
print(events['eventTime'])

print(events.columns)
print('***************************************')
print(o.data.endOfRun['SCUDB'])
metrics = o.data.endOfRun['SCUDB']
print(metrics.columns)

print('impact time is ', metrics.at[0,'simTime'])
print('apogee alt is ', metrics.at[0,'Dynamics:apogeeAlt'])
print('burnout is ', events.at[2,'eventTime'])
print('apogee time is',events.at[3,'eventTime'])

print(o.omi.help('SCUDB.targetInputOption'))
print(o.omi.help('SCUDB.windOption'))
print(o.omi.help('SCUDB.MassProperties.initialMass'))
print(o.omi.help('SCUDB.targetRange'))

'''
