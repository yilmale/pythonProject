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

import random

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

def randomize(spec):
    spec.set('SCUDB.Propulsion.mainPurgeThrust', random.uniform(10.0,20.0))

def omegaDriver(**arg):
    outFinal = {}
    for i in outputMap.keys():
        outFinal[i] = []
    o.reset()
    setUpExperiment(**arg)
    for r in range(0,REPLICATIONS):
        randomize(o.omi)
        o.run()
        events = o.data.events[PLAYER]
        metrics = o.data.endOfRun[PLAYER]
        out = collectOutputs(events, metrics)
        for i in outputMap.keys():
            outFinal[i].append(out[i])
    return outFinal


if __name__ == '__main__':

    ema_logging.LOG_FORMAT = '[%(name)s/%(levelname)s/%(processName)s] %(message)s'
    ema_logging.log_to_stderr(ema_logging.INFO)

    model = Model('omegaDriver', function=omegaDriver)  # instantiate the model
    # specify uncertainties

    model.uncertainties = [RealParameter("SCUDB.targetRange", 100000.0, 200000.0),
                           RealParameter("SCUDB.targetAltitude",15000.0,20000.0)]

    model.levers = [RealParameter("SCUDB.MassProperties.initialMass", 5000.0, 6000.0)]

    model.outcomes = [ArrayOutcome('burnout'),
                      ArrayOutcome('impact'),
                      ArrayOutcome('apogeeAlt'),
                      ArrayOutcome('apogeeTime')]

    #model.constants = [Constant('replications', 10)]

    n_contextScenarios = 3
    n_designPolicies = 3

    res = perform_experiments(model, n_contextScenarios, n_designPolicies, levers_sampling=LHS)

    experiments, outcomes = res

    #data = experiments[getFactors(model)]

    #frame, schema = generateAnalysisFrame(model, outcomes)
    #y = pd.DataFrame(frame, columns=schema)

    print(experiments)
    print(res)

