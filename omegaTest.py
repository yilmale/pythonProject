from __future__ import (absolute_import, print_function, division,
                        unicode_literals)
from omegaPythonTools import pyomega
from omegaPythonTools import omiEditor
import omegaJupyterNotebookTools as ojnt

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

from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text




omiFilePath = 'C:\OMEGA\OMEGA\MODELS\SRBM\SCUD_Variants\scudb\scudb.omi'
omegaDirectory = 'C:\OMEGA\OMEGA'
executablePath = 'C:\OMEGA\OMEGA\omega.exe'
modelsDirectory = 'C:\OMEGA\OMEGA\MODELS'


o = pyomega(omiFilename=omiFilePath, omegaDirectory=omegaDirectory)


def omegaDriver(r, m, alt):
        o.reset()
        o.omi.set('SCUDB.targetRange', r)
        o.omi.set('SCUDB.MassProperties.initialMass', m)
        o.omi.set('SCUDB.targetAltitude', alt)
        o.run()
        events = o.data.events['SCUDB']
        metrics = o.data.endOfRun['SCUDB']
        bOut = events.at[2,'eventTime']
        impactTime = metrics.at[0, 'simTime']
        apogeeAlt = metrics.at[0, 'Dynamics:apogeeAlt']
        apogeeTime = events.at[3, 'eventTime']
        return {'burnout': bOut,'impact': impactTime, 'apogeeAlt': apogeeAlt, 'apogeeTime': apogeeTime}


if __name__ == '__main__':

    ema_logging.LOG_FORMAT = '[%(name)s/%(levelname)s/%(processName)s] %(message)s'
    ema_logging.log_to_stderr(ema_logging.INFO)

    model = Model('omegaDriver', function=omegaDriver)  # instantiate the model
    # specify uncertainties

    model.uncertainties = [RealParameter("r", 100000.0, 200000.0),
                           RealParameter("alt",15000.0,20000.0)]

    model.levers = [RealParameter("m", 5000.0, 6000.0)]

    # specify outcomes
    model.outcomes = [ScalarOutcome('burnout'),
                      ScalarOutcome('impact'),
                      ScalarOutcome('apogeeAlt'),
                      ScalarOutcome('apogeeTime')]

    #model.constants = [Constant('replications', 10)]

    n_scenarios = 3
    n_policies = 3

    res = perform_experiments(model, n_scenarios, n_policies)

    experiments, outcomes = res
    data = experiments[['r', 'm', 'alt']]

    data.to_csv('outOmega.csv', index=False)

    experiments, outcomes = res
    data = experiments[['r', 'm', 'alt']]
    data.to_csv('outExperiment.csv', index=False)
    frame ={'burnout': outcomes['burnout'], 'impact': outcomes['impact'],
            'apogeeAlt': outcomes['apogeeAlt'], 'apogeeTime': outcomes['apogeeTime']}
    y = pd.DataFrame(frame,columns=['burnout','impact','apogeeAlt','apogeeTime'])
    y.to_csv('outResults.csv')
    print(data)
    print(res)





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
