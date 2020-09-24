from __future__ import (absolute_import, print_function, division,
                        unicode_literals)
from omegaPythonTools import pyomega
from omegaPythonTools import omiEditor

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

''' 
pyomega(omiFilename=omiFilePath, omegaDirectory=omegaBin,
        outputDirectory=outFilePath)
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

