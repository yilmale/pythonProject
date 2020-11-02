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

import matplotlib.pyplot as plt



data = pd.read_csv('omegaExperiment.csv')
outcomes = pd.read_csv('omegaResults.csv')

data = data[['SCUDB.targetRange', 'SCUDB.targetAltitude', 'SCUDB.MassProperties.initialMass']]
outcomes = outcomes[['burnout', 'impact', 'apogeeAlt', 'apogeeTime']]

print(data)
print(outcomes)

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


