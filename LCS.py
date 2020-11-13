import logging
logging.root.setLevel(logging.INFO)

from xcs import XCSAlgorithm
from xcs.scenarios import MUXProblem, ScenarioObserver
import pandas as pd
import statistics

''' 
scenario = ScenarioObserver(MUXProblem(50000))

algorithm = XCSAlgorithm()

model = algorithm.new_model(scenario)

model.run(scenario,learn=True)

print(model)

print(len(model))

for rule in model:
    if rule.fitness > .5 and rule.experience >= 10:
        print(rule.condition, '=>', rule.action, ' [%.5f]' % rule.fitness)

'''

from xcs.scenarios import Scenario
from xcs.bitstrings import BitString
import random
import xcs

class HayStackProblem(Scenario):
    def __init__(self, training_cycles = 1000, input_size=25):
        self.input_size = input_size
        self.possible_actions = (True, False)
        self.initial_training_cycles = training_cycles
        self.remaining_cycles = training_cycles
        self.needle_index = random.randrange(input_size)
        self.needle_value = None


    @property

    def is_dynamic(self):
        return False

    def get_possible_actions(self):
        return self.possible_actions

    def reset(self):
        self.remaining_cycles = self.initial_training_cycles
        self.needle_index = random.randrange(self.input_size)

    def more(self):
        return self.remaining_cycles > 0

    def sense(self):
        haystack = BitString.random(self.input_size)
        self.needle_value = haystack[self.needle_index]
        return haystack

    def execute(self,action):
        self.remaining_cycles -= 1
        return action == self.needle_value


class TestProblem(Scenario):
    def __init__(self, myData, training_cycles = 1000, ):
        self.possible_actions = ('1', '0')
        self.initial_training_cycles = training_cycles
        self.remaining_cycles = training_cycles
        self.testDat = myData
        self.index = 0
        self.dataSize = len(myData)
        self.outcome = None


    @property

    def is_dynamic(self):
        return False

    def get_possible_actions(self):
        return self.possible_actions

    def reset(self):
        self.remaining_cycles = self.initial_training_cycles
        self.index = 0

    def more(self):
        return self.remaining_cycles > 0

    def sense(self):
        situationData = self.testDat[self.index]
        condition = situationData[0:3]
        situation = BitString(condition)
        self.outcome = situationData[3]
        self.index = (self.index + 1) % self.dataSize
        return situation

    def execute(self,action):
        self.remaining_cycles -= 1
        return action == self.outcome


''' 
problem = HayStackProblem(training_cycles=10000,input_size=100)
algorithm = xcs.XCSAlgorithm()
algorithm.exploration_probability = .1

algorithm.ga_threshold = 1
algorithm.crossover_probability = .5
algorithm.do_action_set_subsumption = True
algorithm.do_ga_subsumption = False
algorithm.wildcard_probability = .998
algorithm.deletion_threshold = 1
algorithm.mutation_probability = .002

#xcs.test(algorithm,scenario = ScenarioObserver(problem))
scenario = ScenarioObserver(problem)
model = algorithm.new_model(scenario)

model.run(scenario,learn=True)

print(model)
'''

RANGE_MID = 160000.00
ALT_MID = 17500.00
MASS_MID = 5500.00

data = pd.read_csv('omegaExperiment.csv')
outcomes = pd.read_csv('omegaResults.csv')

data = data[['SCUDB.targetRange', 'SCUDB.targetAltitude', 'SCUDB.MassProperties.initialMass']]
outcomes = outcomes[['burnout', 'impact', 'apogeeAlt', 'apogeeTime']]

print(data)
print(outcomes)


burnout_MID = statistics.median(outcomes['burnout'])
impact_MID = statistics.median(outcomes['impact'])
apogeeAlt_MID = statistics.median(outcomes['apogeeAlt'])
apogeeTime_MID = statistics.median(outcomes['apogeeTime'])
print('Apogee Time Mid level')
print(apogeeTime_MID)


print(data['SCUDB.targetRange'])
rangeSeries = []
altSeries = []
massSeries = []
apogeeAltSeries = []
apogeeTimeSeries = []

for x in data['SCUDB.targetRange']:
    if (x < RANGE_MID):
        rangeSeries.append(0)
    else:
        rangeSeries.append(1)
print('RangeSeries')
print(rangeSeries)

for x in data['SCUDB.targetAltitude']:
    if (x < ALT_MID):
        altSeries.append(0)
    else:
        altSeries.append(1)

print('altSeries')
print(altSeries)

for x in data['SCUDB.MassProperties.initialMass']:
    if (x < MASS_MID):
        massSeries.append(0)
    else:
        massSeries.append(1)

print('massSeries')
print(massSeries)

for x in outcomes['apogeeTime']:
    if (x < apogeeTime_MID):
        apogeeTimeSeries.append(0)
    else:
        apogeeTimeSeries.append(1)

print('apogeeTimeSeries')
print(apogeeTimeSeries)


bStr = []

for x in range(len(rangeSeries)):
    z = ''
    if rangeSeries[x] == 0:
        z = z + '0'
    else:  z = z + '1'
    if altSeries[x] == 0:
        z = z + '0'
    else:  z = z + '1'
    if massSeries[x] == 0:
        z = z + '0'
    else:  z = z + '1'
    if apogeeTimeSeries[x] == 0:
        z = z + '0'
    else:  z = z + '1'
    bStr.append(z)

print(bStr)

scenario = ScenarioObserver(TestProblem(bStr,training_cycles=1000))

algorithm = XCSAlgorithm()
algorithm.exploration_probability = .1

algorithm.ga_threshold = 1
algorithm.crossover_probability = .5
algorithm.do_action_set_subsumption = True
algorithm.do_ga_subsumption = False
algorithm.wildcard_probability = .7
algorithm.deletion_threshold = 1
algorithm.mutation_probability = .002

model = algorithm.new_model(scenario)

model.run(scenario,learn=True)

print(model)

print(len(model))

for rule in model:
    if rule.fitness > .5 and rule.experience >= 10:
        print(rule.condition, '=>', rule.action, ' [%.5f]' % rule.fitness)