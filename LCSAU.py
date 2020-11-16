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



class TestProblem(Scenario):
    def __init__(self, myData, training_cycles = 1000, ):
        self.possible_actions = ('00', '01', '10', '11')
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
        condition = situationData[0:14]
        situation = BitString(condition)
        self.outcome = situationData[14:16]
        self.index = (self.index + 1) % self.dataSize
        return situation

    def execute(self,action):
        self.remaining_cycles -= 1
        return action == self.outcome


def createBinaryEncoding(dFrame,medianValue,dataSeries):
    for x in dFrame:
        if (x < medianValue):
            dataSeries.append(0)
        else:
            dataSeries.append(1)
    return dataSeries

def generateBitString(dataSeries,x, z):
    if dataSeries[x] == 0:
        z = z + '0'
    else:
        z = z + '1'
    return z


data = pd.read_csv('grainNov13.csv')

experiment = data[['DBODY','PC','DNOSE','LNOSE','THROAT','LNOZ',
                   'FN','TBURN','TRCR','TTR','TAILB2','TLE','TLEWINGLENGTH','LAUNCH']]

outcomes = data[['burnout','RangeBurn','TOF','thrust','Range','Apogee']]

print(experiment)

print(outcomes)


burnout_MID = statistics.median(outcomes['burnout'])
thrust_MID = statistics.median(outcomes['thrust'])

DBODY_MID = statistics.median(data['DBODY'])
PC_MID = statistics.median(data['PC'])
DNOSE_MID = statistics.median(data['DNOSE'])
LNOSE_MID = statistics.median(data['LNOSE'])
THROAT_MID = statistics.median(data['THROAT'])
LNOZ_MID = statistics.median(data['LNOZ'])
FN_MID = statistics.median(data['FN'])
TBURN_MID = statistics.median(data['TBURN'])
TRCR_MID = statistics.median(data['TRCR'])
TTR_MID = statistics.median(data['TTR'])
TAILB2_MID = statistics.median(data['TAILB2'])
TLE_MID = statistics.median(data['TLE'])
TLEWINGLENGTH_MID = statistics.median(data['TLEWINGLENGTH'])
LAUNCH_MID = statistics.median(data['LAUNCH'])


burnoutSeries = []
thrustSeries = []

DBODYSeries = []
PCSeries = []
DNOSESeries = []
LNOSESeries = []
THROATSeries = []
LNOZSeries = []
FNSeries = []
TBURNSeries = []
TRCRSeries = []
TTRSeries = []
TAILB2Series = []
TLESeries = []
TLEWINGLENGTHSeries = []
LAUNCHSeries = []

burnoutSeries = createBinaryEncoding(outcomes['burnout'],burnout_MID, burnoutSeries)
thrustSeries = createBinaryEncoding(outcomes['thrust'],thrust_MID, thrustSeries)
DBODYSeries = createBinaryEncoding(data['DBODY'],DBODY_MID, DBODYSeries)
PCSeries = createBinaryEncoding(data['PC'],PC_MID, PCSeries)
DNOSESeries = createBinaryEncoding(data['DNOSE'],DNOSE_MID, DNOSESeries)
LNOSESeries = createBinaryEncoding(data['LNOSE'],LNOSE_MID, LNOSESeries)
THROATSeries = createBinaryEncoding(data['THROAT'],THROAT_MID, THROATSeries)
LNOZSeries = createBinaryEncoding(data['LNOZ'],LNOZ_MID, LNOZSeries)
FNSeries = createBinaryEncoding(data['FN'],FN_MID, FNSeries)
TBURNSeries = createBinaryEncoding(data['TBURN'],TBURN_MID, TBURNSeries)
TRCRSeries = createBinaryEncoding(data['TRCR'],TRCR_MID, TRCRSeries)
TTRSeries = createBinaryEncoding(data['TTR'],TTR_MID, TTRSeries)
TAILB2Series = createBinaryEncoding(data['TAILB2'],TAILB2_MID, TAILB2Series)
TLESeries = createBinaryEncoding(data['TLE'],TLE_MID, TLESeries)
TLEWINGLENGTHSeries = createBinaryEncoding(data['TLEWINGLENGTH'],TLEWINGLENGTH_MID, TLEWINGLENGTHSeries)
LAUNCHSeries = createBinaryEncoding(data['LAUNCH'],LAUNCH_MID, LAUNCHSeries)


bStr = []
instances = len(DBODYSeries)

for x in range(instances):
    z = ''
    z = generateBitString(DBODYSeries, x, z)
    z = generateBitString(PCSeries, x, z)
    z = generateBitString(DNOSESeries, x, z)
    z = generateBitString(LNOSESeries, x, z)
    z = generateBitString(THROATSeries, x, z)
    z = generateBitString(LNOZSeries, x, z)
    z = generateBitString(FNSeries, x, z)
    z = generateBitString(TBURNSeries, x, z)
    z = generateBitString(TRCRSeries, x, z)
    z = generateBitString(TTRSeries, x, z)
    z = generateBitString(TAILB2Series, x, z)
    z = generateBitString(TLESeries, x, z)
    z = generateBitString(TLEWINGLENGTHSeries, x, z)
    z = generateBitString(LAUNCHSeries, x, z)
    z = generateBitString(burnoutSeries, x, z)
    z = generateBitString(thrustSeries, x, z)
    bStr.append(z)

print(bStr)

print(len(bStr))

scenario = ScenarioObserver(TestProblem(bStr,training_cycles=660000))

algorithm = XCSAlgorithm()
algorithm.exploration_probability = .1

algorithm.ga_threshold = 1
algorithm.crossover_probability = .5
algorithm.do_action_set_subsumption = True
algorithm.do_ga_subsumption = False
algorithm.wildcard_probability = .5
algorithm.deletion_threshold = 1
algorithm.mutation_probability = .002

model = algorithm.new_model(scenario)

model.run(scenario,learn=True)

print(model)


print(len(model))

for rule in model:
    if rule.fitness > .5 and rule.experience >= 10:
        print(rule.condition, '=>', rule.action, ' [%.5f]' % rule.fitness)

'''
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

scenario = ScenarioObserver(TestProblem(bStr,training_cycles=15000))

algorithm = XCSAlgorithm()
algorithm.exploration_probability = .1

algorithm.ga_threshold = 1
algorithm.crossover_probability = .5
algorithm.do_action_set_subsumption = True
algorithm.do_ga_subsumption = False
algorithm.wildcard_probability = .5
algorithm.deletion_threshold = 1
algorithm.mutation_probability = .002

model = algorithm.new_model(scenario)

model.run(scenario,learn=True)

print(model)


print(len(model))

for rule in model:
    if rule.fitness > .5 and rule.experience >= 10:
        print(rule.condition, '=>', rule.action, ' [%.5f]' % rule.fitness)
'''