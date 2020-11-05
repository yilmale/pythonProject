import logging
logging.root.setLevel(logging.INFO)

from xcs import XCSAlgorithm
from xcs.scenarios import MUXProblem, ScenarioObserver

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
