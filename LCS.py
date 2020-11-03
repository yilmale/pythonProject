import logging
logging.root.setLevel(logging.INFO)

from xcs import XCSAlgorithm
from xcs.scenarios import MUXProblem, ScenarioObserver


scenario = ScenarioObserver(MUXProblem(50000))

algorithm = XCSAlgorithm()

model = algorithm.new_model(scenario)

model.run(scenario,learn=True)

print(model)

print(len(model))

for rule in model:
    if rule.fitness > .5 and rule.experience >= 10:
        print(rule.condition, '=>', rule.action, ' [%.5f]' % rule.fitness)


from xcs.scenarios import Scenario

class HayStackProblem(Scenario):
    def __init__(self, training_cycles = 1000, input_size=500):
        self.input_size = input_size
        self.possible_actions = (True, False)
        self.initial_training_cycles = training_cycles
        self.remaining_cycles = training_cycles

    @property

    def is_dynamic(self):
        return False

    def get_possible_actions(self):
        return self.possible_actions

    def reset(self):
        self.remaining_cycles = self.initial_training_cycles

    def more(self):
        return self.remaining_cycles > 0