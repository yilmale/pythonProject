class FeatureExpression:
    kind = 'Feature Expression'

class F(FeatureExpression):
    kind = "Feature"
    def __init__(self, fname):
        self.name = fname

class CompositeExpression(FeatureExpression):
    kind = "Composite"

class FM(CompositeExpression):
    kind = "FM"
    def __init__(self, fname, fm):
        self.op = fm
        self.name = fname

class Mandatory(CompositeExpression):
  kind = "Mandatory"
  def __init__(self, fl):
      self.op = fl

class Optional(CompositeExpression):
    kind = "Optional"
    def __init__(self, fl):
        self.op = fl

class Alternative(FeatureExpression):
    kind = "Alternative"
    def __init__(self, fl):
        self.op = fl

def evaluate(func,fm,res):
    func(fm,res)

def traverse(fexpr,res):
    if (isinstance(fexpr, F)):
        if fexpr.name in res: print(fexpr.name, " is selected")
    elif (isinstance(fexpr, FM)):
        print(fexpr.kind)
        for f in fexpr.op:
            traverse(f,res)
    elif (isinstance(fexpr, Mandatory)):
        print(fexpr.kind)
        for f in fexpr.op:
            traverse(f,res)
    elif (isinstance(fexpr, Optional)):
        print(fexpr.kind)
        for f in fexpr.op:
            traverse(f,res)





def M1(self):
    print("Hello-M1")
    self.i = 15

def M2(self):
    print("Hello-M2")
    self.i = 20

ParentClass = type('ParentClass',(),{})
MyClass = type('MyClass',(ParentClass,),{
    'M1': M1,
    'i': 10
})

x = MyClass()
print(x.i)
x.M1()
print(x.i)

d = MyClass.__dict__
print(d)
e=d.copy()
e.update({'M2': M2})
print(e)

MyClass= type('MyClass',(ParentClass,),e)

y=MyClass()
y.M2()
print(y.i)

from ema_workbench import (Model, RealParameter, ScalarOutcome, Constant,
                           ema_logging, MultiprocessingEvaluator)
from ema_workbench.em_framework.evaluators import MC
from aumain import lake_problem

ps = {'b': (0.1, 0.45, 0.57),
      'q': (2.0, 4.5, 3.25),
      'mean': (0.01, 0.05, 0.03),
      'stdev': (0.001, 0.005, 0.003),
      'delta': (0.93, 0.99, 0.96),
      'alpha': (0,0,0.41),
      'nsamples': (0,0,50),
      'design': (0,0.1,0.05)
      }

pft = FM('ParameterFeatureTree',[
          Mandatory([FM('context',
                        [Optional([F('b'),F('q'),F('mean'),F('stdev'),F('delta')])])]),
          Mandatory([FM('design',
                        [Mandatory([F(str(i)) for i in range(10)])])]),
          Mandatory([FM('outcome',
                        [Optional([F('max_P'),F('utility'),F('inertia'),F('reliability')])])]),
          Optional([FM('constants',
                       [Optional([F('nsamples'),F('alpha')])])])]
         )

x = traverse
resolution = {'context','b','q','mean','stddev','delta','design'
              'outcomes','max_P','utility','inertia','reliability',
              'constants','nsamples','alpha'}
design = {str(i) for i in range(10)}

resolution = resolution.union(design)
print(resolution)

evaluate(x,pft,resolution)



