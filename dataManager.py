import math

import numpy as np
import pandas as pd
import seaborn as sns
import csv
from ema_workbench.analysis import feature_scoring
from SALib.sample import saltelli
from SALib.analyze import sobol
from SALib.test_functions import Ishigami


def filter(file):
    with open(file, "r+") as f:
        # First read the file line by line
        lines = f.readlines()

        # Go back at the start of the file
        f.seek(0)
        # Filter out and rewrite lines
        for line in lines:
            if not line.__contains__(',0.000000'):
                f.write(line)

def tab_to_DF(files):
    frames= []
    for f in files:
        cdf =pd.read_fwf(f)
        fname = f+".csv"
        cdf.to_csv(fname)
        df = pd.read_csv(fname)
        df1 = df.iloc[:,1:28]
        frames.append(df1)
    result = pd.concat(frames)
    return result

def csv_to_DF(files):
    frames = []
    for f in files:
        df = pd.read_csv(f)
        frames.append(df)
    result = pd.concat(frames)
    return result



inp = csv_to_DF(['part-00000','part-00001'])
print(inp)




filter('result-00000')
filter('result-00001')

out = csv_to_DF(['result-00000','result-00001'])
print("printing out.......")
print(out)

means = []
stdevs = []
for i in range(0,100):
    df1 = out[(out['MISSNUM'] == i+1)]
    v = float(df1[['THRUST']].mean())
    means.append(v)

print("printing mean thrust.......")

thrust = pd.DataFrame(means,columns=['ThrustMean'])
print(thrust)
print("printing mean thrust matrix")
thrustMat = np.array(means)
print(thrustMat)
print(thrustMat.shape)


data = inp[['Kfuel','EQ_RATIO','PC']]
print(data)
print("inputMatrix.....")
dataMat = data.to_numpy()
print(dataMat)
print(dataMat.shape)


z = feature_scoring.get_feature_scores_all(x=data,y=thrust)
print(z)


problem = {
    'num_vars': 3,
    'names': ['x1', 'x2', 'x3'],
    'bounds': [[-3.14159265359, 3.14159265359],
               [-3.14159265359, 3.14159265359],
               [-3.14159265359, 3.14159265359]]
}

param_values = saltelli.sample(problem, 1000)
print(param_values.shape)
Y = np.zeros([param_values.shape[0]])

print(param_values)

Y = Ishigami.evaluate(param_values)
print(Y.shape)
print(Y)

Si = sobol.analyze(problem,Y)

print(Si['S1'])

print("x1-x2:", Si['S2'][0,1])
print("x1-x3:", Si['S2'][0,2])
print("x2-x3:", Si['S2'][1,2])








