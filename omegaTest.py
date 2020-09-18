from omegaPythonTools import pyomega

omiFilePath = 'C:\OMEGA\OMEGA\MODELS\SRBM\SCUD_Variants\scudb\scudb.omi'
outFilePath = 'C:\OMEGA\OMEGA\Temp'
omegaBin = 'C:\OMEGA\OMEGA'
executablePath = 'C:\OMEGA\OMEGA\omega.exe'

''' 
pyomega(omiFilename=omiFilePath, omegaDirectory=omegaBin,
        outputDirectory=outFilePath)
'''

o = pyomega(omiFilename=omiFilePath, exe=executablePath)
o.run()
df = o.data.getData()

print(df)