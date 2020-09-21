from omegaPythonTools import pyomega
from xml.dom.minidom import parse
import xml.dom.minidom

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

inputSpec = o.omi
print(inputSpec)

#DOMTree = xml.dom.minidom.parse('.\scudb.omi')
#collection = DOMTree.documentElement

#variables = collection.getElementsByTagName("variable")


#for elem in variables:
#    print(elem.toxml())
#    print(elem.attributes['name'].value)
#    for c in elem.childNodes:
#        print(c.data)

#newsource = DOMTree.toprettyxml()
print('--------------------------------------------------------------------')
#print(newsource)


#x= o.omi.get('SCUDB.targetRange')
#print(x)
o.omi.set('SCUDB.targetRange', 100000.0)
#print('updated: ', o.omi.get('SCUDB.targetRange'))

#print(o.omi)

o.run()

