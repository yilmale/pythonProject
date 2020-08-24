import pyNetLogo

netlogo = pyNetLogo.NetLogoLink(gui=False,netlogo_home='/Users/yilmaz/Desktop/NetLogo-6.1.1') #Show NetLogo GUI
netlogo.load_model('/Users/yilmaz/Desktop/swarms/Wolf Sheep Predation.nlogo')

netlogo.command('setup')

netlogo.repeat_command('go', 50)
x = netlogo.report('map [s -> [xcor] of s] sort sheep')
y = netlogo.report('map [s -> [ycor] of s] sort sheep')
p = netlogo.report('map [s -> [energy] of s] sort sheep')
energy_wolves = netlogo.report('map [w -> [energy] of w] sort wolves')
sheepCount = netlogo.report('count sheep')
wolvesCount = netlogo.report('count wolves')
print(sheepCount,wolvesCount)

netlogo.kill_workspace()