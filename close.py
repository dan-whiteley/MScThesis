'''

Visualise the attractors of boolean networks

x axis gives the states of the attractors, y axis represents the networks

attractors are coloured, black for stable points, red for cycles.

'''
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


data = [[4, [7], [1,2,3]], [1, [7], []],[2, [7,1], []],[3, [7], [1,0,5]]]

fig = plt.figure()																	

#plot
for x in data:
	#plot a black dot for points, red dot for cycles
	for i in x[1]:
		plt.plot(i,x[0],'k.',markersize=8)
	for i in x[2]:
		plt.plot(i,x[0],'r.',markersize=8)
	
plt.axis([-1,8,0,5])
plt.yticks([1,2,3,4],['a','b','c','original'])
plt.xlabel('states in attractors')
plt.ylabel('networks')
plt.savefig('close.png')