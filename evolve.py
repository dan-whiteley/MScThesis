from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)



# for lines of best fit
def correl(x,y):
    df = len(x)-2
    SP = np.sum(x*y)-(np.sum(x)*np.sum(y))/len(x)
    SSx = np.sum(x**2)-(np.sum(x)**2)/len(x)
    SSy = np.sum(y**2)-(np.sum(y)**2)/len(y)
    m = SP/SSx
    c = np.mean(y)-m*np.mean(x)
    r,p=np.corrcoef(x,y)
    return m,c,r,p

# define the fitness function
'''
THE FITNESS FUNTION FOR SMOOTH.PNG

def fitness(net):
	f = net.count('1')/80

	#because the number of networks with all ones is vanishingly small we relax the criteria for optimal fitness
	if f > 0.7:
		f = 1

	return f



THE FITNESS FUNCTION FOR ROUGH.PNG

def fitness(net):
	f = np.random.rand()
	
	if f > 0.999:
		f = 1

	return f
'''

#f = 1 - 0.1*d :

def fitness(net):
	#find where each state, 16 and 0, end up, if stable: count mutations from desired 21 and 10.
	anterior = '10000'
	effect = anterior
	output = 'start'
	i=0

	while output!=effect:
		output = effect
		effect = net[int(output[1:],2)] + net[int(output[:1]+output[2:],2)+16] + net[int(output[:2]+output[3:],2) + 32] + net[int(output[:3]+output[4],2)+48] + net[int(output[:4],2)+64]	
		i = i+1 # in case of limit cycles
		if i == 33:
			effect = 'cycle'
			break

	if effect!='cycle':
		mut = bin(int(effect,2)^21)[2:]
		d_ant = mut.count('1')

		posterior = '00000'
		effect = posterior
		output = 'start'
		i=0

		while output!=effect:
			output = effect
			effect = net[int(output[1:],2)] + net[int(output[:1]+output[2:],2)+16] + net[int(output[:2]+output[3:],2) + 32] + net[int(output[:3]+output[4],2)+48] + net[int(output[:4],2)+64]	
			i = i+1 # in case of limit cycles
			if i == 33:
				effect = 'cycle'
				break

		if effect!='cycle':
			mut = bin(int(effect,2)^10)[2:]
			d_pos = mut.count('1')

			d = d_ant+d_pos
			f = 1 - 0.1*d 

		else:
			f = 0
	else:
		f = 0

	return f




gens = 20000000

pvalues = [0.1,0.2,0.3,0.4,0.5,0.05,0.15,0.25,0.35,0.45]
P = len(pvalues)

results = np.zeros((gens,P))


for t in range(P):
	p = pvalues[t]
	n = 0
	while n < gens:
		
		#80 bits is too long for randint...
		pt1 = np.random.randint(0,2**40)
		pt1 = np.binary_repr(pt1,40)
		pt2 = np.random.randint(0,2**40)
		pt2 = np.binary_repr(pt2,40)
		ref = pt1 + pt2

		#a is fitness of reference
		a = fitness(ref)
		n = n + 1
		results[n,t] = a

		while (a < 1):
			#flip some bits to get new genome
			new = list(ref)
			for z in range(80):
				if np.random.rand() < p:
					#flip it
					new[z] = bin(int(new[z],2)^1)[2:]
			#b is fitness of new
			n = n + 1
			if n == gens:
				break
			b = fitness(new)
			if a > b:
				results[n,t] = a
			else:
				results[n,t] = b
				a = b
				ref = "".join(new)



#get an array of length of trials

length = [[] for p in pvalues]

for t in range(P):
	length[t] = np.nonzero(results[:,t] == 1)[0]
	length[t][1:] -= length[t][:-1].copy()
	print t
	print len(length[t])/gens 

'''
0 (p=0.1)
0.00043555
1
0.0001868
2
7.84e-05
3
3.525e-05
4
1.39e-05
5
0.0005476
6
0.0002879
7
0.0001229
8
5.305e-05
9
2.295e-05
'''


#now to plot the frequency distribution...
mvalues = []
colours = ['r','b','g','m','c']

plt.figure()

for t in range(P):
	counts,bin_edges = np.histogram(length[t],bins=20)
	bin_centres = (bin_edges[:-1] + bin_edges[1:])/2.
	if t<len(colours):
		plt.plot(bin_centres,np.ma.log(counts),colours[t]+'o',label=("p = "+str(pvalues[t])))

	# fits
	m,c,r,p = correl(bin_centres,np.ma.log(counts))
	mvalues.append(m)
	if t<len(colours):
		plt.plot(bin_centres,m*bin_centres + c, colours[t]+'--')
	
plt.legend()
plt.xlabel('generations til max')
plt.ylabel('log frequency')
plt.savefig("f1.png")

#also plot p vs slope
plt.figure()

plt.plot(pvalues,mvalues, 'o')
plt.xlabel('p')
plt.ylabel('slope')
plt.savefig("f1slopes.png")
