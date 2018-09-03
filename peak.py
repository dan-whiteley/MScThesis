from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib import rc

rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)


#sampling from n steps away

nets = ['00011100100000111110111000110010001011010110100111111000100001100111000010111001','01111110110111111011010001111000001111011111011110011010111000000101001100100111','01011100100111011110110011000011010011010111011011101100101010000010101111101101','10010101000111101010100100110010010101001111001100111001001011000011000011110111','10000111010001000111100100110010111011000101110101101010001011110010101011100100','11011100110110110111111000000001110111010100111111111110101000011000001111100010','11111100100011101110000101011011011011000100100101011000011010100001001010100110','00101100010110000010011110010000101000010110000010011110010011111110001011111011','01000110110010000111000000110010010000001111111010101100001011000111100010110001','11000101110011000010110000011011000110011111110011111111010011000001000010110111','00000111000000101111110101001000111001001100110000011011111010100011100101111010','01001111100010001010100010110000001011001101000110001111000000010001001000100100','10010101010011010010110101100000000000000101100110101100110001100001101011110110','01110101010010111111110100011000010011011100011110011110000000000111101000101011','01010100100101011111101101000001011110011100111110101101000011110111000110101111','11010111010100110011110100111010100101011111011100001010000000100000000011110101','01101100110010110110101010100000110010000101010110101110111000100110000111101011','00110110100100001010001101001011001111011110001110101101001011010111101101111111','10110100000011101111100011101010001010010111010100001001101010100100101001100111','10000110010011101011100101100011010001011110011110011100101010000010001000111101','10011101000101011011010001110010011101001111010110001010110010010100001011100000','01010111010101011110111010011011011001001110111000001111001010010011001111100100','00100110100000100110110101011011001001001100101010011010111010001101100101111110','01100101110011101011000000010010010111011100111111101111010000100111001100110111','10110110010101011111000001101011110001001101010110111010101010101000101010110010','10000100010111010011000011000010010100000100000111011000110010000001001010110110','11001100000011001111101011111011010110011100110001111011001000100010000010101011','10110111110101001111111000111010100100000111000101101010011011111001000010100110','10100111010100101111110000110010001101001101000110101110000001001110001110101001','01010101000101101011100000111000000100001101011001011010100010010111000010110101','00110101000001100111001101101001011100001111101111001011001011000010100011111000','01011110010111001010010101000001001100011110000000001101100010010101001001101000','10101101000100011011110000011010100011000111010000101111101010110111001011111101','10011111110100001111101101101000010000011111000110111011001010000110000000110110','00000111000100011110101000010000001101011111100100111111111010100001001100111101','11001110100010010110100100001000000110001110110110111011011001101010000000110011','01111110000111011010010101111011111100001100000001001110001001001111001101110010','11101101010110110111001110111010100011000101111110111101001000100001001100100001','00101110110011001011001010010010010001001101111110011100010000000001001000111101','00100100100100001011110001101000000111001101110110101011101001100001000111111100','01101111010001111011100010100001000011000110001011111001000000110010001010111101','00010111010000111011001111100010001101010100010111111010010000010010100011100000','01100101000000011010100000000011111100010101000001001000111000100101101011101011','10101110000000011110100000100011110100001110001111001000000001011001001100111110','10000101000101100010111101111000110100001101000111111110000010010100101011111100','00001111110011111011000001011001100001000100000110101110100010110110001010110001','10101101000011100111111001110001000101011111101110111001010011111110000001101010','01001110000010110011001110000011011010011100011111101101101010001011101110111111','01110111100111001111111111000000010101001101011111001011010010100101101101100100','01010110000001101010101100010011010011001111010111001110011001000011101111110010','10101100010010001011011101010001100001011101011001001110011011011100001010110111','10110110000011011111000111110001010101011111010011001000001010100000100101111010','01100111110001101110111000000011011101011111000011101011111010111001000111111100','01001111010011010111101110111010000001011101101001111110001011001101000011110101','10100100110010000110000010110010101000010111100011001111001001000001000110111010','00001100100111101010101110110000001011011110111010001100110001000101101111111111','10000111100111001011010101011010110100011111111010011111010011111001101010100101','01111101010100111110011000001010001111010100100110101011000010100000100010111101','11000101100001011010001111001001111101010111011101001110100000100011000001111000','10100111110110101111010001011011011011011101100111001101111010100000101110100110','10001100000111110010000110010010101001001110111010011101100001010100001111100101','00000100110101111110000101000000011101000100110111001010111000110101001101110110','01100100000110111110000100011011001101010111000100101101100011110110001010100111','00100110110101000011000111000010101100000101001110011111101010111001100010101000','01000110000101001010000000010001011101001100011010101100000001110110001100100100','10110101110110000011110100010010100111000110100111001101001001110000000010111100','10000101110001000011101111110011000100011101010000001011011000001110001010101010','11101101010000001010110101110011001010001100110101111011011011101000100011101001','10111111000001011111001000100000000001010101000110011000000001110111000011101011','00111111100110000110100110101001011101000111111101001011011001011101101001110101','10000111010011010011010100000010010000001110010111001011010011101011001100111110','10001111010000001111011001111010010011001100110111011011010010000101001011101110','01010100100101101111010100110000001110001101010111101111001000000100101110111010','00001111100011011011100011000011011011010110111110101111011001100100001110111111','11011100100110000011000101001011101100001110111111011011010011110111000010111010','01011101100101101011111000001000010100001101110011011011011001011010000100110100','11001101000000100110010000111010010101011101100010101110010010011000001010111000','01011101100100011111100011010010010111001110001111101011110011110001001011110011','10010100000011111111010101001011000110000101101101001011101011111110101111100000','11100101000001111011100101110010010100001110100111101111010000101111101100100111','10000110000011101011110100101011011101001111010000011110010010010000100011101010','00010110110001111011000010101000010001000111001011001011001010000011000101100101','00011111000111101110001011011001011000011111011100111010100001100010000000101110','00110101010111000110010101000010100010011101001111011110011010100001101111101111','00010101110111100010010010111001000010001100110010001100000010000000001000100100','00110110000101101010010101111011010100001110011100001010111010110100001011110100','11010101000001000110100101001001001111011111000001001010010010111100100111101100','01110111110101011011111101100000010010001110001111111100110001100001001010110111','11100101010010101011010001001010000001001101110111101010011011101001100110100000','11101110110010010111000110111011011011000101111010111001101001010001001001111011','00101111010000111010001000000000010001011101111110001100100000110101001001111010','11011100010000001011001010101000110001000110100100111110001001100010000011101000','00010101000100011011000111101010011110000111000111001101101001100111000110110011','00000101010000001010101100111000010001001110001110001101011011101010000110101011','00101101110110001111010001001010010110010111101101011011100010011111100111110101','01011101000111001111100100100010000000011101001000011011010001011111000111111000','00110101000111101010011000011010110100001101011010111010011011110110100111100001','00111110100010011110100001110000011000010111110001101000001010011110001110100001','10101111010111100110000111001011111100011101000111011111000010110001000100100101','00110100100101011110100110100011001111010100100111001110011011011011101011100011','01110101100000011111110101010010110110000101000101101011110000001100100010101111','10000100110011000111110101110010010010011101011011101110001001110011001011101101','00101110010011101011100100010010110110001100000110001101010001100000100110111011','10100101000010010011101001111001111101001110010000111111001011110100101011110000','00101101110011001010101111111011011011011101110011101100011000100010101100101110','11010110010100010011110110011011000111000101010011111000110010000001101110101101','10001100110011110011011000110011010111010110101010011101011001011010000010101101','11011100110100101110001111101011000000011100000110011101111010110100001001101101','10110111000101100110110000110001010100001111111011011110000011000111001110110000','01110100110000010010000100100010111011000101101010011101001000011000000010101001','00110101110000101011011011111000001010011100101110111100101011010001100001100111','01010110100100111110010010001010101110010100110100111001001001001010100000110101','10100100000000101011010100010000000001001100100111101111001011011101001111100011','00010100110001001010100110110001001001000110010111011010010011100011101110111101','01111100010000101110100000101010001001010100011111101010101011010011101111111101','11000111010110101010010100001011000010011101010110001011110010111010100100111000','11010100100011001111011111100011000001001111100111101100010011010000000001110011','10011100010001010111001100000010100010011101101010001110101000000110101010100111','00101110010010011010100010110001000101011100110110111100010010110010001100100101','01101111100100001010001001111001000110001110010011101111001010011011001010111010','10100101110010101110111100010010100110011101100010111010001011010101100111110001','01011110110111111111110000100000011111001100111010011001110010110011000100100100','00011100000010001111000000010011110101011101011010001110000001000001100001101110','00010101000110111110100001000011010100001110011111001111000011100010100101100011','01001101110101101111011100000001010011000101001111111100101000000101001110111101','01110101010010101010101100001001000100010100111011001100000010010111000111100011','01111100010001011011001111010001001011000101010011111010110010100111101110100110','01000110010111010010011000000010110000001111010000011110101001101101101110101010','00000101110011011010001010110010000000011101110011101111101010000101001110110110','11100111110101000111001100101001110101000111101110101011100010100110001011100110','00001100110111110011010101100010101110000110110101111010110010001100001100100010','01010100010001111011101010101011011111011111101110101110101000100001001100100111','01111100100001011111001100110000001100001111011110111111001001000111100001101110','00001111010001011011100111100010011101011100010111001000110010001110000110110001','00011110000010101011100100011001011100011100000100011111111001000111000000100110','00011101110111011011011111011011010001010100111010101001011011100110100011110001','10001111100100101111010101100010001110001101001100111100011011001110101000110010','11011110000000011010001110110001010110001100001100011111000010110001100000100011','01001111010111000011100110111011011001000101001011111001000001110111100011111000','10101101010101111011010110101010110010000111110101111011000010010001100110101001','01100110010000110110110100101000011111011110001010001111011000100101000001111101','10011101100001110111001011111011111011000101101100001001110011010111100010110000','01010111110000011011011110011000010001001111010111011111101000001101101011111101','11000100010011001111110101100001000101010100110110111010111000000111001111101010','01110111000011110111000010100000010100011111110011011100100011100010100111110011','00011110010110011111101010100011000011010110001010011010010011110010101110101110','01101101110011111011001011110010010001000101101110001110010001000001001001101110','11010111100010001111101101000000100110011100111010101001001000111001101111100100','01001110100011101110001001000011011101001101100011011000110011010010001101110100','10101100000011111111010101100011011100001101010001101111000010111000100001101101','00100100110001100011001110100010010011010111101111001110001011101011100111110011','11000100110010010110111110100001101001000100111001001011010000010111000010101111','11100101010001011011111000001011001000001101110101111000011011001110001111100101','10111110100010111110111100011011001011011110010011101011011001000111000111100101','00010111100010100110001000010010011011011100100010001010101001010000100101111010','10110100110110011010001100100010110110001100101110111011011010100011000010111001','01001101010111001011011111101010000110010101111110001000101001100010000111110101','10010100010010011110110011000011100000011101000110011000000010110101001001100111','11000100110100010111111000011000110101001111001000111110000011110011000111111101','11101111100110100110000111000011110110001100111110001011110011001110001000111011','01110111010011011010001000000011011001000100000000111110100000110101000110110110','01011111100010111111001011010011000101001101110010001000100010110100001001111110','10000100110000111010000000100000010000001101010011101101001001101000100010100101','10011111010000001010101111010011100101010101000010011000101011111100101010111110','11011101000000111010110010011001001100001100001111111110001001010111000110110001','01111100100010001110100010110000000101000101101110111100000000000001100111110100','01111110000111111111001110111010001010001111000110001110001011000011001101111111','00100100110010011111010100001010011111001101110100001101011011011011101000110110','00100111010110000110111010100010010110000111001100111011010010111111100100110000','10000111100011110111000000010010111110011101001011111111001011010111001000111011','10101110010100011111001001101001111001011110111001001010010010011000001011111001','00101101110111011110001111110000000001010111011111011111001001110111101101111111','11010110110111100111001010110000011110010101111011101100110001110010101110111001','01110111110110111011000010011000010101011101010010001011000011100111101001100000','11100100010010111010101101000010110100011111011100001100010010111010000110110100','10100110000001011111010100100001001100001111010011011001110001000000001101110101','10011110010100111111101100001001100001011110000001001001001011000110100111111010','00011110110110111010110001000001001011000110000110001011110001100111100000101001','10110100100001001110011101010000011110001111010001011101000001101001101010100001','01111110000001101010010111000000101000011110101000001011110001110000000000111011','01001100010100001011001000101011001100011101000011001101101010001010001110100011','01100100110111001111000110110010010111010110011110001111011000100010101100100011','11110101100001001011011000011010011111000101111001101111001001000000000010110100','11011111100101011111011111100011111101001110100010001111011001001010001000110101','01110101010010011111001000100000111001001101100111001010100001110000001001100011','00011110010101010111111101001000001100011110110110111010001000100011100011110001','01101111000111111010011111011001101101010110001010101101110001011001001010110110','10011100000100010011011101000010100000001111110111111001001011110100000010111010','11011100010011001011111000001011100101010111110000011101001001110111000011100111','00000101010111101110010010110011011011001111000011001100011011110000101100110111','00011110000100001011000010000011001001001101100110011101000001100111000010110010','01000100110110110011001101001010111010000111011110111010111011101100101100100011','01111100010111101110110000000010000100000111011110011000100001010110101111100000','00011110100011011111110001110000011110011110010111111100110011000100101011110100','11011110000011010011110100010011100011001110110011111001000011110000100010100111','00011100110111001110101000010010011111001101111111001111011011110011000011110011','00100111110111011011010101100000000000011101010111011101010010010111001111111111','11010101010101111110100111011001010101011111011111111101101001000101101100111011','00111101110001101111010011011011100111001111001011001110011001010001001100101101','11100101000010110111110010000010010100001101000010101101010010100001000101111010','01100111100010001111010101000010011110010111001011101010101000010111001010110110','11011110000011101011100001010010000100011111111010011111001000111101000010101011','10011101100011111110100001110010101100001100110100101111011000110011000011110011','11000100110110101010100110111011101111011100111110111000101000000100000110100011','00000101100011100110101011100001010111000111100110101000000001000000101110110000','11001110110000111111001001111010001010011110111011111110000011011110001101100000','01101111110100011110011100011001011000011110110011101000011000110101001001110100','00010110000011000111001110100001111101011101001110101011000001110011001001101111','11111110000111001010101001001000000101011111110100001011000011101110001001110110','11010101010010011010000100100011010101001100010100101111011000101101001111100001','10010101110010111011111000110001000100011100101011001111110011011010000101111001','11110111110111111111001000110010000011011110100011011111101011001110000011110011','00010110010100101111100010011011001000000110111110001010000000010111100011111100','00000111000101001010001100001001001111011110100010101110100001010001100001111101','10001110100100101110110100101010001001011111010111001110010011110000101110110010','01001101000100010010111100001010110110010111100010111111101001000011100011100011','11010110000110101111100101010010101000011101010110001110010000111011001111111011','00111101110100011010101001110010110011010100111000001010110001000000000001111001','10001110100100110110100100101001100001001110111110011011000000000101100110100000','01000110000100001010001100111001010110000101111010011101011000001100101011110001','11001111010010000110111001010001011100001101101010111001001000011101000011110010','01111100000101001010001101111000001111001111010010101110111011000001101111110101','10110101100000001111110100101010001010011110011000001101001000101000000110100010','00010110000001101110001111000000000110001110010000001111110011001110000000101110','01000110100111010011100000000010010111001111110011001110010011101011100011111110','00001100010100101110011111010001011011001110001011001100101011010100001111100011','10010101110010100011010000101011010100001110010000001001000010011100100010100110','01011111110100111011110100000000010110001111001011011100100011000001101001110101','01010101110110100010100011111011111111010100000001011011110010001110101011100010','01111100100101011011111001110001000110011101111011101010010001010000101000111111','01101111110001001010001001101011000101011100101011011100100000000010000000101110','01110101110100001010110000101001101100000110000100011111101001011111000100101011','10100110010101111010011110111010001011011100010111001010110000101110100100100100','10010110100111100111011111100010111001001111100001011101101000000000101000100110','10110100000001100110001100000000101001011111011101011010011001110010000010100111','10011100010011001110001000001000000000001101110011001110010010011010101110110111','01000101000001000111001011001000000001011111011010111000001011110010101010101011','11101110010110111111000110001000101001001100101000011011110000011010001001111011','00110111100011111110011101000001000111010111111010101100010010000001000110100111']

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

plt.figure()
trials = 1000
steps = 10 #how far away do you want to look?
for idx in range(20):
	#top row is a counter 
	results = np.zeros((trials,steps))

	while sum(results[0,:])<(steps*(trials-1)):

		# pick a reference network at random

		net = nets[idx]

		# pick a nearby network at random
		comp = list(net)
		for z in range(80):
			if np.random.rand() < 0.1:
				#flip it
				comp[z] = bin(int(comp[z],2)^1)[2:]

		comp = "".join(comp)

		#find number of steps away:
		s = bin(int(net,2)^int(comp,2))[2:].count('1')

		if s<=steps:
			#find number of trials so far
			count = results[0,s-1]

			count = count + 1

			

			if count<trials:
				results[0,s-1] = count
				#find fitness
				results[count,s-1] = fitness(comp)

	freq = np.zeros(steps)

	for j in range(steps):
			freq[j] = (trials -1 - np.count_nonzero(results[1:,j] == 0))/trials
	#non-zeros



	plt.plot(np.arange(steps)+1,freq,'-')

plt.xlabel('mutations away from peak')
plt.ylabel('proportion with non-zero fitness')
plt.savefig("peak.png")


