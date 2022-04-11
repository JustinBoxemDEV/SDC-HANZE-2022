"""
Basic implementation of a genetic algorithm to solve a minimization function and the onemax problem.

Resources:
https://www.section.io/engineering-education/the-basics-of-genetic-algorithms-in-ml/
"""
from numpy.random import randint
from numpy.random import rand
 
# Solve onemax 
def onemax(x):
	return -sum(x)


# Solve minimization function
def objective(x):
	return x[0]**2.0 + x[1]**2.0


def selection(pop, scores, k=3):
	# first random selection
	selection_ix = randint(len(pop))
	for ix in randint(0, len(pop), k-1):
		# check if better
		if scores[ix] < scores[selection_ix]:
			selection_ix = ix
	return pop[selection_ix]
 

def crossover(p1, p2, r_cross):
	c1, c2 = p1.copy(), p2.copy()
	if rand() < r_cross:
		# select crossover point that is not on the end of the string
		pt = randint(1, len(p1)-2)
		# perform crossover
		c1 = p1[:pt] + p2[pt:]
		c2 = p2[:pt] + p1[pt:]
	return [c1, c2]
 

def mutation(bitstring, r_mut):
	for i in range(len(bitstring)):
		# check for a mutation
		if rand() < r_mut:
			# flip the bit
			bitstring[i] = 1 - bitstring[i]


# decode bitstring to numbers
def decode(bounds, n_bits, bitstring):
	decoded = list()
	largest = 2**n_bits
	for i in range(len(bounds)):
		start, end = i * n_bits, (i * n_bits)+n_bits
		substring = bitstring[start:end]

		# convert bitstring to a string of chars
		chars = ''.join([str(s) for s in substring])

		# convert string to integer
		integer = int(chars, 2)

		# scale integer to desired range
		value = bounds[i][0] + (integer/largest) * (bounds[i][1] - bounds[i][0])

		decoded.append(value)
	return decoded

 
def genetic_algorithm_onemax(objective, n_bits, n_iter, n_pop, r_cross, r_mut):
	# initial population of random bitstring
	pop = [randint(0, 2, n_bits).tolist() for _ in range(n_pop)]
	
	best, best_eval = 0, objective(pop[0])
	
	for gen in range(n_iter):
		# evaluate all candidates
		scores = [objective(c) for c in pop]
		# check for new best solution
		for i in range(n_pop):
			if scores[i] < best_eval:
				best, best_eval = pop[i], scores[i]
				print(">%d, new best f(%s) = %.3f" % (gen,  pop[i], scores[i]))
		
		selected = [selection(pop, scores) for _ in range(n_pop)]
		# create the next generation
		children = list()
		for i in range(0, n_pop, 2):
			p1, p2 = selected[i], selected[i+1]

			for c in crossover(p1, p2, r_cross):
				mutation(c, r_mut)
				# store for next generation
				children.append(c)
		# replace population
		pop = children
	return [best, best_eval]

def genetic_algorithm_minimization(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut):
	# initial population of random bitstring
	pop = [randint(0, 2, n_bits*len(bounds)).tolist() for _ in range(n_pop)]

	best, best_eval = 0, objective(decode(bounds, n_bits, pop[0]))

	for gen in range(n_iter):
		# decode population
		decoded = [decode(bounds, n_bits, p) for p in pop]
		# evaluate all candidates
		scores = [objective(d) for d in decoded]
		# check for new best solution
		for i in range(n_pop):
			if scores[i] < best_eval:
				best, best_eval = pop[i], scores[i]
				print(">%d, new best f(%s) = %f" % (gen,  decoded[i], scores[i]))
		
		selected = [selection(pop, scores) for _ in range(n_pop)]
		# create the next generation
		children = list()
		for i in range(0, n_pop, 2):
			p1, p2 = selected[i], selected[i+1]
			
			for c in crossover(p1, p2, r_cross):
				mutation(c, r_mut)
				# store for next generation
				children.append(c)
		# replace population
		pop = children
	return [best, best_eval]
 
# onemax
n_iter = 100
n_bits = 20
n_pop = 100
r_cross = 0.9
r_mut = 1.0 / float(n_bits)

best, score = genetic_algorithm_onemax(onemax, n_bits, n_iter, n_pop, r_cross, r_mut)
print('Done')
print('f(%s) = %f' % (best, score))

# minization function
bounds = [[-5.0, 5.0], [-5.0, 5.0]]
n_iter = 100
n_bits = 16
n_pop = 100
r_cross = 0.9
r_mut = 1.0 / (float(n_bits) * len(bounds))

best, score = genetic_algorithm_minimization(objective, bounds, n_bits, n_iter, n_pop, r_cross, r_mut)
print('Done')
decoded = decode(bounds, n_bits, best)
print('f(%s) = %f' % (decoded, score))