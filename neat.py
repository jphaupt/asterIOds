#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 20:15:19 2018

@author: jph

NEAT implementation

Thought it might be fun to implement neuroevolution after all, but I do 
ultimately plan to run reinforcement learning

input : polar coordinates of all (or just top 5?) asteroids, and their size
    maybe also the polar coordinates of missiles, idk
    all shortest distance (recall that the map wraps!) 
    dx = abs(x1 - x2);
    if (dx > width/2)
        dx = width - dx;
    // again with x -> y and width -> height

output : probabilities of the following actions 
    shoot
    rotate left
    rotate right
    activate thrust

what parameters should be tuned by this genetic algo? Ideas : 
    # layers 
    # neurons per layer
    activation function per layer
probably just keep all layers as fully connected

for now: just do a vanilla network

simple version for now, perhaps: 
    3 layer perceptron 
    15 neuron input layer (5 closest rocks, for each have: size, loc in polar)
    ??? hidden layer width (say, 50?) 
    4 output neurons, as above (make sigmoid) 
doesn't make any use of previous frames though... can't infer any knowledge 
like rock velocity, etc.
later would like to use neuroevolution to also change the topology (# neurons),
etc.

TODO : decide what to use in the activation function for the hidden layer... 
idea : make this an evolved parameter ??

this is a very simple version of a genetic algorithm and at the present 
implementation does not involve any advanced techniques 

TODO : dare I say... MORE LAYERS ?!
"""
# TODO : would be nice to figure out how to parallelize this/make it concurrent
# TODO : crazy idea : try weight sharing?! 
# %% imports 
#import math
import numpy as np
from scipy.special import expit 
import asteroids
import random
#import operator
import time 
import matplotlib.pyplot as plt

# %% constants for evolution
#INITIAL_POP_SIZE = 10
NUM_ROCK_IN = 9
NUM_IN_PER_ROCK = 2 # TODO ? input size of rock 
INPUT_WIDTH = NUM_ROCK_IN * NUM_IN_PER_ROCK + 1 # one for # missiles fired 
OUTPUT_WIDTH = 4
HIDDEN_WIDTH = 40 # play with this parameter
IN_W_MUTATE_PROB = 0.25
OUT_W_MUTATE_PROB = 0.25
ACTIVATION_MUTATE_PROB = 0.1
NB_GAMES_PER_INDIV = 1 # TODO : increase, significantly 

# time it
timeStart = time.time()

# %% activation functions
# TODO : somehow encode this as something to be mutated. Dictionary? 

def linear(x) :
    return x    
    
def sigmoid(x):
#    print("sigmoid", x)
#    fn = lambda y : 1/(1+math.exp(-y)) 
#    fn = np.vectorize(fn)
    return expit(x) # TODO : figure out if expit deals with overflow

def tanh(x) :
    return 2*sigmoid(x) - 1

def relu(x) :
    fn = lambda y : np.max((0, y))
    fn = np.vectorize(fn) 
    return fn(x) 

# TODO : other activation functions to evolve from! 
    
#activations = [linear, sigmoid, tanh, relu, np.sin]  
activations = [relu, tanh] # check 100% relu performance - I suspect it's the best anyway

# %% 
class Individual() :
    '''
    an individual in the population, i.e. a neural network
    initializing one gives random weights
    '''    
    
    def __init__(self, activation, nb_input=INPUT_WIDTH, 
                 nb_hidden=HIDDEN_WIDTH, nb_output=OUTPUT_WIDTH,
                 rand_weights=True) :
        # TODO : something about random weights...
        self.nb_input = nb_input
        self.nb_hidden = nb_hidden
        self.nb_output = nb_output
        # includes bias
        # TODO : decide on normal mean and std. Should this be randomized 
        # during initialization? maybe a parameter to __init__? 
        if rand_weights :
            #[2*np.random.randn(nb_hidden, nb_input+1), 2*np.random.randn(nb_output, nb_hidden+1)]
            self.W = [np.random.normal(0, 2.5, (nb_hidden, nb_input+1)), 
                      np.random.normal(0, 2.5, (nb_output, nb_hidden+1))]
        else : #initialize to zeros for space preallocation
            self.W = [np.zeros((nb_hidden, nb_input+1)), np.zeros((nb_output, nb_hidden+1))] 
        self.activation = activation
    
    def predict(self, input_vect) : 
        input_vect = np.append(1, input_vect) # bias
        self.last_input = input_vect
#        print(input_vect) 
        self.last_h = np.dot(self.W[0], np.transpose(input_vect))
#        print(self.last_h) 
        self.last_h = self.activation(self.last_h)
#        print(self.last_h) 
        self.last_h  = np.append(1, self.last_h)
#        print(self.last_h) 
        self.last_out = np.dot(self.W[1], self.last_h)
        self.last_out = sigmoid(self.last_out)
        return self.last_out
    
# %% genetic functions 
def fitness(indiv) :
    '''
    fitness is simply the score from asteroids game
    TODO : average a few games to reduce randomly succeeding ? 
    currently averaging five games
    TODO : should I make this more complicated? time alive? # times cleared?
    '''
    acc = 0
    for i in range(NB_GAMES_PER_INDIV) : 
        acc += asteroids.game_loop(isAI=True, nn=indiv, visualize=False) 
    acc /= NB_GAMES_PER_INDIV
    return acc 

def generate_rand_indiv() : 
    '''
    randomly generates a random neural network (individual) 
    randomly selects (with equal probabilities) a hidden layer activation 
    function from the ones defined above
    note : current implementation has weights randomly initialized in the 
    constructor of the Individual class itself (maybe change?) 
    '''
    return Individual(random.choice(activations))
   
def generate_initial_pop(size_pop) :
    '''
    generate the initial population, which will be a random set of individuals
    these will eventually mutate to (hopefully) the optimal solution
    '''
    # TODO : no idea how to preallocate storage for this
    population = []
    i = 0
    while i < size_pop :
        population.append(generate_rand_indiv())
        i += 1
    return population

def compute_pop_score(population) :
    '''
    population performance (i.e. score in playing asteroids) 
    returns sorted list of individuals and their fitness
    '''
    # TODO : preallocation...
#    pop_score = {}
#    print(population)
    for indiv in population :
#        print("in loop!")
#        pop_score[indiv] = fitness(indiv)
        indiv.fitness = fitness(indiv)
    # highest score to lowest
#    print("left loop!")
#    return sorted(pop_score.items(), key = operator.itemgetter(1), reverse=True)
    return sorted(population, key = fitness, reverse=True)

def select_from_pop(population_sorted, best_sample, lucky_few) :
    '''
    select the top best_sample individuals from a population, as well as 
    lucky_few random individuals from a population
    '''
    # TODO : again, preallocation of data
    next_gen = []
    for i in range(best_sample):
        next_gen.append(population_sorted[i])
    for i in range(lucky_few):
        next_gen.append(random.choice(population_sorted))
    random.shuffle(next_gen) # important detail
    return next_gen

def create_child(individual1, individual2) :
    '''
    reproduce (create child) from two individuals
    randomly gets a vacancy or not from either parent
    returns a new individual based on the other two
    I am using a uniform crossover, but am not confident if this is the best
    TODO : improve baby-making algorithm
    
    this function is also known as crossover, recombination, breeding, 
    or sexy time
    
    this function will have to be significantly changed when/if implementing
    "true" NEAT (i.e. neuron topology, not just weights, change) 
    currently assumes that the two individuals having sexy time have the exact
    same neural architecture (well, except for hidden layer activation func)
    
    TODO : make probability of taking a certain allele proportional to the 
    parent's overall fitness? Done... not 100% on if it's the best idea
    but eh, I like the concept 
    '''
    # TODO : find a more efficient way to run this? I suspect this is a 
    # bottleneck for run time...
    W1 = individual1.W
    W2 = individual2.W
    BETA = 5 # initially aggressive Laplace smoothing
    prob1 = (individual1.fitness + BETA) / (individual1.fitness + individual2.fitness + 2*BETA) 
    if random.random() <= prob1 : # = case... ? 
        child = Individual(individual1.activation, rand_weights=False)
    else : 
        child = Individual(individual2.activation, rand_weights=False)
    a, b = child.W[0].shape # assume == W1[0].shape == W2[0].shape
    for i in range(a) :
        for j in range(b) :
            if random.random() <= prob1 :
                child.W[0][i][j] = W1[0][i][j]
            else : 
                child.W[0][i][j] = W2[0][i][j]
    # repeat for output layer...
    a, b = child.W[1].shape # assume == W1[1].shape == W2[1].shape
    for i in range(a) :
        for j in range(b) :
            if random.random() <= prob1 :
                child.W[1][i][j] = W1[1][i][j]
            else : 
                child.W[1][i][j] = W2[1][i][j]
            
    return child

def create_children(breeders, nb_children) :
    '''
    create children based on create_child method using the individuals that have 
    been selected to reproduce (based on select_from_pop method)
    i.e. lots of sexy times : produces the next population
    '''
    # TODO : preallocation
    next_pop = []
    for i in range(int(len(breeders)/2)) :
        for j in range(nb_children) :
            next_pop.append(create_child(breeders[i], breeders[len(breeders) -1 -i]))
    return next_pop

def mutate_W(individual) :
    '''
    mutate the given individual's weights 
    TODO : figure out a more appropriate way to handle mutation 
    (at current, I just add a random weight matrix) 
    TODO : play with std parameters
    NOTE this will have to be severely changed when/if implementing proper NEAT
    '''
    if random.random() <= IN_W_MUTATE_PROB : # mutate input layer
        mu = 0#random.uniform(-0.1, 0.1) # mean 
        sigma = random.uniform(0.05, 0.3) # std
        individual.W[0] += np.random.normal(mu, sigma, individual.W[0].shape) 
    if random.random() <= OUT_W_MUTATE_PROB : # mutation output layer weights
        mu = 0#random.uniform(-0.1, 0.1) 
        sigma = random.uniform(0.05, 0.3) 
        individual.W[1] += np.random.normal(mu, sigma, individual.W[1].shape)
        
def mutate_activation(individual) : 
    '''
    randomly changes the individual's hidden layer activation function
    '''
    if random.random() <= ACTIVATION_MUTATE_PROB:
        individual.activation = random.choice(activations)
    
def mutate_population(population):
    '''
    mutates the entire population (both weights and activation) 
    '''
    for indiv in population :
        mutate_W(indiv) 
        mutate_activation(indiv) 

def next_generation(curr_gen, nb_children, best_sample, lucky_few) : 
    '''
    breed the next generation from current generation and mutate 
    parameters may want to be tweaked. I'm using a naive genetic algo, AFAIK
    returns both the next generation and the best of the previous
    (for comparison and visualization... basically for fun)
    '''
#    print(curr_gen) 
    sorted_pop = compute_pop_score(curr_gen) 
#    print("sorted pop")
    breeders = select_from_pop(sorted_pop, best_sample, lucky_few)
#    print("best selected") 
    next_pop = create_children(breeders, nb_children)
#    print("had sexy times") 
    mutate_population(next_pop)
#    print(next_pop) 
    return next_pop, sorted_pop[0] 

def multi_gen(nb_generation, size_pop, best_sample, lucky_few, nb_children):
    '''
    start with a random population and produce nb_gen new generations based on 
    simple genetic algorithm
    returns a list of the best performing individual per generation
    '''
    # TODO : memory preallocation
    historic = []
    curr_pop = generate_initial_pop(size_pop)
#    historic.append(prev_best)
    for i in range(nb_generation):
#        print(i+1) # "progress bar"
        curr_pop, prev_best = next_generation(curr_pop, nb_children, best_sample, lucky_few)
        historic.append(prev_best) 
#        print(prev_best[1]) # current best performing individual in the population
        print("%i(%i)," % ((i+1), prev_best.fitness), end="")
    historic.append(compute_pop_score(curr_pop)[0]) # last best 
    return historic, curr_pop

# %% what actually happens when you run the program 
if __name__ == "__main__":
    # TODO : run for lots of generations 
    # hyperparameters for algorithm to run on
    size_population = 200 # size of the population each generation
    best_sample = 85 # how many of the most fit individuals reproduce in a population
    lucky_few = 15 # number of randomly selected individuals who get to reproduce (for genetic diversity)
    nb_children = 4 # how many offspring each couple produces
    nb_gens = 100 #  number of generations until program terminates
    
    # genetic algo
    if ((best_sample + lucky_few) / 2 * nb_children != size_population):
        	print ("population size not stable")
    else:
        print("generations completed:")
        # last population saved in case we want to revisit and continue evolving
        historic, curr_pop = multi_gen(nb_gens, size_population, best_sample, lucky_few, nb_children)
        historic = np.array(historic)
        historic_fitness = np.zeros((len(historic),))
        for i in range(len(historic)) :
            historic_fitness[i] = historic[i].fitness
        plt.plot(historic_fitness)
        plt.title("peak fitness per population")
        plt.show()
        reverse_ind = np.argsort(historic_fitness)
        best_of_gen = historic[reverse_ind]
        best = best_of_gen[-1]
        # show the best player's play style
        asteroids.game_loop(isAI=True, nn=best, visualize=True) 
        
        # TODO : add a convergence criteria so that it doesn't run unnecessarily long
        # TODO : might be fun to automatically run best of generation after each iteration
        # might slow down the algo though...
        
    print(time.time() - timeStart)
    