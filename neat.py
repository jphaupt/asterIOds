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
    
hidden layers... lol idk ??

just as an exercise, I also plan to write this network entire in tensorflow 
(no keras) 

what parameters should be tuned by this genetic algo? Ideas : 
    # layers 
    # neurons per layer
    activation function per layer
probably just keep all layers as fully connected

for now: just do a vanilla network? 

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
"""
# %% imports 
import math
import numpy as np

# %% constants for evolution
INITIAL_POP_SIZE = 10
INPUT_WIDTH = 15 
OUTPUT_WIDTH = 4
HIDDEN_WIDTH = 20 # play with this parameter

# %% activation functions
# TODO : somehow encode this as something to be mutated
def linear(x) :
    return x
    
def sigmoid(x):
    fn = lambda x : 1/(1+math.exp(-x))
    fn = np.vectorize(fn)
    return fn(x) 

def tanh(x) :
    return 2*sigmoid(x) - 1

def relu(x) :
    fn = lambda x : np.max((0, x))
    fn = np.vectorize(fn) 
    return fn(x) 

# TODO : other activation functions to evolve from! 
    
# %% 
class Individual() :
    '''
    an individual in the population, i.e. a neural network
    initializing one gives random weights
    '''    
    
    def __init__(self, activation, nb_input=INPUT_WIDTH, 
                 nb_hidden=HIDDEN_WIDTH, nb_output=OUTPUT_WIDTH) :
        self.nb_input = nb_input
        self.nb_hidden = nb_hidden
        self.nb_output = nb_output
        # TODO : make gaussian to allow large outliers
        # includes bias
        self.W = [np.random.randn(nb_hidden, nb_input+1), np.random.randn(nb_output, nb_hidden+1)]
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