#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

batch_size          = 64
input_dimension     = 1000
hidden_dimension    = 100
output_dimension    = 10
training_iterations = 500

# Create batch_size number of random input vectors
x = np.random.randn(batch_size, input_dimension)

# Create batch_size number of random "gold standard" label vectors
labels = np.random.randn(batch_size, output_dimension)

# Create a random weight matrix connecting the input to the hidden layer
w1 = np.random.randn(input_dimension, hidden_dimension)

# Create a random weight matrix connecting the hidden layer to the output layer
w2 = np.random.randn(hidden_dimension, output_dimension)

# Specify a learning rate
learning_rate = 1e-6

for iteration in range(training_iterations):

    ################
    # Forward pass #
    ################

    # Compute value of hidden layer before activation function
    pre_h = x.dot(w1)

    # Apply activation function, thus creating the final values of the hidden layer
    #
    # Here we are using a rectified linear unit (ReLU) as the activation function.
    # The ReLU function is defined as f(x) = max(x, 0)
    h = np.maximum(pre_h, 0)

    # Calculate the output layer. No activation function is applied.
    output = h.dot(w2)
    

    ##################
    # Calculate loss #
    ##################

    # Calculate loss as the mean squared error between the actual output value and the label
    loss = np.square(output - label).mean()
