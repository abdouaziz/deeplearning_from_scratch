
import numpy as np


def initialize_parameters_deep(layer_dims):


    np.random.seed(3)

    parameters = {}
    L = len(layer_dims)
    for l in range (1, L):
        parameters['W'+str(l)] = np.random.randn(layer_dims[l] , layer_dims[l-1])*0.01
        parameters['b'+str(l)] =  np.zeros((layer_dims[l] , 1))

        assert(parameters["W"+str(l)].shape == (layer_dims[l] , layer_dims[l-1]))
        assert(parameters["b" +str(l)].shape == (layer_dims[l] , 1))
    
    print("the shape of W1",parameters["W1"].shape)
    print("the shape of W1",parameters["W2"].shape)
    
    return parameters 

    