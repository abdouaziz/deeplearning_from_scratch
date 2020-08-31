import numpy as np 
from linear_forward import linear_forward



def  relu(x):

    Z= np.maximum(0,x)
    cache = x

    return Z, cache

def  sigmoid(x) :

    Z= 1 /(1+ np.exp(-x))
    cache= x

    return Z, cache


def linear_activation_forward_test_case():
    """
    X = np.array([[-1.02387576, 1.12397796],
 [-1.62328545, 0.64667545],
 [-1.74314104, -0.59664964]])
    W = np.array([[ 0.74505627, 1.97611078, -1.24412333]])
    b = 5
    """
    np.random.seed(2)
    A_prev = np.random.randn(3,2)
    W = np.random.randn(1,3)
    b = np.random.randn(1,1)
    return A_prev, W, b



def linear_activation_forward(A_prev, W, b, activation):

    if activation =="sigmoid":
        Z , linear_cache = linear_forward(A_prev,W,b)
        A , activation_cache = sigmoid(Z)
    elif activation == "relu":
        Z,linear_cache = linear_forward(A_prev,W,b)
        A,activation_cache = relu(Z)

    assert(A.shape == (W.shape[0] ,A_prev.shape[1]))
    cache = (linear_cache , activation_cache)

    return A, cache 

