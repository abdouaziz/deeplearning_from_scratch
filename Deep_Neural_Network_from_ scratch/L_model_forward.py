import numpy as np
from linear_activation_forward import linear_activation_forward


def L_model_forward_test_case_2hidden():
    np.random.seed(6)
    X = np.random.randn(5,4)
    W1 = np.random.randn(4,5)
    b1 = np.random.randn(4,1)
    W2 = np.random.randn(3,4)
    b2 = np.random.randn(3,1)
    W3 = np.random.randn(1,3)
    b3 = np.random.randn(1,1)
  
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    
    return X, parameters

def L_model_forward(X, parameters):


    caches=[]
    A=X
    L=len(parameters)//2

    for l in range(1 ,L):

        A_prev = A
        A, cache = linear_activation_forward(A_prev, parameters['W'+str(l)], parameters['b'+str(l)],"relu")
        A_prev=A
        caches.append(cache)

    AL, cache =linear_activation_forward(A_prev, parameters['W'+str(L)], parameters['b'+str(L)],"sigmoid")
    caches.append(cache)

    assert(AL.shape[1]== (1, X.shape[1]))

    return AL,caches



