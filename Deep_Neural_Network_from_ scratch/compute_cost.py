import numpy as np




def compute_cost_test_case():
    Y = np.asarray([[1, 1, 0]])
    aL = np.array([[.8,.9,0.4]])
    
    return Y, aL


def compute_cost(AL, Y):

    m=Y.shape[1]

    cost = (-1/m)* np.sum(np.multiply(np.log(AL),Y) +np.multiply((1-Y), np.log(1-AL)))

    print("Je suis la ",cost)

    cost= np.squeeze(cost)

    print("Apres je suis la ",cost)

    return cost


