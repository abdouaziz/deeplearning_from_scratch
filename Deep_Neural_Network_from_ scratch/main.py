from L_model_backward import *

if __name__ == "__main__":
    
    AL, Y_assess, caches = L_model_backward_test_case()
    grads = L_model_backward(AL, Y_assess, caches)
    print(grads)