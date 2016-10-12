import numpy as np

def max_return(one_hot):
    try:
        max_loc = np.argmax(one_hot, axis = 0)
        #max_val = one_hot[max_loc]
        return max_loc
    except:
        print "Something wrong with your array"
