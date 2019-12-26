# manually writing a simple neural network
import numpy as np

# sigmoid function with runs in each neuron
def nonlin(x, deriv = False): # deriv is false by default
    if(deriv == True):
        return (x *(1-x))
    if(deriv == False):
        return 1/(1+np.exp(-x)) # output if derivative is not true


# set a seed value
np.random.seed(123)

# lets generate some input data
x = np.array([[0,0,1],
             [0,1,1],
             [1,0,1],
             [1,1,1]])
# and output
y = np.array([[0],
             [1],
             [1],
             [0]])

# creation of synapses
s0 = 2*np.random.random((3,4)) - 1 # 3x4 matrix, and 1 is our bias
s1 = 2*np.random.random((4,1)) - 1 # 4 nodes 1 output

# develop the neural net with backprop
# develop a loop
for i in range(100000):

    l0 = x # input our data into layer 0
    l1 = nonlin(np.dot(l0, s0)) # matrix multiply
    l2  = nonlin(np.dot(l1, s1)) # matrix multiply




    # back prop
    l2_error = y-l2 # init value

    # now we add a check
    if (i % 10000) == 0:
        print('Current error: ', np.mean(np.abs(l2_error))) # print absolute value of the average at intervals

    # determine delta/change
    l2_delta = l2_error * nonlin(l2, deriv=True) # define a delta value
    l1_error = l2_delta.dot(s1.T)
    l1_delta = l1_error * nonlin(l1, deriv= True)

    # synapses updates
    s1 += l1.T.dot(l2_delta)
    s0 += l0.T.dot(l1_delta)

print('final is equal to : ', l2)

