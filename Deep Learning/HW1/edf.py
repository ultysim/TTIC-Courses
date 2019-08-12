import numpy as np

DT=np.float32
eps=1e-12

# Globals
components = []
params = []


############################################## Utility components####################################

# Global forward/backward
def Forward():
    for c in components: c.forward()

def Backward(loss):
    for c in components:
        if c.grad is not None: c.grad = DT(0)
    loss.grad = np.ones_like(loss.value)
    for c in components[::-1]: c.backward()

# Optimization functions
def SGD(lr):
    for p in params:
        p.value = p.value - lr*p.grad
        p.grad = DT(0)

# Values
class Value:
    def __init__(self,value=None):
        self.value = DT(value).copy()
        self.grad = None
        
        
    def set(self,value):
        self.value = DT(value).copy()

# Parameters
class Param:
    def __init__(self,value,opts = {}):
        self.value = DT(value).copy()
        self.opts = {}
        params.append(self)
        self.grad = DT(0)
        
        
# Xavier initializer
def xavier(shape):
    sq = np.sqrt(3.0/np.prod(shape[:-1]))
    return np.random.uniform(-sq,sq,shape)


# Verify the circuit is valid by checking the gradient of specific component
'''
def Verify(loss, c, epsilon = .001):
    
    Forward()
    Backward(loss)
    V0 = loss.value
    size = np.int32(np.prod(c.grad.shape))
    numgrad = np.empty(size)
    linval = c.value.reshape(size)
    for j in range(size):
        oldval = linval[j]
        linval[j] = oldval + epsilon
        forward_users(c)
        numgrad[j] = (loss.value - V0)/epsilon
        linval[j] = oldval
        forward_users(c)           
            
    diff = np.linalg.norm(numgrad.reshape(c.grad.shape) - c.grad)
    print ("l2 norm difference between backprop gradient and numerical gradient:")
    print (diff/(np.linalg.norm(c.grad)))
                
def forward_users(c):
    for user in c.users:
        user.forward()
        forward_users(user)
    return

'''

######################### Actual Components#############################
'''
Class Name: Add
Class Usage: add two vectors
Class Functions:
     forward: compute the result x + y
     backward: compute the derivative w.r.t x and y
 '''
class Add:
    def __init__(self,x,y):
        components.append(self)
        self.x = x
        self.y = y
        self.grad = None if x.grad is None and y.grad is None else DT(0)

    def forward(self):
        self.value = self.x.value + self.y.value

    def backward(self):
        if self.x.grad is not None:
            self.x.grad = self.x.grad + self.grad

        if self.y.grad is not None:
            self.y.grad = self.y.grad + self.grad

'''
Class Name: Mul
Class Usage: elementwise multiplication with two vectors 
Class Functions:
    forward: compute the result x*y
    backward: compute the derivative w.r.t x and y
'''
class Mul:
    def __init__(self,x,y):
        components.append(self)
        self.x = x
        self.y = y
        self.grad = None if x.grad is None and y.grad is None else DT(0)

    def forward(self):
        self.value = self.x.value * self.y.value

    def backward(self):
        if self.x.grad is not None:
            self.x.grad = self.x.grad + self.grad*self.y.value

        if self.y.grad is not None:
            self.y.grad = self.y.grad + self.grad*self.x.value

'''
Class Name: VDot
Class Usage: multiply vector with matrix where x is vector and y is matrix
y is expected to be a parameter and there is a convention that parameters come last.
Class Functions:
forward: compute the vector matrix multplication result
backward: compute the derivative w.r.t x and y, where derivative of x is a vector and derivative of y is a matrix 
'''
class VDot: # Matrix multiply (fully-connected layer)
    def __init__(self,x,y):
        components.append(self)
        self.x = x
        self.y = y
        self.grad = None if x.grad is None and y.grad is None else DT(0)

    def forward(self):
        self.value = np.dot(self.x.value,self.y.value)

    def backward(self):
        if self.y.grad is not None:
            self.y.grad = self.y.grad + np.outer(self.x.value, self.grad)
        if self.x.grad is not None:
            self.x.grad = self.x.grad + np.dot(self.y.value, self.grad)

'''
Class Name: Sigmoid
Class Usage: compute the sigmoid activation. Input is vector [x_{0}, x_{1}, ..., x_{n}],
output is vector [y_{0}, y_{1}, ..., y_{n}] where y_{i} = 1/(1 + exp(-x_{i}))
Class Functions:
    forward: compute activation y_{i} for all i.
    backward: compute the derivative w.r.t input vector x  
'''
class Sigmoid:
    def __init__(self,x):
        components.append(self)
        self.x = x
        self.grad = None if x.grad is None else DT(0)

    def forward(self):
        self.value = 1. / (1. + np.exp(-self.x.value))

    def backward(self):
        if self.x.grad is not None:
            self.x.grad = self.x.grad + self.grad * self.value * (1.-self.value)

'''
Class Name: SoftMax
Class Usage: compute the softmax activation. Input is vector [x_{0}, x_{1}, ..., x_{n}],
output is vector [p_{0}, p_{1}, ..., p_{n}] where p_{i} = exp(x_{i})/(exp(x_{0}) + ... + exp(x_{n}))
Class Functions:
    forward: compute probability p_{i} for all i.
    backward: compute the derivative w.r.t input vector x 
'''
class SoftMax:
    def __init__(self,x):
        components.append(self)
        self.x = x
        self.grad = None if x.grad is None else DT(0)

    def forward(self):
        lmax = np.max(self.x.value,axis=-1,keepdims=True)
        ex = np.exp(self.x.value - lmax)
        self.value = ex / np.sum(ex,axis=-1,keepdims=True)

    def backward(self):
        if self.x.grad is None:
            return
        gvdot = np.dot(self.grad, self.value)
        self.x.grad = self.x.grad + self.value * (self.grad - gvdot)

'''
Class Name: Log
Class Usage: compute the elementwise log(x) given x.
Class Functions:
    forward: compute log(x)
    backward: compute the derivative w.r.t input vector x
'''
class Log: # Elementwise Log
    def __init__(self,x):
        components.append(self)
        self.x = x
        self.grad = None if x.grad is None else DT(0)

    def forward(self):
        self.value = np.log(self.x.value)

    def backward(self):
        if self.x.grad is not None:
            self.x.grad = self.x.grad + self.grad/self.x.value

'''
Class Name: Aref
Class Usage: get one specific entry in a vector. x is the vector and idx is the entry index and x is differentiable.
Class Functions:
    forward: compute x[idx]
    backward: compute the derivative w.r.t input vector x
'''
class Aref: # out = x[idx]
    def __init__(self,x,idx):
        components.append(self)
        self.x = x
        self.idx = idx
        self.grad = None if x.grad is None else DT(0)
    def forward(self):

        self.value = self.x.value[np.int32(self.idx.value)] 
        self.shape = self.x.value.shape
    def backward(self):
        if self.x.grad is not None:
            self.x.grad = np.zeros(self.shape)
            self.x.grad[np.int32(self.idx.value)] = self.x.grad[np.int32(self.idx.value)] + self.grad

'''
Class Name: Accuracy
Class Usage: check the predicted label is correct or not.
x is the probability vector where each probability is for each class. idx is ground truth label.
Class Functions:
    forward: find the label that has maximum probability and compare it with the ground truth label.
    backward: None 
'''
class Accuracy:
    def __init__(self,x,idx):
        components.append(self)
        self.x = x
        self.idx = idx
        self.grad = None

    def forward(self):
        self.value = (np.argmax(self.x.value,axis=-1)==self.idx.value)

    def backward(self):
        pass
