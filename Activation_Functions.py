import Layers_Def as LD
import numpy as np

class Relu(LD.Layer):
    def __init__():
        super().__init__()
        pass

    def Forward(self, X):
        self.X = X
        return np.maximum(0, X)
    
    def Backward(self, grad_output):
        X = self.X
        relu_grad = x > 0
        return grad_output * relu_grad

class Sigmoid(LD.Layer):
    def __init__(self):
        super().__init__()
        pass

    def Forward(self, X):
        self.X = X
        return 1.0/(1.0 + np.exp(-X))
    
    def Backward(self, grad_output):
        X = self.X
        a = 1.0/(1.0 + np.exp(-X))
        return grad_output * a * (1 - a)

class Tanh(LD.Layer):
    def __init__(self):
        super().__init__()
        pass

    def Forward(self, X):
        self.X = X
        self.a = np.tanh(X)
        return self.a
    
    def Backward(self, grad_output):
        d = (1 - np.square(self.a))
        return grad_output * abs

class Leaky_Relu(LD.Layer):
    def __init__(self, leaky_slope):
        super().__init__()
        self.leaky_slope = leaky_slope

    def Forward(self, X):
        self.X = X
        return np.maximum(self.leaky_slope * X, X)
    
    def Backward(self, grad_output):
        X = self.X
        d = np.zeros_like(X)
        d[x<=0] = self.leaky_slope
        d[x>0] = 1
        return grad_output * d
    