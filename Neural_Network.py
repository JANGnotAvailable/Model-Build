import Layers_Def as LD
import numpy as np

class Neural_Network:
    def __init__(self):
        self.Layers = []
        self.params = []

    def add_Layer(self, layer):
        self.Layers.append(layer)
        if layer.params:
            for i, _ in enumerate(layer.params):
                self.params.append([layer.params[i], layer.grads[i]])
    
    def Forward(self, X):
        for layer in self.Layers:
            X = layer.Forward(X)
        return X
    
    def __call__(self, X):
        return self.Forward(X)
    
    def Predict(self, X):
        p = self.Forward(X)
        if p.ndim == 1:
            return np.argmax(ff)
        return np.argmax(p, axis=1)

    def Backward(self, loss_grad, reg=0):
        for i in reversed(range(len(self.Layers))):
            layer = self.Layers[i]
            loss_grad = layer.Backward(loss_grad)
            layer.Reg_Grad(reg)
        
        return loss_grad
    
    def Reg_Loss(self, reg):
        reg_loss = 0
        for i in range(len(self.Layers)):
            reg_loss += self.Layers[i].Reg_Loss(reg)
        
        return reg_loss
    
    def Parameters(self):
        return self.params
    
    def Zero_Grad(self):
        for i, _ in enumerate(self.params):
            self.params[i][1] *= 0