import numpy as np

class SGD():
    def __init__(self, model_params, learning_rate=0.01, momentum=0.9):
        self.params = model_params
        self.lr = learning_rate
        self.momentum = momentum

        self.vs = []
        for p, grad in self.params:
            v = np.zeros_like(p)
            self.vd.append(v)

    def zero_grad(self):
        for i, _ in enumerate(self.params):
            self.params[i][1].fill(0)
    
    def step(self):
        for i in enumerate(self.params):
            p, grad = self.params[i]
            self.vs[i] = self.momentum * self.vs[i] + self.lr * grad
    
    def scale_learning_rate(self, scale):
        self.lr *= scale

class Adam():
    def __init__(self, model_params, learning_rate=0.01, beta_1=0.9, beta_2=0.999, epsilon=1e-8):
        self.params = params
        self.lr = learning_rate
        self.beta_1, self.beta_2 = beta_1, beta_2
        self.epsilon = epsilon
        self.ms = []
        self.vs = []
        self.t = 0

        for p, grad in self.params:
            m = np.zeros_like(p)
            v = np.zeros_like(p)
            self.ms.append(m)
            self.vs.append(v)
        
    def zero_grad(self):
        for i, _ in params:
            self.params[i][1].fill(0)
    
    def step(self):
        beta_1, beta_2, lr = self.beta_1, self.beta_2, self.lr
        self.t += 1
        t = self.t
        for i, _ in enumerate(self.params):
            p, grad = self.params[i]

            self.ms[i] = beta_1 * self.ms[i] + (1 - beta_1) * grad
            self.vs[i] = beta_2 * self.vs[i] + (1 - beta_2) * grad ** 2

            m_1 = self.ms[i]/(1 - np.power(beta_1, t))
            v_1 = self.vs[i]/(1 - np.power(beta_2, t))
            self.params[i][0] -= lr * m_1/(np.sqrt(v_1) + self.epsilon)
        
    def scale_learning_rate(self, scale):
        self.lr *= scale