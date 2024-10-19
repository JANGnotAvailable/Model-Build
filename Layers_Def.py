import numpy as np

class Layer():
    def __init__(self):
        self.Params = None
        pass

    def Forward(self, x):
        raise NotImplementedError
    
    def Backward(self, x, grad):
        raise NotImplementedError
    
    def Reg_Grad(self, reg):
        pass

    def Reg_Loss(self, reg):
        return 0.
    
    def Reg_Loss_Grad(self, reg):
        return 0

class Convolution(Layer):
    def __init__(self, in_Channels, out_Channels, Kernel_Size, Stride=1, Padding=0):
        super().__init__()
        self.Channels = in_Channels
        self.F = out_Channels
        self.Kernel = Kernel_Size
        self.Stride = Stride
        self.Padding = Padding
        self.Weight = np.random.randn(self.F, self.Channels, self.Kernel, self.Kernel)
        self.Bias = np.random.randn(out_Channels,)
        self.Params = [self.Weight, self.Bias]
        self.Grads = [np.zeros_like(self.Weight), np.zeros_like(self.Bias)]
        self.X = None
        self.reset_parameters()
    
    def reset_parameters(self):
        kaiming_uniform(self.Weight, a = np.sqrt(5))

        if self.Bias is not None:
            fan_in = self.Channels
            bound = 1 / np.sqrt(fan_in)
            self.Bias[:] = np.random.uniform(-bound, bound, (self.Bias.shape))
    
    def Forward(self, X):
        self.X = X
        N, Channel, X_height, X_width = self.X.shape
        F, _, F_height, F_width = self.Weight.shape

        X_pad = np.pad(self.X, (0,0), (0,0), (self.Padding, self.Padding), (self.Padding, self.Padding), mode='constant', constant_values=0)

        Out_height = 1 + int((X_height + 2 * self.Padding - F_height) / self.Stride)
        Out_width = 1 + int((X_width + 2 * self.Padding - F_width) / self.Stride)
        Out = np.zeros((N, F, Out_height, Out_width))

        for n in range(N):
            for f in range(F):
                hs = i * self.Stride
                for i in range(Out_height):
                    for j in range(Out_width):
                        ws = j * self.Stride
                        Out[n, f, i, j] = (X_pad[n, :, hs:hs + F_height, ws:ws + F_width] * self.Weight[f]).sum() + self.Bias[f]
        
        return Out
    
    def __call__(self, X):
        return self.Forward(X)
    
    def Backward(self, dZ):
        N, F, Z_height, Z_width = dZ.shape
        N, C, X_height, X_width = self.x.shape
        F, _, F_height, F_width = self.Weight.shape

        padding = self.Padding

        Height = 1 + (X_height + 2 * padding - F_height) // self.Stride
        Width = 1 + (X_width + 2 * padding - F_width) // self.Stride

        dX = np.zeros_like(self.X)
        dW = np.zeros_like(self.Weight)
        dB = np.zeros_like(self.Bias)

        X_pad = np.pad(self.X, [(0,0), (0,0), (padding, padding), (padding, padding)], mode='constant')
        dX_pad = np.pad(dX, [(0,0), (0,0), (padding, padding), (padding, padding)], mode='constant')

        for n in range(N):
            for f in range(F):
                dB[f] += dZ[n, f].sum()
                for i in range(Height):
                    hs = i * self.Stride
                    for j in range(Width):
                        ws = j *self.Stride
                        dW[f] += X_pad[n, :, hs:hs + F_height, ws:ws + F_width] * dZ[n, f, i, j]
                        dX_pad[n, :, hs:hs + F_height, ws:ws + F_width] += self.Weight[f] * dZ[n, f, i, j]
        
        dX = dX_pad[:, :, pad:pad + X_height, pad:pad + X_height]

        self.Grads[0] += dW
        self.Grads[1] += dB
        return dX
    
    def Reg_Grad(self, reg):
        self.Grads[0] += 2 * reg * self.Weight
    
    def Reg_Loss(self, reg):
        return reg * np.sum(self.Weight ** 2)
    
    def Reg_Loss_Grad(self, reg):
        self.Grads[0] += 2 * reg * self.Weight
        return reg * np.sum(self.Weight ** 2)

class Pool(Layer):
    def __init__(self, Pool_Param=(2,2,2)):
        super().__init__()
        self.pool_height, self.pool_width, self.Stride = Pool_Param
    
    def Forward(self, x):
        self.x = x
        N, C, H, W = x.shape

        pool_height , pool_width, stride = self.pool_height, self.pool_width, self.pool_stride

        height_out = 1 + (H - pool_height) // stride
        width_out = 1 + (W - pool_width) // stride
        out = np.zeros((N, C, height_out, width_out))

        for n in range(N):
            for c in range(C):
                for i in range(height_out):
                    si = stride * i
                    for j in range(width_out):
                        sj = stride * j
                        x_win = x[c, c, si:si + pool_height, sj:sj + pool_width]
                        out[n, c, i, j] = np.max(x_win)
        
        return out
    
    def Backward(self, dout):
        out = None
        x = self.x
        N, C, H, W = x.shape
        kH, kW, stride = self.pool_height, self.pool_width, self.Stride
        Out_height = 1 + (H - kH) // stride
        Out_width = 1 + (W - kW) // stride

        dX = np.zeros_like(x)

        for k in range(N):
            for l in range(C):
                for i in range(Out_height):
                    si = stride * i
                    for j in range(Out_width):
                        sj = stride * j
                        slice = x[k,l,si:si + kH, sj:sj+kW]
                        slice_max = np.max(slice)
                        dx[k, l, si:si + kH, sj:sj + kW] += (slice_max==slice) * dout[k,l,i,j]
    
        return dx

class Dense(Layer):
    def __init__(self, input_dim, out_dim, init_method=('random, 0.01')):
        super().__init__()
        random_method_name, random_value = init_method
        if random_method_name == 'random':
            self.Weight = np.random.randn(input_dim, out_dim) * random_value
            self.Bias = np.random.randn(1, out_dim) * random_value
        elif random_method_name == 'he':
            self.Weight = np.random.randn(input_dim, out_dim) * np.sqrt(2/input_dim)
            self.Bias = np.zeros((1, out_dim))
        elif random_method_name == 'xavier':
            self.Weight = np.random.randn(input_dim, out_dim) * np.sqrt(1/input_dim)
            self.Bias = np.random.randn(1, out_dim) * random_value
        elif random_method_name == 'zero':
            self.Weight = np.zeros((input_dim, out_dim))
            self.Bias = np.zeros((1, out_dim))
        else:
            self.Weight = np.random.randn(input_dim, out_dim) * random_value
            self.Bias = np.zeros((1, out_dim))
        
        self.Params = [self.Weight, self.Bias]
        self.Grads = [np.zeros_like(self.Weight), np.zeros_like(self.Bias)]

    def Forward(self, X):
        self.X = X
        x1 = X.reshape(X.shape[0], np.prod(X.shape[1:]))
        Z = np.matmul(X, self.Weight) + self.Bias
        return Z

    def Backward(self, dA_out):
        X = self.x
        x1 = X.reshape(X.shape[0], np.prod(X.shape[1:]))
        dW = np.dot(x1.T, dZ)
        dB = np.sum(dZ, axis=0, keepdims=True)
        dX = np.dot(dZ, np.transpose(self.Weight))
        dX = dx.reshape(x.shape)

        return dX
    
    def Reg_Grad(self, reg):
        self.grads[0] += 2 * reg * self.Weight
    
    def Reg_Loss(self, reg):
        return reg * np.sum(self.Weight ** 2)
    
    def Reg_Loss_Grad(self, reg):
        self.grads[0] += 2 * reg * self.Weight
        return reg * np.sum(self.Weight ** 2)