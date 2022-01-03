import numpy as np


class Dense():
    """
    
    """
    def __init__(self, nodes = 1, activation = None, initial = "he"):
        self.nodes = nodes
        self.activation = Activation(activation)
        self.initial = initial

    def __initial(self):
        if self.initial == "he":
            self.weights = np.sqrt(np.random.rand(self.n_inputs, self.nodes)*2/self.n_inputs)
            self.bias = np.sqrt(np.random.rand(1, self.nodes)*2/self.n_inputs)    

    def call(self, input_layer):
        self.n_inputs = len(input_layer)
        self.__initial()
    
    def forward(self, inputs):
        self.inputs = inputs
        z = self.inputs @ self.weights + self.bias
        a = self.activation()(z)
        return a
    
    def backward(self, inputs):
        da_dz = self.activation()(inputs, forward = False)
        dz_dw = self.inputs.T @ da_dz 
        dz_db = np.sum(da_dz)
        dz_dx = da_dz @ self.weights.T
        return dz_dw, dz_db, dz_dx
    
    __call__ = call


class Activation():
    """
    
    """
    def __init__(self, activation = None):
        self.activation = activation
        self.dicts = {
            None        :self.none,
            "sigmoid"   :self.sigmoid,
            "softmax"   :self.softmax
        }
        return self
    
    def call(self):        
        return self.dicts[self.activation]
    
    def none(self, x, forward = True):
        return x

    def sigmoid(self, x, forward = True):
        if forward:
            self.y = 1/(1+np.exp(-x))
            return self.y
        else:
            return x*(1-self.y)*self.y
    
    def softmax(self, x, forward = True):
        if forward:
            self.y = np.exp(x)/np.sum(np.exp(x))
            return self.y
        else:
            ...

    __call__ = call




## positional encoding
class Transformer():
    """

    """
    def __init__(self, seq_len = 10, embed_dim = 8, heads = 1, d_model = 8, hs_dim = 8):
        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.heads = heads
        self.d_model = d_model
        self.hs_dim = hs_dim
        
    def positional_encoding(self):
        self.pe = np.zeros(shape = (self.seq_len, self.embed_dim))


if __name__ == "__main__":
    model = Transformer()
    model.positional_encoding()
    print(model.pe)