import numpy as np


class Dense():
    """
    
    """
    def __init__(self, nodes = 1, activation = None):
        self.nodes = nodes
        self.activation = activation

    def __initial(self):
        self.weights = np.sqrt(np.random.rand(self.n_inputs, self.nodes)*2/self.n_inputs)
        self.bias = np.sqrt(np.random.rand(1, self.nodes)*2/self.n_inputs)    

    def call(self, input_layer):
        self.n_inputs = len(input_layer)
        self.__initial()
    
    def feedforward(self, inputs):
        z = inputs @ self.weights + self.bias
        activation = Activation(self.activation)()
        return z, activation(z)
    
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
    
    def none(self, x):
        return x
    
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
    
    def softmax(self, x):
        return np.exp(x)/np.sum(np.exp(x))   

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