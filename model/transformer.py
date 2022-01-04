import numpy as np

class Activation():
    """
    
    """
    def __init__(self, activation = None):
        self.activation = activation
        self.dicts = {
            None        :self.none,
            "sigmoid"   :self.sigmoid,
            "softmax"   :self.softmax,
            "relu"      :self.relu
        }
    
    def call(self, x, backward = False):
        return self.dicts[self.activation](x, backward = backward)  
    
    def none(self, x, backward = False):
        return x

    def sigmoid(self, x, backward = False):
        if not backward:
            self.y = 1/(1+np.exp(-x))
            return self.y
        else:
            return x*(1-self.y)*self.y
    
    def softmax(self, x, backward = False):
        if not backward:
            self.y = np.exp(x)/np.sum(np.exp(x), axis=0)
            return self.y
        else: # https://e2eml.school/softmax.html
            return x * (self.y * np.identity(self.y.shape[-1]) - self.y.T @ self.y)
    
    def relu(self, x, backward = False):
        if not backward:
            self.y = np.maximum(0, x)
            return self.y
        else:
            return x * (self.y != 0)
            
    __call__ = call


class LossFunction():
    """
    
    """
    def __init__(self, loss = None):
        self.loss = loss
        self.dicts = {
            "binary_crossentropy"   :self.binary_crossentropy,
            "categorical_crossentropy":self.categorical_crossentropy
        }
    
    def call(self, y_pred, y_true, backward = False):
        return self.dicts[self.loss](y_pred, y_true, backward = backward)
    
    def binary_crossentropy(self, y_pred, y_true, backward = False):
        if not backward:
            self.y = - (y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
            return np.mean(self.y)
        else:
            return - (y_true/y_pred - (1 - y_true)/(1 - y_pred))
        
    def categorical_crossentropy(self, y_pred, y_true, backward = False):
        if not backward:
            self.y = - np.sum(y_true * np.log(y_pred))
            return np.mean(self.y)
        else:
            return - np.sum(y_true/y_pred)
    
    __call__ = call


class InputLayer():
    """
    
    """
    def __init__(self, shape=None):
        if shape:
            self.n_outputs = shape
    
    def call(self, inputs):
        self.inputs = inputs
        self.n_outputs = inputs.shape[-1]

    def forward(self):
        return self.inputs
    
    def backward(self, inputs):
        return None


    __call__ = call

class Dense():
    """
    
    """
    def __init__(self, units, activation = None, initial = "xavier"):
        self.n_outputs = units
        self.activation = Activation(activation)
        self.initial = initial

    def __initial(self):
        if self.initial == "he":
            self.weights = np.random.normal(0,np.sqrt(2/self.n_outputs), (self.n_inputs, self.n_outputs))
            self.bias = np.random.normal(0,np.sqrt(2/self.n_outputs), (1, self.n_outputs))
        elif self.initial == "xavier":
            self.weights = np.random.normal(0,np.sqrt(2/(self.n_inputs+self.n_outputs)), (self.n_inputs, self.n_outputs))
            self.bias = np.random.normal(0,np.sqrt(2/(self.n_inputs+self.n_outputs)), (1, self.n_outputs))

    def call(self, input_layer):
        self.input_layer = input_layer
        self.n_inputs = self.input_layer.n_outputs
        self.__initial()

        return self
    
    def forward(self):
        self.inputs = self.input_layer.forward()
        z = self.inputs @ self.weights + self.bias
        a = self.activation(z)

        return a
    
    def backward(self, inputs):
        da_dz = self.activation(inputs, backward = True)
        da_dw = self.inputs.T @ da_dz 
        da_db = np.array([np.sum(da_dz, axis=0)])
        da_dx = da_dz @ self.weights.T

        # 가중치 갱신
        self.renew(da_dw, da_db)

        # 역전파
        self.input_layer.backward(da_dx)

    def renew(self, da_dw, da_db, lr = 0.001):
        self.weights = self.weights - lr * da_dw
        self.bias = self.bias - lr * da_db
    
    __call__ = call


class ScaledDotProductAttention():
    """

    """
    def __init__(self):
        self.activation = Activation("softmax")
        self.n_outputs = 1
    

    def call(self, input_layer_q, input_layer_k, input_layer_v):
        self.input_layer_q = input_layer_q
        self.input_layer_k = input_layer_k
        self.input_layer_v = input_layer_v
        
        self.n_input_q = input_layer_q.n_outputs
        self.n_input_k = input_layer_k.n_outputs
        self.n_input_v = input_layer_v.n_outputs

        return self
        
    def forward(self):
        self.q = self.input_layer_q.forward()
        self.k = self.input_layer_k.forward()
        self.v = self.input_layer_v.forward()
        qk = self.q @ self.k
        self.a = self.activation((qk)/np.sqrt(self.n_input_k)) 
        self.av = self.a @ self.v

        return self.av

    def backward(self, x):
        dav_da = x @ self.v.T
        dav_dv = self.a.T @ x
        dav_dqk = self.activation(dav_da, backward = True) # dav_da * da_dqk = dav_dqk
        dav_dq = dav_dqk @ self.q.T
        dav_dk = self.k.T @ dav_dqk

        self.input_layer_q.backward(dav_dq)
        self.input_layer_k.backward(dav_dk)
        self.input_layer_v.backward(dav_dv)

    __call__ = call


class MultiHeadAttention():
    """
    
    """
    def __init__(self, hs_dim = 8, heads = 1):
        self.dense_q = Dense(hs_dim)
        self.dense_k = Dense(hs_dim)
        self.dense_v = Dense(hs_dim)


class Transformer():
    """

    """
    def __init__(self, heads = 1, d_model = 128, hs_dim = 8):
        self.heads = heads
        self.d_model = d_model
        self.hs_dim = hs_dim
        self.positional_encoding()

    def positional_encoding(self):
        pos = np.array([np.arange(self.seq_len)]).T
        i = np.arange(self.embed_dim)
        angle = pos/10000**((i//2*2)/self.d_model)
        self.pe = (1+(-1)**i)/2 * np.sin(angle) + (1+(-1)**(i+1))/2 * np.cos(angle)

    def call(self, input_embed_layer):
        self.seq_len = input_embed_layer.shape[-2]
        self.embed_dim = input_embed_layer.shape[-1]

        Dense()(input_embed_layer)

        return self


class Model():
    """
    
    """
    def __init__(self, inputs, outputs):
        self.inputs = inputs
        self.outputs = outputs

    def compiler(self, optimizer, loss, metrics):
        self.optimizer = optimizer
        self.loss = LossFunction(loss)
        self.metrics = metrics

    def fit(self, X, y, epochs = 1):
        self.X = X
        self.inputs(X)
        self.y = y
        for i in range(1, epochs+1):
            y_pred = self.outputs.forward()
            loss = self.loss(y_pred, self.y)
            self.outputs.backward(self.loss(y_pred, self.y, backward=True))
            print(f"\repoch = {i} => {round(i/epochs*100,2)} %, loss : {loss}")#, end="")


    def predict(self, X):
        self.X = X
        self.inputs(X)
        return self.outputs.forward()



def dense_test():
    import matplotlib.pyplot as plt
    def test(seed = 0):
        np.random.seed(seed)

        x11 = np.random.uniform(low=0, high=5, size=(50,))
        x12 = np.random.uniform(low=10, high=15, size=(50,))
        x21 = np.random.uniform(low=0, high=5, size=(50,))
        x22 = np.random.uniform(low=10, high=15, size=(50,))


        x1 = np.append(x11, x12)
        x2 = np.append(x21, x22)

        y11 = np.random.uniform(low=10, high=15, size=(50,))
        y12 = np.random.uniform(low=0, high=5, size=(50,))
        y21 = np.random.uniform(low=0, high=5, size=(50,))
        y22 = np.random.uniform(low=10, high=15, size=(50,))

        y1 = np.append(y11, y12)
        y2 = np.append(y21, y22)

        x_1 = np.vstack([x1, y1]).T
        x_2 = np.vstack([x2, y2]).T
        y_1 = np.ones_like(x_1[:, 0])
        y_2 = np.zeros_like(x_2[:, 0])
        x = np.vstack([x_1, x_2])
        y = np.hstack([y_1, y_2])

        # fig, ax = plt.subplots(figsize = (12,5))
        # ax.plot(x_1[:, 0], x_1[:,1], 'bo')
        # ax.plot(x_2[:,0], x_2[:,1], 'ro')
        # ax.grid()

        return x, y
    X_train, y_train = test()
    y_train = np.array([y_train]).T
    inputs = InputLayer(shape=X_train.shape[-1])
    dense = Dense(2, activation='relu', initial='he')(inputs)
    dense = Dense(1, activation='sigmoid')(dense)
    model = Model(inputs, dense)

    model.compiler(optimizer=None,loss='binary_crossentropy',metrics=None)
    model.fit(X_train, y_train, epochs=200)

    X_test, y_test = test(42)
    preds = model.predict(X_test)
    preds_1d = preds.flatten()
    pred_class = np.where(preds_1d > 0.5, 1 , 0)

    y_true = X_test[pred_class==1]
    y_false = X_test[pred_class==0]

    fig, ax = plt.subplots(figsize = (12,5))
    ax.plot(y_true[:, 0], y_true[:,1], 'bo')
    ax.plot(y_false[:,0], y_false[:,1], 'ro')
    ax.grid()
    plt.show()

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    # model = Transformer()
    # pe = model.pe
    # print(np.round(pe,3))
    # ax = plt.subplot()
    # plt.pcolormesh(pe, cmap='RdBu')
    # plt.xlabel('Depth')
    # ax.xaxis.tick_top()
    # ax.xaxis.set_label_position('top') 
    # plt.xlim((0, 128))
    # plt.ylabel('Position')
    # plt.ylim((128,0))
    # plt.colorbar()
    # plt.show()

    # test = Activation("relu")
    # forward = test(np.array([[1,-2,3,-4,5,-6,7,-8,9]]))
    # backward = test(np.array([[1,3,2,4,3,6,4,8,5]]), backward = True)
    # print(forward)
    # print(backward)
    dense_test()