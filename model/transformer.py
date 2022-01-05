import numpy as np
from lossfunction import LossFunction
from layers import Dense, InputLayer
from activation import Activation

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

        return \
            self.input_layer_q.backward(dav_dq),\
            self.input_layer_k.backward(dav_dk),\
            self.input_layer_v.backward(dav_dv)

    __call__ = call


# 멀티 해드 어텐션 => 병렬처리 안 하는 버전.
class MultiHeadAttention():
    """
        이거는 연습용으로 병렬처리가 진또배기!
    """
    def __init__(self, heads = 1, hs_dim = 8, initial = "he"):
        self.heads = heads
        self.hs_dim = hs_dim
        self.initial = initial
        self.dense_qs = []
        self.dense_ks = []
        self.dense_vs = []
        self.sdp_attentions = []
        for i in range(self.heads):
            self.dense_qs.append(Dense(self.hs_dim, initial=initial))
            self.dense_ks.append(Dense(self.hs_dim, initial=initial))
            self.dense_vs.append(Dense(self.hs_dim, initial=initial))
            self.sdp_attentions.append(ScaledDotProductAttention())

    def call(self, input_layer_q, input_layer_k, input_layer_v):
        self.input_layer_q = input_layer_q
        self.input_layer_k = input_layer_k
        self.input_layer_v = input_layer_v
        self.os = []
        for i in range(self.heads):
            q = self.dense_qs[i](self.input_layer_q)
            k = self.dense_ks[i](self.input_layer_k)
            v = self.dense_vs[i](self.input_layer_v)
            self.os.append(self.sdp_attentions[i](q, k, v))
        self.n_inputs = self.heads*self.hs_dim
        self.n_outputs = input_layer_q.n_outputs
        self.__initial()

        return self

    def __initial(self):
        if self.initial == "he":
            self.weights = np.random.normal(0,np.sqrt(2/self.n_outputs), (self.n_inputs, self.n_outputs))
            self.bias = np.random.normal(0,np.sqrt(2/self.n_outputs), (1, self.n_outputs))
        elif self.initial == "xavier":
            self.weights = np.random.normal(0,np.sqrt(2/(self.n_inputs+self.n_outputs)), (self.n_inputs, self.n_outputs))
            self.bias = np.random.normal(0,np.sqrt(2/(self.n_inputs+self.n_outputs)), (1, self.n_outputs))   

    def forward(self):
        ...
        ## concatenate 먼저 할 것.
        o = self.os[0].forward()
        for i in range(1, self.heads):
            o = np.concatenate((o,self.os[i].forward()), axis=0)
        self.inputs = o
        z = o @ self.weights + self.bias

        return z

    def backward(self, inputs):
        da_dw = self.inputs.T @ inputs
        da_db = np.array([np.sum(inputs, axis=0)])
        da_dx = inputs @ self.weights.T

        # 가중치 갱신
        self.renew(da_dw, da_db)

        # 역전파
        
        for i in range(self.heads):
            x = da_dx[...,i*self.hs_dim:(i+1)*self.hs_dim]
            try:
                da_dq, da_dk, da_dv += self.os[i].backward(x)
            except:
                da_dq, da_dk, da_dv = self.os[i].backward(x)

        return da_dq, da_dk, da_dv

    __call__ = call


class Encoder():
    """

    """
    def __init__(self, heads = 1, hs_dim = 8, dff=512):
        self.middle = InputLayer()
        self.mha = MultiHeadAttention(heads=heads, hs_dim=hs_dim, initial="he")

        self.middle2 = InputLayer()
        self.hidden = Dense(dff, activation='relu', initial='he')

    def call(self, input_layer):
        self.input_layer = input_layer
        self.n_output = self.input_layer.n_output

        # self attention
        self.mha(self.middle, self.middle, self.middle)

        # FFNN
        x = self.hidden(self.middle2)
        self.output = Dense(self.n_output, activation='relu', initial='he')
        x = self.output(x)

        return self
    
    def forward(self):
        x = self.input_layer.forward()
        self.middle(x)
        z = self.mha.forward()

        ## 노멀라이즈도 해야되는데... 일단 생략하고
        ## 스킵 커넥션
        x = z + x
        
        ## FFNN 
        self.middle2(x)
        z = self.output.forward()
        
        ## 스킵 커넥션
        return x + z

    def backword(self, inputs):
        ## FFNN 백워드
        dx_dz = self.output.backward(inputs)

        inputs = inputs + dx_dz # inputs는 스킵 커넥션으로 연결된 것!

        ## Multi-head Attention 백워드
        da_dq, da_dk, da_dv = self.mha.backward(inputs)
        
        inputs = inputs + da_dq + da_dk + da_dv
        self.input_layer.backward(inputs)
    
    __call__ = call


class Transformer():
    """

    """
    def __init__(self, heads = 1, d_model = 128, hs_dim = 8):
        self.heads = heads
        self.d_model = d_model
        self.hs_dim = hs_dim
        self.positional_encoding()
        self.encoder = Encoder()

    def positional_encoding(self):
        pos = np.array([np.arange(self.seq_len)]).T
        i = np.arange(self.embed_dim)
        angle = pos/10000**((i//2*2)/self.d_model)
        self.pe = (1+(-1)**i)/2 * np.sin(angle) + (1+(-1)**(i+1))/2 * np.cos(angle)

    def call(self, input_embed_layer):
        self.seq_len = input_embed_layer.shape[-2]
        self.embed_dim = input_embed_layer.shape[-1]

        

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
    dense_test()