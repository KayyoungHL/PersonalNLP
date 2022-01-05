import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Layer, Dense, Input, Embedding, Dropout, LayerNormalization


def positional_encoding(seq_len, embed_dim, d_model=None):
    if d_model is None:
        d_model = embed_dim
    pos = np.array([np.arange(seq_len)]).T
    i = np.arange(embed_dim)
    angle = pos/np.power(10000, (((i//2)*2)/d_model))
    pos_encoding = (1+np.power((-1),i))/2 * np.sin(angle) + (1+np.power((-1),(i+1)))/2 * np.cos(angle)
    return tf.cast(pos_encoding[tf.newaxis, ...], dtype=tf.float32) # (1, seq_len, embed_dim)


# 패딩 마스크
def create_padding_mask(seq):
    seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


# 룩 어해드 마스크(디코더 data leakage를 방지하기 위함)
def create_look_ahead_mask(size):
    mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask  # (seq_len, seq_len)


# 스케일드 내적 어텐션
def scaled_dot_product_attention(q, k, v, mask):
    """
        Calculate the attention weights.
        q, k, v must have matching leading dimensions.
        k, v must have matching penultimate dimension, 
            i.e.: seq_len_k = seq_len_v => attention_weights랑 행렬곱 하려면 같아야 한다.
        The mask has different shapes depending on its type(padding or look ahead)
        but it must be broadcastable for addition.

        Args:
            q: query shape == (..., seq_len_q, depth_q)
            k:   key shape == (..., seq_len_k, depth_k)
            v: value shape == (..., seq_len_v, depth_v)
            mask: Float tensor with shape broadcastable
                to (..., seq_len_q, seq_len_k). Defaults to None.

        Returns:
                       output:            output shape == (..., seq_len_q, depth_v)
            attention_weights: attention_weights shape == (..., seq_len_q, seq_len_k)
    """
    matmul_qk = tf.matmul(q, k, transpose_b=True) # (..., seq_len_q, seq_len_k)
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk) # (..., seq_len_q, seq_len_k)

    # 마스킹
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v) # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads, hs_dim=None):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        
        self.hs_dim = hs_dim # 내가 추가한 부분

        if hs_dim is None:
            # 텐서플로우 예제는 depth는 무조건 d_model / num_heads 나누어 떨어져야함
            assert d_model % self.num_heads == 0

            # 아마도 Attention Is All You Need 논문에서 제안한 방식 때문인 듯.
            self.depth = d_model // self.num_heads

            self.wq = Dense(d_model)
            self.wk = Dense(d_model)
            self.wv = Dense(d_model)
        else:
            ## 이론상으로 가능한 self.hs_dim을 하이퍼 파라미터로 사용해보겠다.
            assert hs_dim % self.num_heads == 0

            self.depth = hs_dim//self.num_heads

            self.wq = Dense(hs_dim)
            self.wk = Dense(hs_dim)
            self.wv = Dense(hs_dim)

        self.dense = Dense(d_model)


    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)

        ex) 
                                                                    Input x.shape == (batch_size, seq_len, hs_dim) [※ hs_dim = num_heads * depth ]
        x = tf.reshape(x, (batch_size, seq_len, num_heads, depth))   ==>  x.shape == (batch_size, seq_len, num_heads, depth) [※ seq_len <-> num_heads ]
        x = tf.transpose(x, perm=[0, 2, 1, 3]),                      ==>  x.shape == (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])


    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model) or (batch_size, seq_len, hs_dim)
        k = self.wk(k)  # (batch_size, seq_len, d_model) or (batch_size, seq_len, hs_dim)
        v = self.wv(v)  # (batch_size, seq_len, d_model) or (batch_size, seq_len, hs_dim)

        # 멀티 해드 나누기.
        # All 1. [ hs_dim -> num_heads * depth ] ==> 2. [ seq_len <-> num_heads ]
        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # Scaled Dot-product Attention
        scaled_attention, attention_weights = scaled_dot_product_attention( # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
            q, k, v, mask)                                                  # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)

        # 멀티 해드 합치기. 
        # [ num_heads <-> seq_len ]
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        if self.hs_dim is None: # [ num_heads * depth -> d_model ]
            concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        else: # 내가 추가한 부분, # [ num_heads * depth -> hs_dim ]
            concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.hs_dim)) # (batch_size, seq_len_q, hs_dim)

        # output = concat_attention @ Wo + bo, Wo.shape = (hs_dim, d_model), bo.shape = (1, d_model)
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return \
    Sequential([
        Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, hs_dim=None):
        super().__init__()

        self.mha = MultiHeadAttention(d_model, num_heads, hs_dim)
        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)


    def call(self, x, training, mask):
        # self attention
        attn_output, _ = self.mha(x, x, x, mask)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)

        # skip connection + normalize
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        # point wise feed forwad network
        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)

        # skip connection + normalize
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class DecoderLayer(Layer):
    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1, hs_dim=None):
        super().__init__()

        self.mha1 = MultiHeadAttention(d_model, num_heads, hs_dim)
        self.mha2 = MultiHeadAttention(d_model, num_heads, hs_dim)

        self.ffn = point_wise_feed_forward_network(d_model, dff)

        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.layernorm3 = LayerNormalization(epsilon=1e-6)

        self.dropout1 = Dropout(dropout_rate)
        self.dropout2 = Dropout(dropout_rate)
        self.dropout3 = Dropout(dropout_rate)


    def call(self, x, enc_output, training, look_ahead_mask, padding_mask):
        # 마스크드 멀티헤드 셀프 어텐션
        attn1, attn_weights_block1 = self.mha1(x, x, x, look_ahead_mask)  # (batch_size, target_seq_len, d_model)
        attn1 = self.dropout1(attn1, training=training)

        # skip connection + normalize
        out1 = self.layernorm1(attn1 + x)

        # Encoder-Decoder 멀티헤드 어텐션
        # enc_output.shape == (batch_size, input_seq_len, d_model)
        attn2, attn_weights_block2 = self.mha2(
            enc_output, enc_output, out1, padding_mask)  # (batch_size, target_seq_len, d_model)
        attn2 = self.dropout2(attn2, training=training)

        # skip connection + normalize
        out2 = self.layernorm2(attn2 + out1)  # (batch_size, target_seq_len, d_model)

        # Point wise feed forward network
        ffn_output = self.ffn(out2)  # (batch_size, target_seq_len, d_model)
        ffn_output = self.dropout3(ffn_output, training=training)

        # skip connection + normalize
        out3 = self.layernorm3(ffn_output + out2)  # (batch_size, target_seq_len, d_model)

        return out3, attn_weights_block1, attn_weights_block2


class Encoder(Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                maximum_position_encoding, dropout_rate=0.1, hs_dim=None):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = Embedding(input_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding,
                                                self.d_model)

        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, dropout_rate, hs_dim)
                           for _ in range(num_layers)]

        self.dropout = Dropout(dropout_rate)


    def call(self, x, training, mask):
        seq_len = tf.shape(x)[1]

        # adding embedding and position encoding.
        x = self.embedding(x)  # (batch_size, input_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x = self.enc_layers[i](x, training, mask)

        return x  # (batch_size, input_seq_len, d_model)


class Decoder(Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size,
                maximum_position_encoding, dropout_rate=0.1, hs_dim=None):
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        self.embedding = Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(maximum_position_encoding, d_model)

        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, dropout_rate, hs_dim)
                           for _ in range(num_layers)]
        self.dropout = Dropout(dropout_rate)


    def call(self, x, enc_output, training,
            look_ahead_mask, padding_mask):

        seq_len = tf.shape(x)[1]
        attention_weights = {}

        x = self.embedding(x)  # (batch_size, target_seq_len, d_model)
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]

        x = self.dropout(x, training=training)

        for i in range(self.num_layers):
            x, block1, block2 = self.dec_layers[i](
                x, enc_output, training,
                look_ahead_mask, padding_mask)

        attention_weights[f'decoder_layer{i+1}_block1'] = block1
        attention_weights[f'decoder_layer{i+1}_block2'] = block2

        # x.shape == (batch_size, target_seq_len, d_model)
        return x, attention_weights


class Transformer(Model):
    def __init__(self, num_layers, d_model, num_heads, dff, input_vocab_size,
                target_vocab_size, pe_input, pe_target, dropout_rate=0.1, hs_dim=None):
        super().__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, dff,
                               input_vocab_size, pe_input, dropout_rate, hs_dim)

        self.decoder = Decoder(num_layers, d_model, num_heads, dff,
                               target_vocab_size, pe_target, dropout_rate, hs_dim)

        self.final_layer = Dense(target_vocab_size)


    def call(self, inputs, training):
        # Keras models prefer if you pass all your inputs in the first argument
        inp, tar = inputs

        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(inp, tar)

        enc_output = self.encoder(inp, training, enc_padding_mask)  # (batch_size, inp_seq_len, d_model)

        # dec_output.shape == (batch_size, tar_seq_len, d_model)
        dec_output, attention_weights = self.decoder(
            tar, enc_output, training, look_ahead_mask, dec_padding_mask)

        final_output = self.final_layer(dec_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights


    def create_masks(self, inp, tar):
        # Encoder padding mask
        enc_padding_mask = create_padding_mask(inp)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by
        # the decoder.
        look_ahead_mask = create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = create_padding_mask(tar)
        look_ahead_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, look_ahead_mask, dec_padding_mask


class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super().__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)

        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)


def accuracy_function(real, pred):
    accuracies = tf.equal(real, tf.argmax(pred, axis=2))

    mask = tf.math.logical_not(tf.math.equal(real, 0))
    accuracies = tf.math.logical_and(mask, accuracies)

    accuracies = tf.cast(accuracies, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    return tf.reduce_sum(accuracies)/tf.reduce_sum(mask)




if __name__ == "__main__":






    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8
    dropout_rate = 0.1
    EPOCHS = 20

    learning_rate = CustomSchedule(d_model)

    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,
                                        epsilon=1e-9)
    
    temp_learning_rate_schedule = CustomSchedule(d_model)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.Mean(name='train_accuracy')

    transformer = Transformer(
        num_layers=num_layers,
        d_model=d_model,
        num_heads=num_heads,
        dff=dff,
        input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
        target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
        pe_input=1000,
        pe_target=1000,
        rate=dropout_rate)

    checkpoint_path = "./checkpoints/train"

    ckpt = tf.train.Checkpoint(transformer=transformer,
                            optimizer=optimizer)

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')


    # The @tf.function trace-compiles train_step into a TF graph for faster
    # execution. The function specializes to the precise shape of the argument
    # tensors. To avoid re-tracing due to the variable sequence lengths or variable
    # batch sizes (the last batch is smaller), use input_signature to specify
    # more generic shapes.

    train_step_signature = [
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
        tf.TensorSpec(shape=(None, None), dtype=tf.int64),
    ]


    @tf.function(input_signature=train_step_signature)
    def train_step(inp, tar):
        tar_inp = tar[:, :-1]
        tar_real = tar[:, 1:]

        with tf.GradientTape() as tape:
            predictions, _ = transformer([inp, tar_inp],
                                        training = True)
            loss = loss_function(tar_real, predictions)

        gradients = tape.gradient(loss, transformer.trainable_variables)
        optimizer.apply_gradients(zip(gradients, transformer.trainable_variables))

        train_loss(loss)
        train_accuracy(accuracy_function(tar_real, predictions))
    
    import time
    for epoch in range(EPOCHS):
        start = time.time()

        train_loss.reset_states()
        train_accuracy.reset_states()

        # inp -> portuguese, tar -> english
        for (batch, (inp, tar)) in enumerate(train_batches):
            train_step(inp, tar)

            if batch % 50 == 0:
                print(f'Epoch {epoch + 1} Batch {batch} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

        if (epoch + 1) % 5 == 0:
            ckpt_save_path = ckpt_manager.save()
            print(f'Saving checkpoint for epoch {epoch+1} at {ckpt_save_path}')

        print(f'Epoch {epoch + 1} Loss {train_loss.result():.4f} Accuracy {train_accuracy.result():.4f}')

        print(f'Time taken for 1 epoch: {time.time() - start:.2f} secs\n')