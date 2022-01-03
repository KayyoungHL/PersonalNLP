from korean import chosung_tokenizer
from crawling import connect_api, get_tweets_by_quary
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import Dense, Input, LSTM, Bidirectional, Attention, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
import pandas as pd
import re

def get_tweets():
    quary = "딥러닝"
    api = connect_api()
    tweets = get_tweets_by_quary(api, quary, count = 100)

    return tweets


def preprocessing(text):
    step1 = text.split("\n")
    step2 = [re.sub("[^ㄱ-ㅎ가-힣 .?!\"\']", "", i) for i in step1]

    return step2


def nlp_tokenize(X, maxlen=100, num_words=1000):
    # 토큰화
    tokenizer = Tokenizer(num_words = num_words)
    tokenizer.fit_on_texts(X)
    X_seq = tokenizer.sequences_to_texts(X)

    # 0 패딩
    X_seq_matrix = sequence.pad_sequences(sequences=X_seq, maxlen=maxlen, padding='post')

    # 워드 인덱싱, 정수 인코딩
    word_encoded = tokenizer.word_index

    return X_seq_matrix, word_encoded


def create_encoder(maxlen=100, num_words=1000):
    inputs = Input(shape = [maxlen])
    layer = Embedding(input_dim = num_words, output_dim = 64, input_length = maxlen)(inputs)
    layer, state_h, state_c = LSTM(
        units = 64, activation='tanh', recurrent_activation='sigmoid', 
        use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', 
        bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, 
        recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
        kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, 
        dropout=0.0, recurrent_dropout=0.0, return_sequences=False, return_state=True, 
        go_backwards=False, stateful=False, time_major=False, unroll=False)(layer)

    model = Model(inputs=inputs,outputs=layer)
    encoder_state = [state_h, state_c]

    return model, encoder_state


def create_decoder(maxlen=100, num_words=1000):
    decoder_states_inputs = [Input(shape=(256,)), Input(shape=(256,))]
    

    inputs = Input(shape = [maxlen])
    layer = Embedding(input_dim = num_words, output_dim = 64, input_length = maxlen)(inputs)
    layer, state_h, state_c = LSTM(
        units = 64, activation='tanh', recurrent_activation='sigmoid', 
        use_bias=True, kernel_initializer='glorot_uniform', recurrent_initializer='orthogonal', 
        bias_initializer='zeros', unit_forget_bias=True, kernel_regularizer=None, 
        recurrent_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
        kernel_constraint=None, recurrent_constraint=None, bias_constraint=None, 
        dropout=0.0, recurrent_dropout=0.0, return_sequences=True, return_state=True, 
        go_backwards=False, stateful=False, time_major=False, unroll=False)(layer, initial_state=decoder_states_inputs)
    
    decoder_states = [state_h, state_c]
    
    layer = Dense(maxlen, activation='softmax')(layer)
    model = Model(
        inputs = [inputs] + decoder_states_inputs,
        outputs = [layer] + decoder_states)

    return model


if __name__ == "__main__":
    # string = "오늘 저녁 치킨 고?"
    # print(chosung_tokenizer(string))
    tweets = get_tweets()
    chosungs = []
    labels = []
    for tweet in tweets:
    # for texts in ["오늘 저녁 치킨 고?"]:
    #     text = preprocessing(texts)
        text = preprocessing(tweet.full_text)
        labels.extend(["\t " + i + " \n" for i in text])
        chosungs.extend([chosung_tokenizer(j) for j in text])
    

    df = pd.DataFrame(zip(chosungs,labels), columns=["chosung","target"])
    print(df)
    X_train = df.chosung
    y_train = df.target